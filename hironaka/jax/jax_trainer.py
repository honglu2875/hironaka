import logging
import time
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Tuple, Union, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import pmap

# Funny that jax doesn't work with tensorflow well, and using PyTorch's tensorboard is the most convenient...
from torch.utils.tensorboard import SummaryWriter

from hironaka.jax.net import DenseNet, DenseResNet, CustomNet, get_apply_fn
from hironaka.jax.players import (
    all_coord_host_fn,
    choose_first_agent_fn,
    choose_last_agent_fn,
    get_host_with_flattened_obs,
    random_agent_fn,
    random_host_fn,
    zeillinger_fn,
)
from hironaka.jax.simulation_fn import get_evaluation_loop, get_single_thread_simulation
from hironaka.jax.util import (
    action_wrapper,
    calculate_value_using_reward_fn,
    compute_loss,
    flatten,
    get_batch_decode_from_one_hot,
    get_done_from_flatten,
    get_dones,
    get_name,
    get_reward_fn,
    get_take_actions,
    make_agent_obs,
    policy_value_loss, generate_pts,
)

RollOut = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class JAXTrainer:
    """
    This class consolidates the whole training process given a config file. One training setup <-> one class.
    For what it is worth, the class caches functions that have compilation overhead. Although pmap is already doing a
        great job at caching its input functions, there are still occasions where manual caching is necessary (e.g.,
        in `pmap(partial(...))`, as `partial` creates a new function on every call). One can certainly have a lot
        of small functions and factory functions, and apply lru_cache everywhere, but I personally find it slightly
        easier to track everything as members of a class.
    JAX is ultimately following the idea of functional programming. I look to avoid fancy usage of classes and only
        stick to the very beginner-friendly class-method-attribute style.
    The whole JAXTrainer life cycle is designed to do the following:
    - Read the config and load all parameters
    - One calls `JAXTrainer.simulate` to do
        1. Generate policy logit and value estimates based on the neural network.
        2. Perform MCTS to improve the policy and value (currently only makes use of `mctx.gumbel_muzero_policy`)
        3. Continue the evaluation (policy improving via MCTS) simulation (perform the actual game step based on the
            improved policy) cycle, and return them as training samples.
    - One calls `JAXTrainer.train` to train the neural network for certain amount of gradient steps.
    - Repeat the simulation-training cycle, or change some parameters here and there, call JAXTrainer.update_fns and
        perform some trainings.
    """
    eval_batch_size: int
    max_num_points: int
    dimension: int
    max_length_game: int
    max_value: int
    max_grad_norm: float
    scale_observation: bool

    # If use_cuda is True, we try to map to all available GPUs.
    use_cuda: bool

    version_string: str
    net_type: str

    num_evaluations: int
    eval_on_cpu: bool
    max_num_considered_actions: int
    discount: float

    optim_dict = {"adam": optax.adam, "adamw": optax.adamw, "sgd": optax.sgd}
    net_dict = {"dense_resnet": DenseResNet,
                "dense": DenseNet,
                "custom": CustomNet}

    def __init__(self, key: jnp.ndarray, config: Union[dict, str], dtype=jnp.float32, loss_fn=policy_value_loss,
                 host_feature_fn=None, agent_feature_fn=None):
        self.logger = logging.getLogger(__class__.__name__)

        if isinstance(config, str):
            with open(config, "r") as stream:
                self.config = yaml.safe_load(stream)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be either a string or a dict. Got {type(config)}.")

        config_keys = [
            "eval_batch_size",
            "max_num_points",
            "dimension",
            "max_length_game",
            "max_value",
            "max_grad_norm",
            "scale_observation",
            "use_cuda",
            "version_string",
            "net_type",
            "num_evaluations",
            "eval_on_cpu",
            "max_num_considered_actions",
            "discount"
        ]
        for config_key in config_keys:
            setattr(self, config_key, self.config[config_key])

        self.dtype = dtype
        self.loss_fn = loss_fn
        self.host_feature_fn=host_feature_fn
        self.agent_feature_fn=agent_feature_fn

        if not self.use_cuda:
            jax.config.update("jax_platform_name", "cpu")
        self.default_backend = jax.default_backend()
        self.device_num = len(jax.devices())

        eval_batch_size, max_num_points, dimension = self.eval_batch_size, self.max_num_points, self.dimension
        self.input_dim = {"host": max_num_points * dimension, "agent": max_num_points * dimension + dimension}
        self.output_dim = {"host": 2 ** dimension - dimension - 1, "agent": dimension}
        sim_keys = {}
        key, sim_keys["host"], sim_keys["agent"] = jax.random.split(key, num=3)

        for role in ["host", "agent"]:
            if role not in self.config:
                continue

            # Note: self.output_dim means the dimension of policy vectors only.
            net = self.net_dict[self.net_type](self.output_dim[role], net_arch=self.config[role]["net_arch"])
            setattr(self, f"{role}_model", net)
            self.set_optim(role, self.config[role]["optim"])

            # Use a fixed reward function for now.
            setattr(self, f"{role}_reward_fn", get_reward_fn(role))
            setattr(self, f"{role}_state", None)

            self.update_policy_fn(role)

        if self.config["tensorboard"]["use"]:
            self.log_string = (
                f"{self.config['version_string']}_{datetime.now().year}_{datetime.now().month}"
                f"_{datetime.now().day}_{time.time_ns()}"
            )
            self.summary_writer = SummaryWriter(log_dir=f"{self.config['tensorboard']['work_dir']}/{self.log_string}")

        for role in ["host", "agent"]:
            self.update_sim_loop(role)

            # get_state will initialize '{role}_state'
            key, subkey = jax.random.PRNGKey(key)
            self.get_state(role, subkey)

        self.host_opponent_policy, self.agent_opponent_policy = None, None
        self.hosts_agents_for_validation = {}  # use to cache the lists of host and agent policy functions

    def simulate(self, key: jnp.ndarray, role: str) -> RollOut:
        """
        Performing a simulation. The core is nothing but calling `{role}_sim_fn`
        Parameters:
            key: the PRNG key.
            role: host or agent.
        Returns:
            obs, target_policy, target_value
        """
        sim_fn = self.get_sim_fn(role)
        opponent = self._get_opponent(role)

        role_train_state = self.get_state(role)
        opp_train_state = self.get_state(opponent)

        # Generate root state
        keys = jax.random.split(key, num=self.device_num + 1)
        root_state = generate_pts(keys[1:], (self.eval_batch_size, self.max_num_points, self.dimension),
                                  self.max_value, self.dtype, self.scale_observation)

        if role == "agent":
            # Get host coordinates, flatten and concatenate to make agent observations
            coords, _ = pmap(self.get_policy_fn("host"))(root_state, opp_train_state.params)
            batch_decode_from_one_hot = get_batch_decode_from_one_hot(self.dimension)
            root_state = jnp.concatenate([pmap(flatten)(root_state), pmap(batch_decode_from_one_hot)(coords)], axis=-1)
        elif role == "host":
            # Host observations are merely flattened arrays
            root_state = pmap(flatten)(root_state)
        else:
            raise ValueError(f"role must be either host or agent. Got {role}.")

        keys = jax.random.split(keys[0], num=self.device_num + 1)
        simulate_output = pmap(sim_fn)(
            keys[1:], root_state, (role_train_state.params,), (opp_train_state.params,)
        )
        return pmap(self.rollout_postprocess, static_broadcasted_argnums=1)(simulate_output, role)

    def train(self, key: jnp.ndarray, role: str, gradient_steps: int, rollouts: jnp.ndarray, mask=None,
              random_sampling=False, verbose=0):
        """
        Trains the neural network with a collection of samples ('rollouts', supposedly collected from simulation).
        Parameters:
            key: the PRNG key.
            role: either host or agent.
            gradient_steps: number of gradient steps to take.
            rollouts: observation, policy_logits, values of the form (device_num, sample_size, ...)
            mask: (Optional) the mask on rollout, so that we ignore some samples. Must be of the shape
                (device_num, sample_size) with bool entries.
            random_sampling: (Optional) whether to do random sampling or cut out contiguous batches in the rollouts.
            verbose: (Optional) whether to print out the loss
        """
        # On each device, choose `batch_size` amount of rollouts out of `sample_size`, and perform regression.
        batch_size = self.config[role]["batch_size"]
        sample_size = rollouts[0].shape[1]
        # The shape of the mask should be (device_num, sample_size) which is the same as value predictions in rollouts.
        mask = jnp.ones_like(rollouts[2], dtype=jnp.float32) if mask is None else mask.astype(jnp.float32)
        state = self.get_state(role)

        for i in range(gradient_steps):
            keys = jax.random.split(key, num=self.device_num + 1)
            key = keys[0]

            if random_sampling:
                sample_idx = pmap(jax.random.choice, static_broadcasted_argnums=(2, 3))(
                    keys[1:], flax.jax_utils.replicate(jnp.arange(sample_size)), (batch_size,), True, mask
                )
            else:
                sample_idx = jnp.repeat(
                    jnp.expand_dims(jnp.arange(i * batch_size, (i + 1) * batch_size) % sample_size, axis=0),
                    self.device_num, axis=0)

            sample = p_get_index(rollouts, sample_idx)

            state, loss, grad = train_loop(state, sample, self.loss_fn, self.max_grad_norm)

            # Tensorboard logging
            if self.config['tensorboard']['use']:
                if state.step[0] % self.config['tensorboard']['log_interval'] == 0:
                    if verbose:
                        self.logger.info(f"Loss: {loss}")

                    self.tensorboard_log_scalar(f"{role}/loss", loss, state.step[0])
                    self.summary_writer.add_histogram(
                        f"{role}/gradient", np.array(self.layerwise_average(grad["params"], [])), state.step[0]
                    )

                    if self.config["tensorboard"]["layerwise_logging"]:
                        self.tensorboard_log_layers(role, state.params["params"], state.step[0])

                if state.step[0] % self.config['tensorboard']['validation_interval'] == 0:
                    rhos, details = self.validate(write_tensorboard=True)
                    if verbose:
                        self.logger.info(f"Rhos:\n{rhos}\nGame length histogram:\n{details}")

        # Save the state
        setattr(self, f"{role}_state", state)

    def validate(
            self, metric_fn: Optional[Callable] = None, verbose=1, batch_size=50, num_of_loops=10, max_length=None,
            write_tensorboard=False, key=None
    ) -> Tuple[List, List]:
        """
        This method validates the current host and agent by pitting them against pre-defined strategies including:
            host: random_host_fn, all_coord_host_fn, zeillinger_fn
            agent: random_agent_fn, choose_first_agent_fn, choose_last_agent_fn
        For the details, please check out the `battle_schedule` variable inside.
        Parameters:
            metric_fn: (Optional) the function that computes metrics given a host-agent pair. Defaults to calculate rho.
            verbose: (Optional) whether to put the result into the logger.
            batch_size: (Optional) the batch size of samples to feed into players.
            num_of_loops: (Optional) number of loops in computing the metric. To be fed into the metric_fn.
            max_length: (Optional) max length of games. Default to self.max_length_game.
            write_tensorboard: (Optional) write the results to tensorboard.
            key: (Optional) the PRNG random key. Default to the key from the seed `time.time_ns()`.
        Returns:
            a tuple of a list of computed metric(rho) and a list of details (histogram of game lengths)
        """
        key = jax.random.PRNGKey(time.time_ns()) if key is None else key
        max_length = self.max_length_game if max_length is None else max_length
        metric_fn = self.compute_rho if metric_fn is None else metric_fn

        hosts, agents = self.get_hosts_agents_for_validation(batch_size)

        # Pit host network against all and agent network against the rest.
        battle_schedule = [(0, i) for i in range(len(agents))] + [(i, 0) for i in range(1, len(hosts))]

        rhos = []
        details = []

        for pair_idx in battle_schedule:
            key, subkey = jax.random.split(key)

            host, agent = hosts[pair_idx[0]], agents[pair_idx[1]]
            rho, detail = metric_fn(
                host, agent, batch_size=batch_size, num_of_loops=num_of_loops, max_length=max_length,
                write_tensorboard=pair_idx == (0, 0) and write_tensorboard, key=key
            )
            rhos.append(rho)
            details.append(detail)
            if verbose:
                self.logger.info(f"{get_name(host)} vs {get_name(agent)}:")
                self.logger.info(f"  {get_name(metric_fn)}: {rho}")
            if write_tensorboard:
                self.summary_writer.add_scalar(f"{get_name(host)}_v_{get_name(agent)}",
                                               float(rho), self.get_state('host').step[0])
                if sum(detail) > 0:
                    hist = np.concatenate(
                        [np.full((detail[i],), i) for i in range(len(detail) - 1)], axis=0
                    )
                    self.summary_writer.add_histogram(f"{get_name(host)}_v_{get_name(agent)}/length_histogram",
                                                      hist, self.get_state('host').step[0])

        return rhos, details

    def compute_rho(
            self, host: Callable, agent: Callable, batch_size=None, num_of_loops=10, max_length=None,
            write_tensorboard=False, key=None
    ) -> Tuple[float, List]:
        """
        Calculate the rho number between the host and agent.
        Parameters:
            host: a function that takes in point state and returns the one-hot action vectors.
            agent: a function that takes in point state and returns the one-hot action vectors.
            batch_size: (Optional) the batch size. Default to self.eval_batch_size.
            num_of_loops: (Optional) the number of times to run a batch of points. Default to 10.
            max_length: (Optional) the maximal length of game. Default is self.max_length_game
            write_tensorboard: (Optional) write the histogram of host/agent policy argmax
            key: (Optional) the PRNG random key (if None, will use time.time_ns() to seed a key)
        Returns:
            a tuple of the rho number and a list of game details (histogram of game lengths).
        """
        key = jax.random.PRNGKey(time.time_ns()) if key is None else key
        max_length = self.max_length_game if max_length is None else max_length
        batch_size = self.eval_batch_size if batch_size is None else batch_size

        max_num_points, dimension = self.max_num_points, self.dimension
        spec = (max_num_points, dimension)

        take_action = pmap(get_take_actions(role="host", spec=spec, rescale_points=self.scale_observation))
        batch_decode = pmap(get_batch_decode_from_one_hot(dimension))
        p_reshape = pmap(jnp.reshape, static_broadcasted_argnums=1)

        details = [0] * max_length
        for _ in range(num_of_loops):
            keys = jax.random.split(key, num=1 + self.device_num)
            key = keys[0]

            pts = generate_pts(keys[1:], (batch_size, self.max_num_points, self.dimension),
                               self.max_value, self.dtype, self.scale_observation)

            prev_done, done = 0, jnp.sum(pmap(get_dones)(pts))  # Calculate the finished games
            pts = pmap(flatten)(pts)

            # For tensorboard logging
            collect_host_actions, collect_agent_actions = [], []

            for step in range(max_length - 1):
                keys = jax.random.split(key, num=2 * self.device_num + 1)
                key = keys[0]
                host_keys = keys[1:self.device_num + 1]
                agent_keys = keys[self.device_num + 1:]

                details[step] += done - prev_done
                host_action = host(pts, key=host_keys)
                coords = batch_decode(host_action).astype(self.dtype)
                agent_obs = pmap(make_agent_obs)(pts, coords)
                axis = jnp.argmax(agent(agent_obs, key=agent_keys), axis=2).astype(self.dtype)
                pts = take_action(pts, coords, axis)

                # Have to sync by summing results across devices. prev_done and done are single scalars.
                prev_done, done = done, jnp.sum(pmap(get_dones)(
                    p_reshape(pts, (-1, *spec))
                ))  # Update the finished games

                if write_tensorboard:
                    collect_host_actions.append(np.array(jnp.ravel(np.argmax(host_action, axis=2))))
                    collect_agent_actions.append(np.array(jnp.ravel(axis)))

            details[max_length - 1] += batch_size - done

            if write_tensorboard:
                self.summary_writer.add_histogram("host_action_distributions",
                                                  np.concatenate(collect_host_actions),
                                                  self.get_state('host').step[0])
                self.summary_writer.add_histogram("agent_action_distributions",
                                                  np.concatenate(collect_agent_actions),
                                                  self.get_state('agent').step[0])

        rho = sum(details[1:]) / sum([i * num for i, num in enumerate(details)])
        return rho, details

    def rollout_postprocess(self, rollouts: RollOut, role: str) -> RollOut:
        """
        Perform postprocessing on the rollout samples. In this default postprocessing, we replace the value_prior
            from the MCTS tree by the ground-truth value depending on the game win/lose.
        Parameters:
            rollouts: the rollout set to be processed. (observations, policy_prior, value_prior). Shapes are below.
                - obs (b, max_length_game, input_dim), where input_dim is the dim of all flattened features.
                - policy (b, max_length_game, dimension)
                - value (b, max_length_game)
            role: the current role (agent value is the negative of host value)
        Returns:
            the processed rollouts. The return shapes are below.
                - obs (b * max_length_game, input_dim)
                - policy (b * max_length_game, dimension)
                - value (b * max_length_game,)
        """
        obs, policy, value = rollouts
        batch_size, max_length_game = obs.shape[0], obs.shape[1]
        value_dtype = value.dtype

        reward_fn = getattr(self, f"{role}_reward_fn")
        done = get_done_from_flatten(obs, role, self.dimension)  # (b, max_length_game)
        prev_done = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=bool), done[:, :-1]], axis=1)
        value = calculate_value_using_reward_fn(done, prev_done, self.discount, max_length_game, reward_fn)
        value = jnp.ravel(value).astype(value_dtype)

        obs = obs.reshape((-1, obs.shape[2]))
        policy = policy.reshape((-1, policy.shape[2]))

        return obs, policy, value

    def tensorboard_log_scalar(self, name: str, number: float, step: int):
        if hasattr(self, "summary_writer") and self.summary_writer is not None:
            self.summary_writer.add_scalar(name, float(number), int(step))

    def tensorboard_log_layers(self, name: str, params, step: int):
        """
        Recursively log each layer.
        """
        if isinstance(params, (flax.core.FrozenDict, dict)):
            for key, item in params.items():
                self.tensorboard_log_layers(f"{name}/{key}", item, step)
        else:
            self.summary_writer.add_histogram(name, np.array(params), step)

    @staticmethod
    def layerwise_average(grads, avg_lst: List) -> List:
        """
        Recursively compute the average of each layer, and put them together into a list.
        """
        if isinstance(grads, (flax.core.FrozenDict, dict)):
            for item in grads.values():
                JAXTrainer.layerwise_average(item, avg_lst)
        else:
            avg_lst.append(jnp.mean(grads))
        return avg_lst

    def save_checkpoint(self, path: str):
        for role in ["host", "agent"]:
            state = self.get_state(role)
            checkpoints.save_checkpoint(ckpt_dir=path, prefix=f"{role}_", overwrite=True,
                                        target=flax.jax_utils.unreplicate(state), step=state.step[0])

    def load_checkpoint(self, path: str, step=None):
        """
        Load checkpoint.
        Parameters:
            path: the checkpoint path.
            step: (Optional) the step to restore. If None, restore the latest.
        """
        for role in ["host", "agent"]:
            file_path = checkpoints.latest_checkpoint(ckpt_dir=path, prefix=f"{role}_") if step is None else path
            if file_path is not None:
                state = flax.jax_utils.replicate(checkpoints.restore_checkpoint(
                    ckpt_dir=file_path, target=self.get_state(role), step=step, prefix=f"{role}_"
                ))
                setattr(self, f"{role}_state", state)
                self.update_fns(role)

    # ---------- Below are getter functions for functions that are cached inside the class after first call ---------- #

    def get_fns(self, role: str, name: str) -> Callable:
        if not hasattr(self, f"{role}_{name}") or getattr(self, f"{role}_{name}") is None:
            self.update_fns(role)
        return getattr(self, f"{role}_{name}")

    def get_policy_fn(self, role: str) -> Callable:
        return self.get_fns(role, "policy_fn")

    def get_eval_loop(self, role: str) -> Callable:
        return self.get_fns(role, "eval_loop")

    def get_sim_fn(self, role: str) -> Callable:
        return self.get_fns(role, "sim_fn")

    def get_state(self, role: str, key=None) -> TrainState:
        """
        TrainState is a pytree object. In our case, the tensors inside are already SharedDeviceArray where the first
            dimension is a device axis (see `flax.jax_utils.replicate`).
        """
        state = getattr(self, f"{role}_state", None)
        if state is None:
            key = jax.random.PRNGKey(time.time_ns()) if key is None else key
            optim = getattr(self, f"{role}_optim")
            net = getattr(self, f"{role}_model")
            apply_fn = self.get_policy_fn(role)
            parameters = net.init(key, jnp.ones((1, self.input_dim[role])))
            state = flax.jax_utils.replicate(
                TrainState.create(apply_fn=apply_fn, params=parameters, tx=optim))
            setattr(self, f"{role}_state", state)
        return state

    def get_hosts_agents_for_validation(self, batch_size: int, force_update=False):
        """
        Get a set of host functions and a set of agent functions. They are policies of different strategies and will
            be used to fight against each other. We cache those functions after the first access.
        Parameters:
            batch_size: the batch size of the input observations.
            force_update: force an update on all the policy functions.
        Returns:
            a list of host functions and a list of agent functions.
        """
        if batch_size not in self.hosts_agents_for_validation or force_update:
            spec = (self.max_num_points, self.dimension)
            host_apply_fn = pmap(action_wrapper(self.host_policy_fn, None))
            agent_apply_fn = pmap(action_wrapper(self.agent_policy_fn, self.dimension))

            def host_fn(x, **_):
                return host_apply_fn(x, self.host_state.params)

            def agent_fn(x, **_):
                return agent_apply_fn(x, self.agent_state.params)

            def expose_name(func):
                if hasattr(func, "func"):
                    func.__name__ = func.func.__name__
                return func

            hosts = [
                host_fn,
                jax.pmap(get_host_with_flattened_obs(spec, random_host_fn)),
                jax.pmap(get_host_with_flattened_obs(spec, zeillinger_fn)),
                jax.pmap(get_host_with_flattened_obs(spec, all_coord_host_fn)),
            ]
            agents = [
                agent_fn,
                jax.pmap(expose_name(partial(random_agent_fn, spec=spec))),
                jax.pmap(expose_name(partial(choose_first_agent_fn, spec=spec))),
                jax.pmap(expose_name(partial(choose_last_agent_fn, spec=spec))),
            ]
            self.hosts_agents_for_validation[batch_size] = hosts, agents

        return self.hosts_agents_for_validation[batch_size]

    # ---------- Below are either static methods or methods that set its members ---------- #

    def set_optim(self, role: str, optim_config: dict):
        """
        Set up/reset the optimizer.
        """
        optim_name = optim_config["name"]
        optim_args = optim_config["args"]
        setattr(self, f"{role}_optim", self.optim_dict[optim_name](**optim_args))

    def update_sim_loop(self, role: str, jitted=True, return_function=False) -> Any:
        """
        Update `{role}_eval_loop` and `{role}_sim_fn`.
        """
        opponent = self._get_opponent(role)
        dim = self.dimension if role == "host" else None
        maybe_jit = jax.jit if jitted else lambda x, *_: x
        simulation_config = {
            "eval_batch_size": self.eval_batch_size,
            "max_num_points": self.max_num_points,
            "dimension": self.dimension,
            "max_length_game": self.max_length_game,
            "max_value": self.max_value,
            "scale_observation": self.scale_observation,
        }

        eval_loop = get_evaluation_loop(
            role,
            getattr(self, f"{role}_policy_fn"),
            action_wrapper(getattr(self, f"{opponent}_policy_fn"), dim),
            getattr(self, f"{role}_reward_fn"),
            spec=(self.max_num_points, self.dimension),
            num_evaluations=self.num_evaluations,
            max_depth=self.max_length_game,
            max_num_considered_actions=self.max_num_considered_actions,
            discount=self.discount,
            rescale_points=self.scale_observation,
            dtype=self.dtype,
        )
        sim_fn = maybe_jit(get_single_thread_simulation(role, eval_loop, config=simulation_config, dtype=self.dtype),
                           backend='cpu' if self.eval_on_cpu or not self.use_cuda else jax.default_backend())
        if return_function:
            return eval_loop, sim_fn
        else:
            setattr(self, f"{role}_eval_loop", eval_loop)
            setattr(self, f"{role}_sim_fn", sim_fn)

    def update_policy_fn(self, role: str, return_function=False) -> Any:
        """
        Update `{role}_policy_fn`.
        """
        feature_fn = getattr(self, f"{role}_feature_fn")
        model = getattr(self, f"{role}_model")
        policy_fn = get_apply_fn(role, model, (self.max_num_points, self.dimension), feature_fn=feature_fn)
        if return_function:
            return policy_fn
        else:
            setattr(self, f"{role}_policy_fn", policy_fn)

    def update_fns(self, role: str):
        self.update_sim_loop(role)
        self.update_policy_fn(role)

    def purge_state(self, role: str):
        """
        Removes the training state of host or agent. TrainState object stores information of an ongoing sequence of
            gradient steps. It will not affect parameters that are already saved in `self.host_policy` and
            `self.agent_policy`.
        """
        setattr(self, f"{role}_state", None)

    @staticmethod
    def _get_opponent(role: str) -> str:
        if role == "host":
            return "agent"
        elif role == "agent":
            return "host"
        else:
            raise ValueError(f"role must be either host or agent. Got {role}.")


@partial(pmap, axis_name='d', static_broadcasted_argnums=(2, 3))
def train_loop(state: TrainState, sample: jnp.ndarray,
               loss_fn: Callable, max_grad=1) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    loss, grad = jax.value_and_grad(partial(compute_loss, loss_fn=loss_fn))(state.params, state.apply_fn, sample)
    grads = jax.tree_util.tree_map(partial(jnp.clip, a_max=max_grad), grad)

    loss, grad = jax.lax.pmean(loss, 'd'), jax.lax.pmean(grad, 'd')
    state = state.apply_gradients(grads=grad)
    return state, loss, grads


# A helper function that selects the indices of the 3-item tuple
#   (used in the selection of training data of the (observation, policy, value)-tuple)
p_get_index = pmap(lambda x, y: (x[0][y, :], x[1][y, :], x[2][y]))
