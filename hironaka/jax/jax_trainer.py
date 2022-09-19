import logging
import time
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Tuple, Union, Optional

import flax
import jax
import jax.numpy as jnp
from jax import pmap
from jax.tree_util import tree_map

import numpy as np
import optax
import yaml
from flax.training import checkpoints
from flax.training.train_state import TrainState
# Funny that jax doesn't work with tensorflow and I have to use PyTorch's version of tensorboard...
from torch.utils.tensorboard import SummaryWriter

from hironaka.jax.net import DenseNet, DenseResNet, CustomNet, PolicyWrapper
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
    policy_value_loss,
)
from hironaka.src import get_newton_polytope_jax, rescale_jax

RollOut = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class JAXTrainer:
    """
    This consolidates the whole training process given a config file. For detailed usage, please see README or check
        out the test (e.g., testJAXTrainer.py).

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

    def __init__(self, key, config: Union[dict, str], dtype=jnp.float32):
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

        if not self.use_cuda:
            jax.config.update("jax_platform_name", "cpu")
        self.default_backend = jax.default_backend()
        self.device_num = len(jax.devices())

        eval_batch_size, max_num_points, dimension = self.eval_batch_size, self.max_num_points, self.dimension
        self.output_dim = {"host": 2 ** dimension - dimension - 1, "agent": dimension}
        policy_keys = {}
        sim_keys = {}
        key, policy_keys["host"], policy_keys["agent"], sim_keys["host"], sim_keys["agent"] = jax.random.split(key,
                                                                                                               num=5)

        for role in ["host", "agent"]:
            if role not in self.config:
                continue
            net = self.net_dict[self.net_type](self.output_dim[role] + 1, net_arch=self.config[role]["net_arch"])
            setattr(
                self,
                f"{role}_policy",
                PolicyWrapper(policy_keys[role], role, (eval_batch_size, max_num_points, dimension), net),
            )
            self.set_optim(role, self.config[role]["optim"])

            # Set up policy functions (policy and evaluation have different batch sizes)
            self.update_eval_policy_fn(role)

            # Use fixed reward function for now.
            setattr(self, f"{role}_reward_fn", get_reward_fn(role))

            setattr(self, f"{role}_state", None)

        if self.config["tensorboard"]["use"]:
            self.log_string = (
                f"{self.config['version_string']}_{datetime.now().year}_{datetime.now().month}"
                f"_{datetime.now().day}_{time.time_ns()}"
            )
            self.summary_writer = SummaryWriter(log_dir=f"{self.config['tensorboard']['work_dir']}/{self.log_string}")

        for role in ["host", "agent"]:
            self.update_sim_loop(role)
            self.update_loss_fn(role)

        self.host_opponent_policy, self.agent_opponent_policy = None, None

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
        policy_wrapper = getattr(self, f"{role}_policy")
        opponent = self._get_opponent(role)
        opp_policy_wrapper = getattr(self, f"{opponent}_policy")

        role_params = policy_wrapper.parameters
        opp_params = opp_policy_wrapper.parameters

        # Generate root state
        keys = jax.random.split(key, num=self.device_num + 1)
        root_state = pmap(partial(self.generate_pts, shape=(self.eval_batch_size, self.max_num_points, self.dimension),
                                  max_value=self.max_value, dtype=self.dtype, rescale=self.scale_observation))(keys[1:])
        if role == "agent":
            coords, _ = pmap(self.get_eval_policy_fn("host"), in_axes=(0, None))(root_state, opp_params)
            batch_decode_from_one_hot = get_batch_decode_from_one_hot(self.dimension)
            root_state = jnp.concatenate([pmap(flatten)(root_state), pmap(batch_decode_from_one_hot)(coords)], axis=-1)
        elif role == "host":
            root_state = pmap(flatten)(root_state)
        else:
            raise ValueError(f"role must be either host or agent. Got {role}.")

        keys = jax.random.split(keys[0], num=self.device_num + 1)
        simulate_output = pmap(sim_fn, in_axes=(0, 0, None, None))(
            keys[1:], root_state, (role_params,), (opp_params,)
        )
        return pmap(partial(self.rollout_postprocess, role=role))(simulate_output)

    def train(self, key: jnp.ndarray, role: str, gradient_steps: int, rollouts: jnp.ndarray, random_sampling=False,
              verbose=0):
        """
        Trains the neural network with a collection of samples ('rollouts', supposedly collected from simulation).
        Parameters:
            key: the PRNG key.
            role: either host or agent.
            gradient_steps: number of gradient steps to take.
            rollouts: observation, policy_logits, values of the form (device_num, batch_size, ...)
            random_sampling: (Optional) whether to do random sampling in the rollouts.
            verbose: (Optional) whether to print out the loss
        """
        opponent = self._get_opponent(role)

        if getattr(self, f"{role}_policy") is None:
            raise RuntimeError(f"opponent {opponent} policy function not initialized.")

        batch_size = self.config[role]["batch_size"]
        state = self.get_state(role)

        p_loss_fn = pmap(self.get_loss_fn(role), in_axes=(None, 0), out_axes=(0, 0))

        sample_size = rollouts[0].shape[0]
        gradient_steps = sample_size // batch_size if not random_sampling else gradient_steps

        for i in range(gradient_steps):
            keys = jax.random.split(key, num=self.device_num + 1)
            if random_sampling:
                sample_idx = pmap(
                    partial(jax.random.randint, shape=(batch_size,), minval=0, maxval=sample_size)
                )(keys[1:])
            else:
                sample_idx = jnp.repeat(
                    jnp.expand_dims(jnp.arange(i * batch_size, (i + 1) * batch_size), axis=0),
                    self.device_num, axis=0)

            sample = pmap(lambda x, y: (x[0][y, :], x[1][y, :], x[2][y]), in_axes=(0, 0), out_axes=(0, 0, 0))(
                rollouts, sample_idx)

            p_loss, p_grad = p_loss_fn(state, sample)
            loss, grad = tree_map(partial(jnp.mean, axis=0), p_loss), tree_map(partial(jnp.mean, axis=0), p_grad)

            state = state.apply_gradients(grads=grad)

            # Tensorboard logging
            if self.config['tensorboard']['use']:
                if state.step % self.config['tensorboard']['log_interval'] == 0:
                    if verbose:
                        self.logger.info(f"Loss: {loss}")

                    self.tensorboard_log_scalar(f"{role}/loss", loss, state.step)
                    self.summary_writer.add_histogram(
                        f"{role}/gradient", np.array(self.layerwise_average(grad["params"], [])), state.step
                    )

                    if self.config["tensorboard"]["layerwise_logging"]:
                        self.tensorboard_log_layers(role, state.params["params"], state.step)

                if state.step % self.config['tensorboard']['validation_interval'] == 0:
                    rhos, details = self.validate(write_tensorboard=True)
                    if verbose:
                        self.logger.info(f"Rhos:\n{rhos}\nGame length histogram:\n{details}")


        # Save the state and parameters
        setattr(self, f"{role}_state", state)
        getattr(self, f"{role}_policy").parameters = state.params

    @staticmethod
    def loss_and_grad(state: TrainState, sample: jnp.ndarray,
                      loss_fn: Callable, max_grad=1) -> Tuple[jnp.ndarray, jnp.ndarray]:
        loss, grads = jax.value_and_grad(partial(compute_loss, loss_fn=loss_fn))(state.params, state.apply_fn, sample)
        grads = jax.tree_util.tree_map(partial(jnp.clip, a_max=max_grad), grads)
        return loss, grads

    def validate(
            self, metric_fn: Optional[Callable] = None, verbose=1, batch_size=10, num_of_loops=10, max_length=None,
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

        spec = (self.max_num_points, self.dimension)

        host_policy_wrapper = self.host_policy
        agent_policy_wrapper = self.agent_policy

        host_fn = jax.jit(
            action_wrapper(partial(host_policy_wrapper.get_apply_fn(batch_size), params=host_policy_wrapper.parameters),
                           None)
        )
        agent_fn = jax.jit(
            action_wrapper(
                partial(agent_policy_wrapper.get_apply_fn(batch_size), params=agent_policy_wrapper.parameters),
                self.dimension
            )
        )

        hosts = [
            host_fn,
            get_host_with_flattened_obs(spec, random_host_fn),
            get_host_with_flattened_obs(spec, zeillinger_fn),
            get_host_with_flattened_obs(spec, all_coord_host_fn),
        ]
        agents = [
            agent_fn,
            partial(random_agent_fn, spec=spec),
            partial(choose_first_agent_fn, spec=spec),
            partial(choose_last_agent_fn, spec=spec),
        ]

        # Pit host network against all and agent network against the rest.
        battle_schedule = [(0, i) for i in range(len(agents))] + [(i, 0) for i in range(1, len(hosts))]

        rhos = []
        details = []
        for pair_idx in battle_schedule:
            key, subkey = jax.random.split(key)

            host, agent = hosts[pair_idx[0]], agents[pair_idx[1]]
            rho, detail = metric_fn(
                host, agent, batch_size=batch_size, num_of_loops=num_of_loops, max_length=max_length,
                write_tensorboard=pair_idx==(0, 0) and write_tensorboard, key=key
            )
            rhos.append(rho)
            details.append(detail)
            if verbose:
                self.logger.info(f"{get_name(host)} vs {get_name(agent)}:")
                self.logger.info(f"  {get_name(metric_fn)}: {rho}")
            if write_tensorboard:
                self.summary_writer.add_scalar(f"{get_name(host)}_v_{get_name(agent)}", rho, self.host_state.step)
                hist = np.concatenate(
                    [np.full((detail[i],), i) for i in range(len(detail) - 1)], axis=0
                )
                self.summary_writer.add_histogram(f"{get_name(host)}_v_{get_name(agent)}/length_histogram",
                                                  hist, self.host_state.step)

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

        take_action = get_take_actions(role="host", spec=spec, rescale_points=self.scale_observation)
        batch_decode = get_batch_decode_from_one_hot(dimension)

        details = [0] * max_length
        for _ in range(num_of_loops):
            key, host_key, agent_key = jax.random.split(key, num=3)

            pts = self.generate_pts(
                host_key,
                (batch_size, max_num_points, dimension),
                self.max_value,
                dtype=self.dtype,
                rescale=self.scale_observation,
            )
            prev_done, done = 0, jnp.sum(get_dones(pts))  # Calculate the finished games
            pts = flatten(pts)

            # For tensorboard logging
            collect_host_actions, collect_agent_actions = [], []

            for step in range(max_length - 1):
                key, host_key, agent_key = jax.random.split(key, num=3)

                details[step] += done - prev_done

                host_action = host(pts, key=host_key)
                coords = batch_decode(host_action).astype(self.dtype)
                axis = jnp.argmax(agent(make_agent_obs(pts, coords), key=agent_key), axis=1).astype(self.dtype)
                pts = take_action(pts, coords, axis)

                prev_done, done = done, jnp.sum(get_dones(pts.reshape((-1, *spec))))  # Update the finished games

                if write_tensorboard:
                    collect_host_actions.append(np.ravel(np.argmax(host_action, axis=1)))
                    collect_agent_actions.append(np.ravel(axis))

            details[max_length - 1] += batch_size - done

            if write_tensorboard:
                self.summary_writer.add_histogram("host_action_distributions",
                                                  np.concatenate(collect_host_actions), self.host_state.step)
                self.summary_writer.add_histogram("agent_action_distributions",
                                                  np.concatenate(collect_agent_actions), self.agent_state.step)

        rho = sum(details[1:]) / sum([i * num for i, num in enumerate(details)])
        return rho, details

    def rollout_postprocess(self, rollouts: RollOut, role: str) -> RollOut:
        """
        Perform postprocessing on the rollout samples. In this default postprocessing, we replace the value_prior
            from the MCTS tree by the ground-truth value depending on the game win/lose.
        Parameters:
            rollouts: the rollout set to be processed. (observations, policy_prior, value_prior).
            role: the current role (agent value is the negative of host value)
        Returns:
            the processed rollouts.
        """
        # Shapes:
        #   obs (b, max_length_game, input_dim)
        #   policy (b, max_length_game, dimension)
        #   value (b, max_length_game)
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
            checkpoints.save_checkpoint(ckpt_dir=path, prefix=f"{role}_", overwrite=True, target=state, step=state.step)

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
                state = checkpoints.restore_checkpoint(
                    ckpt_dir=file_path, target=self.get_state(role), step=step, prefix=f"{role}_"
                )
                setattr(self, f"{role}_state", state)

    # ---------- Below are getter functions for training-related functions ---------- #

    def get_fns(self, role: str, name: str) -> Callable:
        if not hasattr(self, f"{role}_{name}") or getattr(self, f"{role}_{name}") is None:
            self._update_fns(role)
        return getattr(self, f"{role}_{name}")

    def get_eval_policy_fn(self, role: str) -> Callable:
        return self.get_fns(role, "eval_policy_fn")

    def get_train_policy_fn(self, role: str) -> Callable:
        return self.get_fns(role, "train_policy_fn")

    def get_eval_loop(self, role: str) -> Callable:
        return self.get_fns(role, "eval_loop")

    def get_sim_fn(self, role: str) -> Callable:
        return self.get_fns(role, "sim_fn")

    def get_loss_fn(self, role: str) -> Callable:
        return self.get_fns(role, "loss_fn")

    def get_state(self, role: str) -> TrainState:
        state = getattr(self, f"{role}_state")
        if state is None:
            optim = getattr(self, f"{role}_optim")
            policy_wrapper = getattr(self, f"{role}_policy")
            apply_fn = self.get_train_policy_fn(role)
            state = TrainState.create(apply_fn=apply_fn, params=policy_wrapper.parameters, tx=optim)
            setattr(self, f"{role}_state", state)
        return state

    # ---------- Below are either static methods or methods that set its members ---------- #

    def set_optim(self, role: str, optim_config: dict):
        """
        Set up/reset the optimizer.
        """
        optim_name = optim_config["name"]
        optim_args = optim_config["args"]
        setattr(self, f"{role}_optim", self.optim_dict[optim_name](**optim_args))

    def update_eval_policy_fn(self, role: str, jitted=True):
        """
        Update `{role}_eval_policy_fn` based on the current parameters of the network.
        """
        maybe_jit = jax.jit if jitted else lambda x, *_: x
        batch_size = self.eval_batch_size
        setattr(self, f"{role}_eval_policy_fn",
                maybe_jit(getattr(self, f"{role}_policy").get_apply_fn(batch_size),
                          backend='cpu' if self.eval_on_cpu or not self.use_cuda else self.default_backend))

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
            self.get_eval_policy_fn(role),
            action_wrapper(self.get_eval_policy_fn(opponent), dim),
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

    def update_loss_fn(self, role: str, jitted=True, return_function=False) -> Any:
        """
        Update `{role}_loss_fn`.
        """
        loss_fn = (
            jax.jit(partial(self.loss_and_grad, loss_fn=policy_value_loss, max_grad=self.max_grad_norm))
            if jitted
            else partial(self.loss_and_grad, loss_fn=policy_value_loss, max_grad=self.max_grad_norm)
        )
        train_policy_fn = jax.jit(getattr(self, f"{role}_policy").get_apply_fn(self.config[role]["batch_size"]))
        if return_function:
            return loss_fn, train_policy_fn
        else:
            setattr(self, f"{role}_loss_fn", loss_fn)
            setattr(self, f"{role}_train_policy_fn", train_policy_fn)

    def _update_fns(self, role: str):
        self.update_eval_policy_fn(role)
        self.update_sim_loop(role)
        self.update_loss_fn(role)

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

    @staticmethod
    def generate_pts(key: jnp.ndarray, shape: Tuple, max_value: int, dtype=jnp.float32, rescale=True) -> jnp.ndarray:
        pts = jax.random.randint(key, shape, 0, max_value).astype(dtype)
        pts = jnp.where(rescale, rescale_jax(get_newton_polytope_jax(pts)),
                        get_newton_polytope_jax(pts))
        return pts
