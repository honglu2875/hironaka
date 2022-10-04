import logging
import time
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import flax
import flax.linen as nn
import numpy as np
import optax
import yaml
from flax.training import checkpoints
from flax.training.train_state import TrainState

# Funny that jax doesn't work with tensorflow well, and using PyTorch's tensorboard is the most convenient...
from torch.utils.tensorboard import SummaryWriter

import jax
import jax.numpy as jnp
from hironaka.jax.net import CustomNet, DenseNet, DenseResNet, get_apply_fn
from hironaka.jax.players import (
    all_coord_host_fn,
    choose_first_agent_fn,
    choose_last_agent_fn,
    get_host_with_flattened_obs,
    random_agent_fn,
    random_host_fn,
    zeillinger_fn,
)
from hironaka.jax.simulation_fn import get_evaluation_loop, get_simulation
from hironaka.jax.util import (
    action_wrapper,
    apply_agent_action_mask,
    calculate_value_using_reward_fn,
    flatten,
    generate_pts,
    get_done_from_flatten,
    get_dones,
    get_name,
    get_reward_fn,
    get_take_actions,
    make_agent_obs,
    mcts_wrapper, get_feature_fn,
    safe_clip_grads,
)
from hironaka.jax.host_action_preprocess import get_batch_decode_from_one_hot
from hironaka.jax.loss import compute_loss, policy_value_loss, clip_log
from jax import pmap

RollOut = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


# A helper function that selects the indices of the 3-item tuple
#   (used in the selection of training data of the (observation, policy, value)-tuple)
p_get_index = pmap(lambda x, y: (x[0][y, :], x[1][y, :], x[2][y]))


@partial(pmap, axis_name="d", static_broadcasted_argnums=(2, 3, 4))
def p_train_loop(
    state: TrainState, sample: jnp.ndarray, apply_fn: Callable, loss_fn: Callable, max_grad=1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    A pmap compiled training loop. Note that the last 3 arguments are statically broadcast, and the first two are
        already spread to each device.
    Parameters:
        state: the TrainState object including the parameters and the optimizer. Entries copied to each device.
        sample: the self-play samples. (observation, policy, value), pmapped on each element.
        apply_fn: the apply function of the network. Will apply parameters in `state` on `sample`.
        loss_fn: the loss function.
        max_grad: maximum norm of a gradient. Used in clipping.
    Returns:
        (new_state, loss, grads)
    """
    loss, grad = jax.value_and_grad(partial(compute_loss, loss_fn=loss_fn))(state.params, apply_fn, sample)
    grad = safe_clip_grads(grad, max_grad)
    # Use pmean to sync loss and grad on each device.
    loss, grad = jax.lax.pmean(loss, "d"), jax.lax.pmean(grad, "d")
    # Simultaneously update parameters on each device by the same gradient. In some cases slightly better than reducing
    #   gradient to one device, calculate, and then broadcast again.
    state = state.apply_gradients(grads=grad)
    return state, loss, grad


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
    net_dict = {"dense_resnet": DenseResNet, "dense": DenseNet, "custom": CustomNet}

    host_model: nn.Module
    agent_model: nn.Module
    host_reward_fn: Callable
    agent_reward_fn: Callable
    host_feature_fn: Callable
    agent_feature_fn: Callable
    host_state: TrainState
    agent_state: TrainState
    host_eval_loop: Callable
    agent_eval_loop: Callable
    host_eval_loop_as_opp: Callable
    agent_eval_loop_as_opp: Callable
    host_sim_fn: Callable
    agent_sim_fn: Callable
    host_mcts_policy: Callable
    agent_mcts_policy: Callable
    unified_eval_loop: Callable
    unified_sim_fn: Callable

    def __init__(
        self,
        key: jnp.ndarray,
        config: Union[dict, str],
        dtype=jnp.float32,
        loss_fn=policy_value_loss,
        host_feature_fn=None,
        agent_feature_fn=None,
    ):
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
            "num_evaluations_as_opponent",
            "eval_on_cpu",
            "max_num_considered_actions",
            "discount",
        ]
        for config_key in config_keys:
            setattr(self, config_key, self.config[config_key])

        self.dtype = dtype
        self.loss_fn = loss_fn

        self.host_feature_fn = host_feature_fn if host_feature_fn is not None else \
            get_feature_fn('host', (self.max_num_points, self.dimension), scale_observation=self.scale_observation)
        self.agent_feature_fn = agent_feature_fn if agent_feature_fn is not None else \
            get_feature_fn('agent', (self.max_num_points, self.dimension), scale_observation=self.scale_observation)

        if not self.use_cuda:
            jax.config.update("jax_platform_name", "cpu")
        self.default_backend = jax.default_backend()
        self.device_num = len(jax.devices())

        _, max_num_points, dimension = self.eval_batch_size, self.max_num_points, self.dimension
        self.input_dim = {"host": max_num_points * dimension, "agent": max_num_points * dimension + dimension}
        self.output_dim = {"host": 2**dimension - dimension - 1, "agent": dimension}
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
            self.update_eval_sim_and_mcts_policy(role)

            # get_state will initialize '{role}_state'
            key, subkey = jax.random.split(key)
            self.get_state(role, subkey)

        self.host_opponent_policy, self.agent_opponent_policy = None, None
        # Used in caching the lists of host and agent policy functions
        self.cached_hosts_agents_for_validation = {}
        # Used in self.simulate, only when mcts policy is used. Sometimes we want to use old compiled functions
        # ignoring the fact that they contain old parameters to avoid compilation overhead.
        self._cached_sim_fn = {}
        # Used when the training needs to save the best performing player
        self.best_against_choose_first, self.best_against_choose_last, self.best_against_zeillinger = 0, 0, jnp.inf

    def simulate(self, key: jnp.ndarray, role: str, use_mcts_policy=False, use_unified_tree=False) -> RollOut:
        """
        Performing a simulation. The core is nothing but calling `{role}_sim_fn`
        Parameters:
            key: the PRNG key.
            role: host or agent.
            use_mcts_policy: (Optional) use MCTS policy functions to simulate.
                - Note that this is the real key of AlphaZero style MCTS. But we default to False first as the resource
                    consumption is massive for every simulation. If one only uses policy network output, the result
                    is in fact pretty good. It is worth to compare or mix the two ways.
            use_unified_tree: (Optional) use the unified MC tree search. In this case, role will be disregarded. Also,
                the host observation will be padded to the same length as agent (the last `dimension`-entries),
                and the agent action number will be padded to the host's number (`2**dimension-dimension-1`) with
                by -jnp.inf in terms of action logits.
        Returns:
            obs, target_policy, target_value
        """
        if use_mcts_policy:
            sim_fn = self.get_mcts_sim_fn(role)
        elif use_unified_tree:
            sim_fn = self.unified_sim_fn
        else:
            sim_fn = self.get_sim_fn(role)
        opponent = self._get_opponent(role)

        if use_unified_tree:
            role_train_state = self.get_state('host')
            opp_train_state = self.get_state('agent')
        else:
            role_train_state = self.get_state(role)
            opp_train_state = self.get_state(opponent)

        # Generate root state
        # - note: rescale is set to False as rescaling is put into feature functions.
        keys = jax.random.split(key, num=self.device_num + 1)
        root_state = generate_pts(
            keys[1:],
            (self.eval_batch_size, self.max_num_points, self.dimension),
            self.max_value,
            self.dtype,
            False,
        )

        if role == "agent":
            # Get host coordinates, flatten and concatenate to make agent observations
            coords, _ = pmap(self.get_policy_fn("host"))(root_state, self.get_state('host').params)
            batch_decode_from_one_hot = get_batch_decode_from_one_hot(self.dimension)
            coordinate_mask = pmap(batch_decode_from_one_hot)(coords)
            root_state = jnp.concatenate([pmap(flatten)(root_state), coordinate_mask], axis=-1)
        elif role == "host":
            # Host observations are merely flattened arrays
            root_state = pmap(flatten)(root_state)
            if use_unified_tree:
                # Pad zeros to the length of agent observation
                root_state = jnp.concatenate(
                    [
                        root_state,
                        flax.jax_utils.replicate(jnp.zeros((self.eval_batch_size, self.dimension)))
                    ], axis=-1
                )
        else:
            raise ValueError(f"role must be either host or agent. Got {role}.")

        keys = jax.random.split(keys[0], num=self.device_num + 1)

        role_fn_args = (role_train_state.params,)
        opp_fn_args = (opp_train_state.params,) if use_unified_tree else (opp_train_state.params, role_train_state.params)
        simulate_output = pmap(sim_fn)(
            keys[1:], root_state, role_fn_args, opp_fn_args
        )
        return pmap(self.rollout_postprocess, static_broadcasted_argnums=(1, 2))(simulate_output, role, use_unified_tree)

    def train(
        self,
        key: jnp.ndarray,
        role: str,
        gradient_steps: int,
        rollouts: RollOut,
        mask=None,
        random_sampling=False,
        verbose=0,
        save_best=True,
    ):
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
            verbose: (Optional) whether to print out the loss.
            save_best: (Optional) whether to save the best model. Update only if
                host - both higher against choose-first and choose-last,
                agent - higher against zeillinger.
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
                    self.device_num,
                    axis=0,
                )

            sample = p_get_index(rollouts, sample_idx)
            apply_fn = self.get_apply_fn(role)
            state, loss, grad = p_train_loop(state, sample, apply_fn, self.loss_fn, self.max_grad_norm)

            # Save the state
            setattr(self, f"{role}_state", state)

            # Tensorboard logging
            if self.config["tensorboard"]["use"]:
                if state.step[0] % self.config["tensorboard"]["log_interval"] == 0:
                    if verbose:
                        self.logger.info(f"Loss: {loss[0]}")

                    self.tensorboard_log_scalar(f"{role}/loss", loss[0], state.step[0])
                    self.summary_writer.add_histogram(
                        f"{role}/gradient", np.array(self.layerwise_average(grad["params"], [])), state.step[0]
                    )

                    if self.config["tensorboard"]["layerwise_logging"]:
                        self.tensorboard_log_layers(role, state.params["params"], state.step[0])

                if state.step[0] % self.config["tensorboard"]["validation_interval"] == 0:
                    rhos, details = self.validate(write_tensorboard=True)
                    if verbose:
                        self.logger.info(f"Rhos:\n{rhos}\nGame length histogram:\n{details}")
                    if save_best:
                        if rhos[2] > self.best_against_choose_first and rhos[3] > self.best_against_choose_last:
                            self.save_checkpoint(f'{self.version_string}/best_host', roles='host')
                            self.best_against_choose_first = rhos[2]
                            self.best_against_choose_last = rhos[3]
                        if rhos[5] < self.best_against_zeillinger:
                            self.save_checkpoint(f'{self.version_string}/best_agent', roles='agent')
                            self.best_against_zeillinger = rhos[5]

                    self.summary_writer.flush()  # Force writing to file after each validation.

    def validate(
        self,
        metric_fn: Optional[Callable] = None,
        verbose=1,
        batch_size=50,
        num_of_loops=10,
        max_length=None,
        write_tensorboard=False,
        key=None,
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

        hosts, agents = self.get_cached_hosts_agents_for_validation(batch_size)

        # Pit host network against all and agent network against the rest.
        battle_schedule = [(0, i) for i in range(len(agents))] + [(i, 0) for i in range(1, len(hosts))]

        rhos = []
        details = []

        for pair_idx in battle_schedule:
            key, subkey = jax.random.split(key)

            host, agent = hosts[pair_idx[0]], agents[pair_idx[1]]
            rho, detail = metric_fn(
                host,
                agent,
                batch_size=batch_size,
                num_of_loops=num_of_loops,
                max_length=max_length,
                write_tensorboard=pair_idx == (0, 0) and write_tensorboard,
                key=key,
            )
            rhos.append(rho)
            details.append(detail)
            if verbose:
                self.logger.info(f"{get_name(host)} vs {get_name(agent)}:")
                self.logger.info(f"  {get_name(metric_fn)}: {rho}")
            if write_tensorboard:
                self.summary_writer.add_scalar(
                    f"{get_name(host)}_v_{get_name(agent)}", float(rho), self.get_state("host").step[0]
                )
                if sum(detail) > 0:
                    hist = np.concatenate([np.full((detail[i],), i) for i in range(len(detail) - 1)], axis=0)
                    self.summary_writer.add_histogram(
                        f"{get_name(host)}_v_{get_name(agent)}/length_histogram", hist, self.get_state("host").step[0]
                    )

        return rhos, details

    def compute_rho(
        self,
        host: Callable,
        agent: Callable,
        batch_size=None,
        num_of_loops=10,
        max_length=None,
        write_tensorboard=False,
        key=None,
    ) -> Tuple[float, List]:
        """
        Calculate the rho number between the host and agent.
        Parameters:
            host: a pmapped function that takes in point state and returns the one-hot action vectors.
            agent: a pmapped function that takes in point state and returns the one-hot action vectors.
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

            # Generate new sets of points.
            # - note: rescale is always set to False since we put rescaling into feature functions
            pts = generate_pts(
                keys[1:], (batch_size, self.max_num_points, self.dimension), self.max_value, self.dtype, False
            )

            prev_done, done = 0, jnp.sum(pmap(get_dones)(pts))  # Calculate the finished games
            pts = pmap(flatten)(pts)

            # For tensorboard logging
            collect_host_actions, collect_agent_actions = [], []

            for step in range(max_length - 1):
                keys = jax.random.split(key, num=2 * self.device_num + 1)
                key = keys[0]
                host_keys = keys[1: self.device_num + 1]
                agent_keys = keys[self.device_num + 1:]

                details[step] += done - prev_done
                host_action = host(pts, key=host_keys)
                coords = batch_decode(host_action).astype(self.dtype)
                agent_obs = pmap(make_agent_obs)(pts, coords)
                axis = jnp.argmax(agent(agent_obs, key=agent_keys), axis=-1).astype(self.dtype)
                pts = take_action(pts, coords, axis)

                # Have to sync by summing results across devices. prev_done and done are single scalars.
                prev_done, done = done, jnp.sum(pmap(get_dones)(p_reshape(pts, (-1, *spec))))  # Update the finished games

                if write_tensorboard:
                    collect_host_actions.append(np.array(jnp.ravel(np.argmax(host_action, axis=-1))))
                    collect_agent_actions.append(np.array(jnp.ravel(axis)))

            details[max_length - 1] += batch_size * self.device_num - done

            if write_tensorboard:
                self.summary_writer.add_histogram(
                    "host_action_distributions", np.concatenate(collect_host_actions), self.get_state("host").step[0]
                )
                self.summary_writer.add_histogram(
                    "agent_action_distributions", np.concatenate(collect_agent_actions), self.get_state("agent").step[0]
                )

        rho = sum(details[1:]) / sum([i * num for i, num in enumerate(details)])
        return rho, details

    def rollout_postprocess(self, rollouts: RollOut, role: str, use_unified_tree=True) -> RollOut:
        """
        Perform postprocessing on the rollout samples. In this default postprocessing, we replace the value_prior
            from the MCTS tree by the ground-truth value depending on the game win/lose.
        Parameters:
            rollouts: the rollout set to be processed. (observations, policy_prior, value_prior). Shapes are below.
                - obs (b, max_length_game, input_dim), where input_dim is the dim of all flattened features.
                - policy (b, max_length_game, dimension)
                - value (b, max_length_game)
            role: the current role (agent value is the negative of host value)
            use_unified_tree: (Optional) in simulation function, the unified MC search tree is used (in which case,
                the host observation is zero padded at the end).
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
        if use_unified_tree:
            # if the unified tree is used, host observations are zero-padded and should use the same agent-criteria
            #   to identify states that are terminal.
            done = get_done_from_flatten(obs, 'agent', self.dimension)  # (b, max_length_game)
        else:
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

    def save_checkpoint(self, path: str, roles: Optional[Union[List[str], str]] = None):
        roles = ['host', 'agent'] if roles is None else roles
        if isinstance(roles, str):
            roles = [roles]
        for role in roles:
            state = self.get_state(role)
            checkpoints.save_checkpoint(
                ckpt_dir=path, prefix=f"{role}_", overwrite=True, target=flax.jax_utils.unreplicate(state), step=state.step[0]
            )

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
                state = flax.jax_utils.replicate(
                    checkpoints.restore_checkpoint(
                        ckpt_dir=file_path, target=self.get_state(role), step=step, prefix=f"{role}_"
                    )
                )
                setattr(self, f"{role}_state", state)
                self.logger.info(f"Successfully loaded {file_path}.")
                self.update_fns(role)

    # ---------- Below are getter functions that require caching after their first calls ---------- #

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

    def get_apply_fn(self, role: str) -> Callable:
        """
        Get the apply function that evaluates an input with given parameters using a neural net.
        This is the raw version of neural net inference without any postprocessing and masking. It is used in
            self.train.
        """
        if getattr(self, f"{role}_apply_fn", None) is None:
            model = getattr(self, f"{role}_model")
            feature_fn = getattr(self, f"{role}_feature_fn")

            def apply_fn(x, param, **kwargs):
                return model.apply(param, feature_fn(x))

            setattr(self, f"{role}_apply_fn", apply_fn)
        return getattr(self, f"{role}_apply_fn")

    def get_mcts_sim_fn(self, role: str) -> Callable:
        if getattr(self, f"{role}_mcts_sim_fn", None) is None:
            opponent = self._get_opponent(role)
            policy_fn = self.get_policy_fn(role)
            opp_policy_fn = getattr(self, f"{opponent}_mcts_policy_fn")
            _, sim_fn, _ = self.update_eval_sim_and_mcts_policy(role,
                                                                policy_fn,
                                                                opp_policy_fn,
                                                                return_function=True)

            setattr(self, f"{role}_mcts_sim_fn", sim_fn)

        return getattr(self, f"{role}_mcts_sim_fn")

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
            state = flax.jax_utils.replicate(TrainState.create(apply_fn=apply_fn, params=parameters, tx=optim))
            setattr(self, f"{role}_state", state)
        return state

    def get_cached_hosts_agents_for_validation(self, batch_size: int, force_update=False):
        """
        Get a set of host functions and a set of agent functions. They are policies of different strategies and will
            be used to fight against each other. We cache those functions after the first access.
        Parameters:
            batch_size: the batch size of the input observations.
            force_update: force an update on all the policy functions.
        Returns:
            a list of host functions and a list of agent functions.
        """
        if batch_size not in self.cached_hosts_agents_for_validation or force_update:
            spec = (self.max_num_points, self.dimension)
            host_fn_before_wrapper = jax.pmap(self.host_policy_fn)
            agent_fn_before_wrapper = jax.pmap(self.agent_policy_fn)

            def expose_name(func):
                if hasattr(func, "func"):
                    func.__name__ = func.func.__name__
                return func

            hosts = [
                host_fn_before_wrapper,
                jax.pmap(get_host_with_flattened_obs(spec, random_host_fn)),
                jax.pmap(get_host_with_flattened_obs(spec, zeillinger_fn)),
                jax.pmap(get_host_with_flattened_obs(spec, all_coord_host_fn)),
            ]
            agents = [
                agent_fn_before_wrapper,
                jax.pmap(expose_name(partial(random_agent_fn, spec=spec))),
                jax.pmap(expose_name(partial(choose_first_agent_fn, spec=spec))),
                jax.pmap(expose_name(partial(choose_last_agent_fn, spec=spec))),
            ]
            self.cached_hosts_agents_for_validation[batch_size] = hosts, agents

        _cached_hosts, _cached_agents = self.cached_hosts_agents_for_validation[batch_size]
        hosts = [action_wrapper(partial(_cached_hosts[0], params=self.host_state.params), None), *_cached_hosts[1:]]
        agents = [action_wrapper(partial(_cached_agents[0], params=self.agent_state.params), None), *_cached_agents[1:]]
        return hosts, agents

    # ---------- Below are either static methods or methods that set its members ---------- #

    def set_optim(self, role: str, optim_config: dict):
        """
        Set up/reset the optimizer.
        """
        optim_name = optim_config["name"]
        optim_args = optim_config["args"]
        setattr(self, f"{role}_optim", self.optim_dict[optim_name](**optim_args))

    def update_eval_sim_and_mcts_policy(self, role: str,
                                        policy_fn: Optional[Callable] = None,
                                        opp_policy_fn: Optional[Callable] = None,
                                        jitted=False,
                                        return_function=False) -> Any:
        """
        Update `{role}_eval_loop`, '{role}_eval_loop_as_opp', `{role}_sim_fn`, `{role}_mcts_policy_fn`,
            `unified_eval_loop` and `unified_sim_fn`.
        Parameters:
            role: host or agent.
            policy_fn: (Optional) the policy function used to generate simulations.
            opp_policy_fn: (Optional) the opponent policy function used to generate simulations.
            jitted: (Optional) whether to jit the functions. (Might deprecate soon)
            return_function: (Optional) if true, return the functions.
        Returns:
            If return_function is true, return the tuple of all four functions.
            Otherwise, nothing.
        """
        opponent = self._get_opponent(role)
        maybe_jit = jax.jit if jitted else lambda x, **_: x
        simulation_config = {
            "eval_batch_size": self.eval_batch_size,
            "max_num_points": self.max_num_points,
            "dimension": self.dimension,
            "max_length_game": self.max_length_game,
            "max_value": self.max_value,
            "scale_observation": False,  # The burden of rescaling is put to feature functions
        }

        policy_fn = getattr(self, f"{role}_policy_fn") if policy_fn is None else policy_fn
        opp_policy_fn = getattr(self, f"{opponent}_policy_fn") if opp_policy_fn is None else opp_policy_fn

        config = {
            'spec': (self.max_num_points, self.dimension),
            'max_depth': self.max_length_game,
            'max_num_considered_actions': self.max_num_considered_actions,
            'discount': self.discount,
            'rescale_points': False,  # The burden of rescaling is put to feature functions
            'dtype': self.dtype
        }
        eval_loop = get_evaluation_loop(
            role,
            policy_fn,  # output policy logits and value estimates
            action_wrapper(opp_policy_fn, None),  # output definitive actions as one-hot array
            getattr(self, f"{role}_reward_fn"),
            num_evaluations=self.num_evaluations,
            **config
        )
        eval_loop_as_opp = get_evaluation_loop(
            role,
            policy_fn,  # output policy logits and value estimates
            action_wrapper(opp_policy_fn, None),  # output definitive actions as one-hot array
            getattr(self, f"{role}_reward_fn"),
            num_evaluations=self.num_evaluations_as_opponent,
            **config
        )
        unified_eval_loop = get_evaluation_loop(
            'host',
            get_host_with_flattened_obs(config['spec'], self.host_policy_fn, truncate_input=True),
            self.agent_policy_fn,
            self.agent_reward_fn,  # there is an off-by-one-step problem in rewards depending on your perspectives...
            num_evaluations=self.num_evaluations,
            role_agnostic=True,  # <-- this is for the unified MC search tree.
            **config
        )
        sim_fn = maybe_jit(
            get_simulation(role, eval_loop, config=simulation_config, dtype=self.dtype),
            backend="cpu" if self.eval_on_cpu or not self.use_cuda else jax.default_backend(),
        )
        unified_sim_fn = maybe_jit(
            get_simulation('host', unified_eval_loop, config=simulation_config, dtype=self.dtype),
            backend="cpu" if self.eval_on_cpu or not self.use_cuda else jax.default_backend(),
        )
        mcts_policy_fn = mcts_wrapper(eval_loop_as_opp)
        if role == 'agent':
            mcts_policy_fn = apply_agent_action_mask(mcts_policy_fn, self.dimension)

        if return_function:
            return eval_loop, sim_fn, mcts_policy_fn
        else:
            setattr(self, f"{role}_eval_loop", eval_loop)
            setattr(self, f"{role}_eval_loop_as_opp", eval_loop_as_opp)
            setattr(self, f"{role}_sim_fn", sim_fn)
            setattr(self, f"{role}_mcts_policy_fn", mcts_policy_fn)
            if not hasattr(self, "unified_eval_loop") or not hasattr(self, "unified_sim_fn"):
                setattr(self, "unified_eval_loop", unified_eval_loop)
                setattr(self, "unified_sim_fn", unified_sim_fn)

    def update_policy_fn(self, role: str, return_function=False) -> Any:
        """
        Update `{role}_policy_fn`.
        Parameters:
            role: host or agent.
            return_function: (Optional) if true, return the function.
        Returns:
            If return_function is true, return the policy function.
            Otherwise, nothing.
        """
        feature_fn = getattr(self, f"{role}_feature_fn")
        model = getattr(self, f"{role}_model")

        policy_fn = get_apply_fn(role, model, (self.max_num_points, self.dimension), feature_fn=feature_fn)
        if role == "agent":
            policy_fn = apply_agent_action_mask(policy_fn, self.dimension)
        if return_function:
            return policy_fn
        else:
            setattr(self, f"{role}_policy_fn", policy_fn)

    def update_fns(self, role: str):
        self.update_eval_sim_and_mcts_policy(role)
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
