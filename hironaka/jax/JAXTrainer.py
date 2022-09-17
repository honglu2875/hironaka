import logging
import time
from functools import partial
from typing import Any, Callable, List, Tuple, Union

import jax
import jax.numpy as jnp
import optax
import yaml
# from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from hironaka.jax.net import DenseResNet, PolicyWrapper
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
    compute_loss,
    flatten,
    get_batch_decode_from_one_hot,
    get_dones,
    get_name,
    get_reward_fn,
    get_take_actions,
    make_agent_obs,
    policy_value_loss,
)
from hironaka.src import get_newton_polytope_jax, rescale_jax
from hironaka.trainer.Scheduler import ConstantScheduler, ExponentialLRScheduler, InverseLRScheduler

RollOut = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class JAXTrainer:
    """
    This consolidates the whole training process given a config file.
    """

    optim_dict = {"adam": optax.adam, "adamw": optax.adamw, "sgd": optax.sgd}
    lr_scheduler_dict = {"constant": ConstantScheduler, "exponential": ExponentialLRScheduler, "inverse": InverseLRScheduler}

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
            "scale_observation",
            "use_tensorboard",
            "layerwise_logging",
            "use_cuda",
            "version_string",
            "num_evaluations",
            "max_num_considered_actions",
            "discount",
        ]
        for config_key in config_keys:
            setattr(self, config_key, self.config[config_key])

        self.dtype = dtype

        if not self.use_cuda:
            jax.config.update("jax_platform_name", "cpu")

        eval_batch_size, max_num_points, dimension = self.eval_batch_size, self.max_num_points, self.dimension
        self.input_dim = {"host": max_num_points * dimension, "agent": max_num_points * dimension + dimension}
        self.output_dim = {"host": 2**dimension - dimension - 1, "agent": dimension}
        policy_keys = {}
        sim_keys = {}
        key, policy_keys["host"], policy_keys["agent"], sim_keys["host"], sim_keys["agent"] = jax.random.split(key, num=5)

        for role in ["host", "agent"]:
            if role not in self.config:
                continue
            net = DenseResNet(self.output_dim[role] + 1, net_arch=self.config[role]["net_arch"])
            setattr(self, f"{role}_policy", PolicyWrapper(policy_keys[role], (eval_batch_size, self.input_dim[role]), net))
            self.set_optim(role, self.config[role]["optim"])

            # Set up policy functions (policy and evaluation have different batch sizes)
            self.update_eval_policy_fn(role)

            # Use fixed reward function for now.
            setattr(self, f"{role}_reward_fn", get_reward_fn(role))

            setattr(self, f"{role}_state", None)

        for role in ["host", "agent"]:
            self.update_sim_loop(role)
            self.update_train_fn(role)

        self.gradient_step = 0
        self.host_opponent_policy, self.agent_opponent_policy = None, None

    def simulate(self, key: jnp.ndarray, role: str) -> RollOut:
        """
        A helper function of performing a simulation. Nothing but calling `{role}_sim_fn`
        Parameters:
            key: the PRNG key.
            role: host or agent.
        Returns:
            obs, target_policy, target_value
        """
        sim_fn = getattr(self, f"{role}_sim_fn")
        policy_wrapper = getattr(self, f"{role}_policy")
        opponent = self._get_opponent(role)
        opp_policy_wrapper = getattr(self, f"{opponent}_policy")

        return self.rollout_postprocess(
            role, sim_fn(key, role_fn_args=(policy_wrapper.parameters,), opponent_fn_args=(opp_policy_wrapper.parameters,))
        )

    def train(self, key: jnp.ndarray, role: str, gradient_steps: int, rollouts: jnp.ndarray, random_sampling=False):
        """
        Trains the neural network with a collection of samples ('experiences', supposedly collected from simulation).
        Parameters:
            key: the PRNG key.
            role: either host or agent.
            gradient_steps: number of gradient steps to take.
            rollouts: (batch_size, *input_dim), a batch of simulation samples.
            random_sampling: (Optional) whether to do random sampling in the rollouts.
        """
        opponent = self._get_opponent(role)

        if getattr(self, f"{role}_policy") is None:
            raise RuntimeError(f"opponent {opponent} policy function not initialized.")

        state = getattr(self, f"{role}_state", None)
        policy_wrapper = getattr(self, f"{role}_policy")
        train_fn = getattr(self, f"{role}_train_fn")

        sample_size = rollouts[0].shape[0]
        batch_size = self.config[role]["batch_size"]
        gradient_steps = sample_size // batch_size if not random_sampling else gradient_steps

        if state is None:
            optim = getattr(self, f"{role}_optim")
            apply_fn = policy_wrapper.get_apply_fn(batch_size)
            state = TrainState.create(apply_fn=apply_fn, params=policy_wrapper.parameters, tx=optim)

        if train_fn is None:
            self.update_train_fn(role)
            train_fn = getattr(self, f"{role}_train_fn")

        for i in range(gradient_steps):
            key, subkey = jax.random.PRNGKey(key)

            if random_sampling:
                sample_idx = jax.random.randint(subkey, (batch_size,), 0, sample_size)
            else:
                sample_idx = jnp.arange(i * batch_size, (i + 1) * batch_size)

            sample = rollouts[0][sample_idx, :], rollouts[1][sample_idx, :], rollouts[2][sample_idx]

            state, loss = train_fn(state, sample)

            # temporary logging. TODO: change to tensorboard
            # self.logger.info(f"Loss: {loss}")

        # Save the state and parameters
        setattr(self, f"{role}_state", state)
        policy_wrapper.parameters = state.params

    def train_step(self, state: jnp.ndarray, sample: jnp.ndarray, loss_fn: Callable) -> Tuple[TrainState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(partial(compute_loss, loss_fn=loss_fn))(state.params, state.apply_fn, sample)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def evaluate(
        self, eval_fn: Callable = None, verbose=1, batch_size=10, num_of_loops=10, max_length=None, key=None
    ) -> Tuple[List, List]:
        key = jax.random.PRNGKey(time.time_ns()) if key is None else key
        max_length = self.max_length_game if max_length is None else max_length
        eval_fn = self.get_rho if eval_fn is None else eval_fn

        spec = (self.max_num_points, self.dimension)

        host_policy_wrapper = self.host_policy
        agent_policy_wrapper = self.agent_policy

        # jit is forced for policy networks here. Otherwise, it is simply unpractical and too slow.
        host_fn = jax.jit(
            action_wrapper(partial(host_policy_wrapper.get_apply_fn(batch_size), params=host_policy_wrapper.parameters), None)
        )
        agent_fn = jax.jit(
            action_wrapper(
                partial(agent_policy_wrapper.get_apply_fn(batch_size), params=agent_policy_wrapper.parameters), self.dimension
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

        battle_schedule = [(0, i) for i in range(len(agents))] + [(i, 0) for i in range(1, len(hosts))]

        rhos = []
        details = []
        for pair_idx in battle_schedule:
            key, subkey = jax.random.split(key)

            host, agent = hosts[pair_idx[0]], agents[pair_idx[1]]
            rho, detail = eval_fn(
                host, agent, batch_size=batch_size, num_of_loops=num_of_loops, max_length=max_length, key=key
            )
            rhos.append(rho)
            details.append(detail)
            if verbose:
                self.logger.info(f"{get_name(host)} vs {get_name(agent)}:")
                self.logger.info(f"  {get_name(eval_fn)}: {rho}")

        return rhos, details

    def get_rho(
        self, host: Callable, agent: Callable, batch_size=10, num_of_loops=10, max_length=20, key=None
    ) -> Tuple[float, List]:
        """
        Calculate the rho number between the host and agent.
        Parameters:
            host: a function that takes in point state and returns the one-hot action vectors.
            agent: a function that takes in point state and returns the one-hot action vectors.
            batch_size: the batch size.
            num_of_loops: the number of steps to evaluate.
            max_length: the maximal length of game.
            key: (Optional) the PRNG random key (if None, will use time.time_ns() to seed a key)
        Returns:
            the rho number.
        """
        key = jax.random.PRNGKey(time.time_ns()) if key is None else key

        max_num_points, dimension = self.max_num_points, self.dimension
        spec = (max_num_points, dimension)

        take_action = get_take_actions(role="host", spec=spec, rescale_points=self.scale_observation)

        batch_decode = get_batch_decode_from_one_hot(dimension)

        details = [0] * max_length
        for _ in range(num_of_loops):
            key, host_key, agent_key = jax.random.split(key, num=3)

            pts = jax.random.randint(host_key, (batch_size, max_num_points, dimension), 0, self.max_value).astype(self.dtype)
            pts = rescale_jax(get_newton_polytope_jax(pts)) if self.scale_observation else get_newton_polytope_jax(pts)

            prev_done, done = 0, jnp.sum(get_dones(pts))  # Calculate the finished games
            pts = flatten(pts)
            for step in range(max_length - 1):
                key, host_key, agent_key = jax.random.split(key, num=3)

                details[step] += done - prev_done

                coords = batch_decode(host(pts, key=host_key)).astype(self.dtype)
                axis = jnp.argmax(agent(make_agent_obs(pts, coords), key=agent_key), axis=1).astype(self.dtype)
                pts = take_action(pts, coords, axis)

                prev_done, done = done, jnp.sum(get_dones(pts.reshape((-1, *spec))))  # Update the finished games
            details[max_length - 1] += batch_size - done

        rho = sum(details[1:]) / sum([i * num for i, num in enumerate(details)])
        return rho, details

    def rollout_postprocess(self, role: str, rollouts: RollOut) -> RollOut:
        """
        Perform postprocessing on the rollout samples. In this default postprocessing, we replace the value_prior
            from the MCTS tree by the ground-truth value depending on the game win/lose.
        Parameters:
            role: the current role (agent value is the negative of host value)
            rollouts: the rollout set to be processed. (observations, policy_prior, value_prior).
        Returns:
            the processed rollouts.
        """
        # obs (b, max_length_game, input_dim)
        # policy (b, max_length_game, dimension)
        # value (b, max_length_game)
        obs, policy, value = rollouts
        max_length_game = obs.shape[1]

        dones = jnp.sum(obs[:, :, :] >= 0, axis=-1) <= self.dimension
        first_dones = jnp.argmax(dones, axis=-1)
        steps_before_done = first_dones.reshape(-1, 1) - jnp.arange(max_length_game).reshape((1, -1))
        steps_before_done = steps_before_done * (steps_before_done >= 0)

        never_ended = jnp.repeat(jnp.all(~dones, axis=-1).reshape((-1, 1)), max_length_game, axis=-1)

        dtype = value.dtype
        value = self.discount**steps_before_done - never_ended
        if role == "agent":
            value = -value
        value = jnp.ravel(value).astype(dtype)

        obs = obs.reshape((-1, obs.shape[2]))
        policy = policy.reshape((-1, policy.shape[2]))

        return obs, policy, value

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
        maybe_jit = jax.jit if jitted else lambda x: x
        batch_size = self.eval_batch_size
        setattr(self, f"{role}_eval_policy_fn", maybe_jit(getattr(self, f"{role}_policy").get_apply_fn(batch_size)))

    def update_sim_loop(self, role: str, jitted=True, return_function=False) -> Any:
        """
        Update `{role}_eval_loop` and `{role}_sim_fn`.
        """
        opponent = self._get_opponent(role)
        dim = self.dimension if role == "host" else None
        maybe_jit = jax.jit if jitted else lambda x: x
        simulation_config = {
            "eval_batch_size": self.eval_batch_size,
            "max_num_points": self.max_num_points,
            "dimension": self.dimension,
            "max_value": self.max_value,
            "scale_observation": self.scale_observation,
        }

        eval_loop = get_evaluation_loop(
            role,
            getattr(self, f"{role}_eval_policy_fn"),
            action_wrapper(getattr(self, f"{opponent}_eval_policy_fn"), dim),
            getattr(self, f"{role}_reward_fn"),
            spec=(self.max_num_points, self.dimension),
            num_evaluations=self.num_evaluations,
            max_depth=self.max_length_game,
            max_num_considered_actions=self.max_num_considered_actions,
            discount=self.discount,
            rescale_points=self.scale_observation,
            dtype=self.dtype,
        )
        sim_fn = maybe_jit(
            get_single_thread_simulation(
                role, eval_loop, max_length_game=self.max_length_game, config=simulation_config, dtype=self.dtype
            )
        )
        if return_function:
            return eval_loop, sim_fn
        else:
            setattr(self, f"{role}_eval_loop", eval_loop)
            setattr(self, f"{role}_sim_fn", sim_fn)

    def update_train_fn(self, role: str, jitted=True, return_function=False) -> Any:
        """
        Update `{role}_train_fn`.
        """
        train_fn = (
            jax.jit(partial(self.train_step, loss_fn=policy_value_loss))
            if jitted
            else partial(self.train_step, loss_fn=policy_value_loss)
        )
        if return_function:
            return train_fn
        else:
            setattr(self, f"{role}_train_fn", train_fn)

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
