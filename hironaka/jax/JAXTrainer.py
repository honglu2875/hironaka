from functools import partial
from typing import Union, Callable, Tuple, Any

import jax
import jax.numpy as jnp
import optax
import yaml
import logging
from flax.training import train_state
#from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from hironaka.jax.net import DenseResNet, PolicyWrapper
from hironaka.jax.simulation_fn import get_evaluation_loop, get_single_thread_simulation
from hironaka.jax.util import get_reward_fn, policy_value_loss, compute_loss, action_wrapper
from hironaka.trainer.Scheduler import ConstantScheduler, ExponentialLRScheduler, InverseLRScheduler, Scheduler


class JAXTrainer:
    """
    This consolidates the whole training process given a config file.
    """
    optim_dict = {'adam': optax.adam,
                  'adamw': optax.adamw,
                  'sgd': optax.sgd}
    lr_scheduler_dict = {'constant': ConstantScheduler,
                         'exponential': ExponentialLRScheduler,
                         'inverse': InverseLRScheduler}

    def __init__(self, key, config: Union[dict, str], dtype=jnp.float32):
        self.logger = logging.getLogger(__class__.__name__)

        if isinstance(config, str):
            with open(config, "r") as stream:
                self.config = yaml.safe_load(stream)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be either a string or a dict. Got {type(config)}.")

        config_keys = ['eval_batch_size', 'max_num_points', 'dimension', 'max_value', 'scale_observation',
                       'use_tensorboard', 'layerwise_logging', 'use_cuda', 'version_string', 'num_evaluations',
                       'max_depth', 'max_num_considered_actions', 'rollout_size']
        for config_key in config_keys:
            setattr(self, config_key, self.config[config_key])

        self.dtype = dtype

        if not self.use_cuda:
            jax.config.update('jax_platform_name', 'cpu')

        eval_batch_size, max_num_points, dimension = self.eval_batch_size, self.max_num_points, self.dimension
        self.input_dim = {'host': max_num_points * dimension,
                          'agent': max_num_points * dimension + dimension}
        self.output_dim = {'host': 2 ** dimension - dimension - 1,
                           'agent': dimension}
        policy_keys = {}
        sim_keys = {}
        key, policy_keys['host'], policy_keys['agent'], sim_keys['host'], sim_keys['agent'] = \
            jax.random.split(key, num=5)

        for role in ['host', 'agent']:
            if role not in self.config:
                continue
            net = DenseResNet(self.output_dim[role] + 1, net_arch=self.config[role]['net_arch'])
            setattr(self, f"{role}_policy",
                    PolicyWrapper(policy_keys[role], (eval_batch_size, self.input_dim[role]), net))
            self.set_optim(role, self.config[role]['optim'])

            # Set up policy functions (policy and evaluation have different batch sizes)
            self.update_eval_policy_fn(role)

            # Use fixed reward function for now.
            setattr(self, f"{role}_reward_fn", get_reward_fn(role))

            setattr(self, f"{role}_state", None)

        for role in ['host', 'agent']:
            self.update_sim_loop(role)

        self.gradient_step = 0
        self.host_opponent_policy, self.agent_opponent_policy = None, None

    def simulate(self, key: jnp.ndarray, role: str) -> Tuple:
        """
        A helper function of performing a simulation. Nothing but calling `{role}_sim_fn`
        Parameters:
            role: host or agent
        Returns:
            obs, target_policy, target_value
        """
        sim_fn = getattr(self, f"{role}_sim_fn")
        policy_wrapper = getattr(self, f"{role}_policy")
        opponent = self._get_opponent(role)
        opp_policy_wrapper = getattr(self, f"{opponent}_policy")

        return sim_fn(key, role_fn_args=(policy_wrapper.parameters,), opponent_fn_args=(opp_policy_wrapper.parameters,))

    def train(self, key: jnp.ndarray, role: str, gradient_steps: int, experiences: jnp.ndarray,
              update_sim_fn=False, jit_train_fn=False):
        """
        Trains the neural network with a collection of samples ('experiences', supposedly collected from simulation).
        Parameters:
            key: the PRNG key.
            role: either host or agent.
            gradient_steps: number of gradient steps to take.
            experiences: (batch_size, *input_dim), a batch of simulation samples.
            update_sim_fn: (Optional) True if need to update and recompile the opponent policy as well as the simulation
                loop function (would be slow).
            jit_train_fn: (Optional) jit compile the training function
        """
        opponent = self._get_opponent(role)

        if update_sim_fn:
            self.update_eval_policy_fn(role)
            self.update_sim_loop(role)
        else:
            if getattr(self, f"{role}_policy") is None:
                raise RuntimeError(f"opponent {opponent} policy function not initialized.")

        optim = getattr(self, f"{role}_optim")
        state = getattr(self, f"{role}_state", None)
        policy_wrapper = getattr(self, f"{role}_policy")

        batch_size = self.config[role]['batch_size']
        if state is None:
            apply_fn = policy_wrapper.get_apply_fn(batch_size)
            state = TrainState.create(
                apply_fn=apply_fn,
                params=policy_wrapper.parameters,
                tx=optim)
        train_fn = jax.jit(partial(self.train_step, loss_fn=policy_value_loss)) if jit_train_fn else \
            partial(self.train_step, loss_fn=policy_value_loss)
        sample_size = experiences[0].shape[0]

        for i in range(gradient_steps):
            key, subkey = jax.random.PRNGKey(key)
            sample_idx = jax.random.randint(subkey, (batch_size,), 0, sample_size)
            sample = experiences[0][sample_idx, :], experiences[1][sample_idx, :], experiences[2][sample_idx]

            state, loss = train_fn(state, sample)

            # temporary logging. TODO: change to tensorboard
            self.logger.info(f"Loss: {loss}")

        # Save the state and parameters
        setattr(self, f"{role}_state", state)
        policy_wrapper.parameters = state.params

    def train_step(self, state: jnp.ndarray, sample: jnp.ndarray, loss_fn: Callable) -> Tuple[TrainState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(partial(compute_loss, loss_fn=loss_fn))(state.params, state.apply_fn, sample)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def set_optim(self, role: str, optim_config: dict):
        """
        Set up/reset the optimizer.
        """
        optim_name = optim_config['name']
        optim_args = optim_config['args']
        setattr(self, f"{role}_optim", self.optim_dict[optim_name](**optim_args))

    def update_eval_policy_fn(self, role: str, jitted=True):
        """
        Update `{role}_eval_policy_fn` based on the current parameters of the network.
        """
        maybe_jit = jax.jit if jitted else lambda x: x
        batch_size = self.config['eval_batch_size']
        setattr(self, f"{role}_eval_policy_fn", maybe_jit(getattr(self, f"{role}_policy").get_apply_fn(batch_size)))

    def update_sim_loop(self, role: str, jitted=True, return_fn=False) -> Any:
        """
        Update `{role}_eval_loop` and `{role}_sim_fn`.
        """
        opponent = self._get_opponent(role)
        dim = self.dimension if role == 'host' else None
        maybe_jit = jax.jit if jitted else lambda x: x
        simulation_config = {'eval_batch_size': self.eval_batch_size,
                             'max_num_points': self.max_num_points,
                             'dimension': self.dimension,
                             'max_value': self.max_value,
                             'scale_observation': self.scale_observation}

        eval_loop = get_evaluation_loop(role,
                                        getattr(self, f"{role}_eval_policy_fn"),
                                        action_wrapper(getattr(self, f"{opponent}_eval_policy_fn"), dim),
                                        getattr(self, f"{role}_reward_fn"), spec=(self.max_num_points, self.dimension),
                                        num_evaluations=self.num_evaluations, max_depth=self.max_depth,
                                        max_num_considered_actions=self.max_num_considered_actions,
                                        rescale_points=self.scale_observation)
        sim_fn = maybe_jit(get_single_thread_simulation(role, eval_loop, rollout_size=self.rollout_size,
                                                        config=simulation_config, dtype=self.dtype))
        if return_fn:
            return eval_loop, sim_fn
        else:
            setattr(self, f"{role}_eval_loop", eval_loop)
            setattr(self, f"{role}_sim_fn", sim_fn)

    def purge_state(self, role: str):
        """
        Removes the training state of host or agent. TrainState object stores information of an ongoing sequence of
            gradient steps. It will not affect parameters that are already saved in `self.host_policy` and
            `self.agent_policy`.
        """
        setattr(self, f"{role}_state", None)

    @staticmethod
    def _get_opponent(role: str) -> str:
        if role == 'host':
            return 'agent'
        elif role == 'agent':
            return 'host'
        else:
            raise ValueError(f"role must be either host or agent. Got {role}.")

