from functools import partial
from typing import Callable, Tuple

import chex
import mctx
import jax
import jax.numpy as jnp
from jax import jit
import time

from hironaka.jax.util import make_agent_obs, get_dones, get_take_actions, get_preprocess_fns, get_batch_decode, \
    get_batch_decode_from_one_hot
from hironaka.src import rescale_jax, get_newton_polytope_jax, shift_jax


def get_recurrent_fn_for_role(role: str, role_fn: Callable, opponent_action_fn: Callable, reward_fn: Callable,
                              spec: Tuple[int, int], discount=0.99, dtype=jnp.float32, rescale_points=True) -> Callable:
    """
    The factory function for the recurrent_fn corresponding to a role (host or agent).
    Parameters:
        role: Either 'host' or 'agent'.
        role_fn: the Callable that produces policy and value corresponding to the player under evaluation.
         - parameters:
            observations: jnp.ndarray
            *args: any other arguments
         - returns: Tuple of policy_prior, value_prior
        opponent_action_fn: A fixed Callable represents the enemy. It takes an action in response to the `role_fn`,
            but no policy prior and value are generated.
         - parameters:
            observations: jnp.ndarray
            *args: any other arguments
         - returns: a batch of one-hot vectors of host/agent actions
        reward_fn: A fixed Callable returning the rewards according to whether the game has ended.
         - parameters:
            dones, jnp.ndarray
            prev_dones, jnp.ndarray
         - returns: a batch of rewards
        spec: (max_num_points, dimension)
        discount: (Optional) The discount value.
        dtype: (Optional) data type.
        rescale_points: (Optional) whether rescaling the points after shifting.
    Returns:
        the `recurrent_fn` under the given spec.
    """
    obs_preprocess, coords_preprocess = get_preprocess_fns(role, spec)

    if role == 'host':
        def first_obs_update(x, y, z): return x
        opponent_action_preprocess = partial(jnp.argmax, axis=1)  # one-hot agent actions to discrete indices
        second_obs_update = get_take_actions(role='host', spec=spec, rescale_points=rescale_points)
        make_opponent_obs = make_agent_obs
        batch_decode = get_batch_decode(spec[1])
    elif role == 'agent':
        first_obs_update = get_take_actions(role='agent', spec=spec, rescale_points=rescale_points)
        opponent_action_preprocess = get_batch_decode_from_one_hot(spec[1])  # one-hot host actions to multi-binary
        def second_obs_update(x, y, z): return make_agent_obs(x, z)
        def make_opponent_obs(obs, actions): return obs
        def batch_decode(x): return x
    else:
        raise ValueError(f"role must be either 'host' or 'agent'. Got {role}.")

    def recurrent_fn(params, key, actions: jnp.ndarray, observations: jnp.ndarray):
        del key
        role_fn_args, opponent_fn_args = params

        batch_size = observations.shape[0]
        # If the role is host, discrete actions must be converted to multi-binary arrays.
        # E.g., dimension=3, [2, 3] -> [[0, 1, 1], [1, 1, 1]]
        # If the role is agent, batch_decode is the identity function.
        actions = batch_decode(actions)

        # Before doing anything, record the array showing whether each game ended.
        prev_dones = get_dones(obs_preprocess(observations))

        # If host, `first_obs_update` returns the observation directly.
        # If agent, `first_obs_update` takes an action (shift->newton polytope->rescale) and returns flattened obs
        updated_obs = first_obs_update(observations, actions, actions)

        # The enemy observes the `updated_obs` and `actions`, and makes a decision on its actions
        opponent_actions = opponent_action_preprocess(
            opponent_action_fn(make_opponent_obs(updated_obs, actions).astype(dtype)), *opponent_fn_args)

        # If host, `second_obs_update` takes an action (shift->newton polytope->rescale) and returns flattened obs
        # If agent, `second_obs_update` concatenates `updated_obs` and `opponent_actions` and returns flattened obs
        next_observations = second_obs_update(updated_obs, actions, opponent_actions).astype(dtype)
        dones = get_dones(obs_preprocess(next_observations))
        rewards = reward_fn(dones, prev_dones)

        policy_prior, value_prior = role_fn(next_observations, *role_fn_args)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=jnp.array([discount] * batch_size, dtype=dtype),
            prior_logits=policy_prior,
            value=value_prior)
        return recurrent_fn_output, next_observations

    return recurrent_fn
