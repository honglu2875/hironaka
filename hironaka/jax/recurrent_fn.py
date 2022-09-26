from functools import partial
from typing import Callable, Tuple

import jax
import mctx

import jax.numpy as jnp
from hironaka.jax.util import (
    get_dones,
    get_preprocess_fns,
    get_take_actions,
    make_agent_obs, flatten, get_dynamic_policy_fn,
)
from hironaka.jax.host_action_preprocess import get_batch_decode, get_batch_decode_from_one_hot


def get_recurrent_fn_for_role(
    role: str,
    role_fn: Callable,
    opponent_action_fn: Callable,
    reward_fn: Callable,
    spec: Tuple[int, int],
    discount=0.99,
    dtype=jnp.float32,
    rescale_points=True,
) -> Callable:
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

    if role == "host":
        def first_obs_update(x, y, z):
            return x

        opponent_action_preprocess = partial(jnp.argmax, axis=1)  # one-hot agent actions to discrete indices
        second_obs_update = get_take_actions(role="host", spec=spec, rescale_points=rescale_points)
        make_opponent_obs = make_agent_obs
        batch_decode = get_batch_decode(spec[1])
    elif role == "agent":
        first_obs_update = get_take_actions(role="agent", spec=spec, rescale_points=rescale_points)
        opponent_action_preprocess = get_batch_decode_from_one_hot(spec[1])  # one-hot host actions to multi-binary

        def second_obs_update(x, y, z):
            return make_agent_obs(x, z)

        def make_opponent_obs(obs, actions):
            return obs

        def batch_decode(x):
            return x

    else:
        raise ValueError(f"role must be either 'host' or 'agent'. Got {role}.")

    def recurrent_fn(params, key, actions: jnp.ndarray, observations: jnp.ndarray):
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
        key, subkey = jax.random.split(key)
        opponent_actions = opponent_action_preprocess(
            opponent_action_fn(make_opponent_obs(updated_obs, actions).astype(dtype), *opponent_fn_args, key=subkey)
        )

        # If host, `second_obs_update` takes an action (shift->newton polytope->rescale) and returns flattened obs
        # If agent, `second_obs_update` concatenates `updated_obs` and `opponent_actions` and returns flattened obs
        next_observations = second_obs_update(updated_obs, actions, opponent_actions).astype(dtype)
        dones = get_dones(obs_preprocess(next_observations))
        rewards = reward_fn(dones, prev_dones)

        key, subkey = jax.random.split(key)
        policy_prior, value_prior = role_fn(next_observations, *role_fn_args, key=subkey)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=jnp.array([discount] * batch_size, dtype=dtype),
            prior_logits=policy_prior,
            value=value_prior,
        )
        return recurrent_fn_output, next_observations

    return recurrent_fn


def get_unified_recurrent_fn(
    host_fn: Callable,
    agent_fn: Callable,
    reward_fn: Callable,
    spec: Tuple[int, int],
    discount=0.99,
    dtype=jnp.float32,
    rescale_points=True,
) -> Callable:
    """
    This is a unified recurrent function which generates an MC search tree without distinctions of the player roles.
    For this unified tree, the host and agent will share the same state format:
        (device_num, batch_size, max_num_points * dimension + dimension)
    In the case of host state, the last `dimension` entries [:, :, -dimension:] are padded by 0.
    In the case of agent state, we use the exact convention for agent states: flatten the points and concatenate
        with host coordinates.
    Hosts have a bigger action space than agents. So the number of actions (children in MC search tree) will be taken
        to be host's freedom. A bit wasteful on agent nodes but acceptable in small scales.
    """
    # preprocess functions apply to [..., max_num_points * dimension + dimension] array by cutting out/reshape the
    #   point part [..., :max_num_points * dimension] and the tail part [..., -dimension:].
    obs_preprocess, coords_preprocess = get_preprocess_fns('agent', spec)
    batch_decode = get_batch_decode(spec[1])
    # 'host' take_action function is the standard state transformation that takes in):
    #   either flattened/un-flattened observation, coords, axis
    # and output the new observation after the transformation, flattened.
    take_actions = get_take_actions(role="host", spec=spec, rescale_points=rescale_points)
    # As host/agent taking turn making alternating moves, the actual discount should take a minus sign.
    discount = -discount

    # The dynamic policy function will apply host_fn and agent_fn dynamically depending on whether the input state is
    #   a host state or agent state.
    dynamic_policy_fn = get_dynamic_policy_fn(spec, host_fn, agent_fn)

    def recurrent_fn(params, key, actions: jnp.ndarray, observations: jnp.ndarray):
        """
        Parameters:
            params: host parameter, agent parameter.
            key: the PRNG key.
            actions: 0 ~ 2**dimension-dimension-2 if host, 0 ~ dimension-1 if agent.
            observations: the flattened state representation detailed above.
        Returns:
            mctx.RecurrentFnOutput (containing reward, discount, prior logits and value), next_observation
        """
        host_param, agent_param = params
        batch_size = observations.shape[0]
        obs, coord = obs_preprocess(observations), coords_preprocess(observations, None)
        # Assume all states in the batch are uniformly either host or agent.
        # It is easy to implement separate treatments for host/agent for each batch elem, but I do not see the need atm.
        is_host = jnp.any(jnp.all(jnp.isclose(coord, 0), axis=-1), axis=None)
        # If the action comes from host, discrete actions must be converted to multi-binary arrays.
        # E.g., dimension=3, [2, 3] -> [[0, 1, 1], [1, 1, 1]]
        # If from agent, zero-pad the next_coord.
        next_coord = jnp.where(is_host, batch_decode(actions), jnp.zeros((batch_size, spec[1])))
        next_obs = jnp.where(is_host, flatten(obs), take_actions(obs, coord, actions))
        next_state = jnp.concatenate([next_obs, next_coord], axis=-1)

        key, subkey = jax.random.split(key)
        policy_prior, value_prior = dynamic_policy_fn(next_state, (host_param, agent_param), key=subkey)

        prev_dones = get_dones(obs)
        dones = get_dones(next_obs.reshape(-1, *spec))
        rewards = reward_fn(dones, prev_dones)

        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=rewards,
            discount=jnp.array([discount] * batch_size, dtype=dtype),
            prior_logits=policy_prior,
            value=value_prior,
        )
        return recurrent_fn_output, next_state

    return recurrent_fn
