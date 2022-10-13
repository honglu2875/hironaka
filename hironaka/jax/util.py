import functools
import time
from functools import partial
from typing import Callable, Tuple, Union, Optional

import jax
from flax.core import FrozenDict
from jax.example_libraries.optimizers import l2_norm

from hironaka.jax.host_action_preprocess import decode_table, _MAX_DIM, dec_table
from hironaka.jax.loss import clip_log
from hironaka.src import get_newton_polytope_jax, rescale_jax, shift_jax, reposition_jax
from jax import jit, lax
from jax import numpy as jnp
from jax import vmap

Rollout = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]  # observations, policy logits, values

flatten = vmap(jnp.ravel, 0, 0)


def make_agent_obs(pts: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Combine points observation and coordinates into a flattened and concatenated agent observation.
    Parameters:
        pts: jax array of shape (batch_size, max_num_points, dimension)
        coords: jax multi-binary array of shape (batch_size, dimension)
    Returns:
        the combined observation for agent.
    """
    return jnp.concatenate([flatten(pts), coords], axis=1)


def get_dones(pts: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((pts[:, :, 0] >= 0), axis=1) < 2


def get_done_from_flatten(obs: jnp.ndarray, role: str, dimension: int) -> jnp.ndarray:
    return jnp.sum(obs >= 0, axis=-1) <= dimension + (role == "agent") * dimension


@functools.lru_cache()
def get_preprocess_fns(role: str, spec: Tuple[int, int]) -> Tuple[Callable, Callable]:
    """
    Parameters:
        role: host or agent
        spec: (max_num_points, dimension)
    Returns:
        A pair of functions corresponding to the observation preprocessing and coordinate preprocessing.
        Note: the reason we need preprocessing is that an observation is a 2d flattened array(and possibly
            concatenated with coordinates):
            - A host observation just flattens the 1 and 2 axis.
            - An agent observation is a host observation concatenated with a (-1, dimension) array
                of chosen coordinates
    """
    if role == "host":

        def obs_preprocess(observations):
            return observations.reshape(-1, *spec)

        def coords_preprocess(observations, actions):
            return actions

    elif role == "agent":

        def obs_preprocess(observations):
            return vmap(partial(lax.dynamic_slice, start_indices=(0,), slice_sizes=(spec[0] * spec[1],)), 0, 0)(
                observations
            ).reshape(-1, *spec)

        def coords_preprocess(observations, actions):
            return vmap(partial(lax.dynamic_slice, start_indices=(spec[0] * spec[1],), slice_sizes=(spec[1],)), 0, 0)(
                observations
            )

    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return jit(obs_preprocess), jit(coords_preprocess)


@functools.lru_cache()
def get_take_actions(role: str, spec: Tuple[int, int],
                     rescale_points: bool = False, reposition: bool = True) -> Callable:
    """
    Factory function that returns a `take_actions` function to perform observation update depending on the current role.
    Parameters:
        role: 'host' or 'agent'
        spec: (max_num_points, dimension)
        rescale_points: whether to do an L0-rescale at the end.
        reposition: whether to shift the minimal coordinate of each coordinate to 0 (-> ignoring exceptional divisor).
    Returns:
        the `take_actions` function.
    """

    obs_preprocess, coords_preprocess = get_preprocess_fns(role, spec)

    def take_actions(observations: jnp.ndarray, actions: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
        """
        Shift a batch of points depending on the `role` in `get_recurrent_fn_for_role`. Rules are described as below:
        role == 'host',
            points = observations
            coords = actions
            axis = axis
        role == 'agent',
            jnp.concatenate((points, coords), axis=1) is the observation (need to slice it up)
            (actions is not used but the best practice is to keep actions==axis)
            axis = axis
        Parameters:
            observations: see above
            actions: see above
            axis: see above
        Returns:
            The resulting batch of points, flattened into 2d.
            (NOT the concatenated observation when role is 'agent'!)
        """
        points = obs_preprocess(observations)
        coords = coords_preprocess(observations, actions)
        shifted_pts = shift_jax(points, coords, axis)
        shifted_pts = jnp.where(reposition, reposition_jax(shifted_pts), shifted_pts)
        shifted_pts = get_newton_polytope_jax(shifted_pts)
        maybe_rescaled = jnp.where(rescale_points, rescale_jax(shifted_pts), shifted_pts).reshape((-1, spec[0] * spec[1]))
        return maybe_rescaled

    return take_actions


@functools.lru_cache()
def get_reward_fn(role: str) -> Callable:
    """
    Parameters:
        role: host or agent.
    Returns:
        the corresponding reward function.
    """
    if role == "host":

        def reward_fn(dones: jnp.ndarray, prev_dones: jnp.ndarray) -> jnp.ndarray:
            return dones.astype(jnp.float32)

    elif role == "agent":

        def reward_fn(dones: jnp.ndarray, prev_dones: jnp.ndarray) -> jnp.ndarray:
            return -dones.astype(jnp.float32)

    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return reward_fn


@functools.lru_cache()
def get_feature_fn(role: str, spec: Tuple, scale_observation=True) -> Callable:
    """
    Get the feature function on (possibly flattened) observations.
    Parameters:
        role: host or agent.
        spec: (max_num_points, dimension).
        scale_observation: (Optional) normalize the points.
    Returns:
        the feature function that extract the features from a batch of point states
    """
    assert len(spec) == 2
    maybe_rescale = rescale_jax if scale_observation else lambda x: x

    def order_and_rescale(x: jnp.ndarray) -> jnp.ndarray:
        """
        Assume x fits into shape (-1, *spec), it will
        - reshape x into (-1, spec[0], spec[1]).
        - (possibly) rescale the points.
        - for each slice on the 0-th axis, order the spec[0] rows (*, *, :spec[1]) from high to low lexicographically.
        - flatten the resulting (-1, *spec) array into (-1, spec[0]*spec[1]) and return.
        """
        x_reshaped = maybe_rescale(x.reshape(-1, *spec))
        idx = vmap(partial(jnp.lexsort, axis=-1), 0, 0)(-x_reshaped.transpose((0, 2, 1)))
        return flatten(jnp.take_along_axis(x_reshaped, idx[:, :, None], axis=1))

    if role == "host":

        def feature_fn(observations: jnp.ndarray) -> jnp.ndarray:
            return order_and_rescale(observations)

    elif role == "agent":
        obs_preprocess, coords_preprocess = get_preprocess_fns("agent", spec)

        def feature_fn(observations: jnp.ndarray) -> jnp.ndarray:
            points = order_and_rescale(obs_preprocess(observations))
            coords = coords_preprocess(observations, None)
            return make_agent_obs(points, coords)

    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return feature_fn


@functools.lru_cache()
def get_dynamic_policy_fn(spec: Tuple[int, int], host_fn: Callable, agent_fn: Callable) -> Callable:
    """
    Get a dynamic policy function:
        - the input dimension is always (batch_size, (max_num_points + 1) * dimension).
        - it will determine whether it is a host or agent state based on the last `dimension` entries.
            host state will have them padded with 0.
            agent state will have a proper action mask.
        - it will then select the correct policy (host_fn or agent_fn) and evaluate on the input.
    The Dynamic policy function is used when we unify host/agent nodes in an MC tree search (when the `role_agnostic`
        switch is on in get_evaluation_loop and get_simulation functions).
    Parameters:
        spec: (max_num_points, dimension)
        host_fn: the host function that takes in a point state and returns (policy, value) pair.
        agent_fn: the agent function that has similar input/output as above.
    Returns:
        a function that takes in:
            - a uniformized point state of shape (batch_size, (max_num_points + 1) * dimension),
            - a pair of (host_args, agent_args), where each of them usually contains the model parameters.
            - and the rest of *args, **kwargs.
            and it applies host_fn or agent_fn dynamically.
        Note: the agent action space will be padded to the same length as host: 2**dimension-dimension-1 by -jnp.inf.
    """
    obs_preprocess, coords_preprocess = get_preprocess_fns('agent', spec)
    extra_action_dim = 2 ** spec[1] - 2 * spec[1] - 1

    def use_policy_fn(state):
        coord = coords_preprocess(state, None)
        return jnp.any(jnp.all(jnp.isclose(coord, 0), axis=-1), axis=None)

    def agent_padded(*args, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        policy, value = agent_fn(*args, **kwargs)
        return jnp.pad(policy, ((0, 0), (0, extra_action_dim)),
                       mode='constant', constant_values=(0, -jnp.inf)), value

    def dynamic_policy_fn(state, host_and_agent_args, *args, **kwargs):
        host_args, agent_args = host_and_agent_args
        return lax.cond(use_policy_fn(state),
                        lambda x: host_fn(x, *host_args, *args, **kwargs),
                        lambda x: agent_padded(x, *agent_args, *args, **kwargs),
                        state)
    return dynamic_policy_fn


def calculate_value_using_reward_fn(
    done: jnp.ndarray, prev_done: jnp.ndarray, discount: float, max_length_game: int, reward_fn: Callable
) -> jnp.ndarray:
    reward = vmap(reward_fn, (0, 0), 0)(done, prev_done)
    diff = jnp.arange(max_length_game).reshape((1, -1)) - jnp.arange(max_length_game).reshape((-1, 1))
    discount_table = (discount**diff) * (diff >= 0) + (diff < 0)
    discounted_value = vmap(jnp.matmul, (None, 0), 0)(discount_table, reward)
    return discounted_value


def apply_agent_action_mask(agent_policy: Callable, dimension: int) -> Callable:
    """
    Apply a masked agent policy wrapper on top of an `agent_policy` function.
    Assumption: the input x has shape (..., feature_dim), i.e., all features are flattened to the last axis.
    """

    def masked_agent_policy(x: jnp.ndarray, *args, **kwargs) -> Tuple:
        feature_num = x.shape[-1]
        # get the start and end for axis before the last one
        start = (0,) * (len(x.shape) - 1)
        end = x.shape[:-1]
        # Extract the last `dimension` entries as our action mask, and apply it to the final action.
        # Expect a 0/1 array, possibly float dtype. But to err on the safe side, we take a cut-off at 0.5.
        mask = lax.dynamic_slice(x, (*start, feature_num - dimension), (*end, dimension)) > 0.5
        policy_prior, value_prior = agent_policy(x, *args, **kwargs)
        return policy_prior * mask - jnp.inf * (~mask), value_prior

    masked_agent_policy.__name__ = get_name(agent_policy)
    return masked_agent_policy


def action_wrapper(policy_value_fn: Callable, dimension: Optional[Union[int]] = None) -> Callable:
    """
    Turns a policy function (returning (policy_logits, value)) into a function that returns one-hot actions.
    Depending on the value of `dimension`, it can apply action mask based on the last `dimension` entries in the input
        tensor. Used in the inference of agent actions.
    Parameters:
        policy_value_fn: the policy function that returns the (policy, value) pair.
        dimension: (Optional) the dimension of action output.
            if nonzero, will first extract the last `dimension` entries from the input and apply it as an action mask.
            if None, will not apply action mask.
    """
    masked_action = policy_value_fn if dimension is None else apply_agent_action_mask(policy_value_fn, dimension)

    def wrapped_action_fn(x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        out, _ = masked_action(x, *args, **kwargs)
        action_dim = out.shape[-1]
        return jax.nn.one_hot(jnp.argmax(out, axis=-1), action_dim)

    wrapped_action_fn.__name__ = get_name(policy_value_fn)
    return wrapped_action_fn


def mcts_wrapper(eval_loop: Callable) -> Callable:
    """
    Wrap around an eval loop to turn it into a function that returns policy logits and value.
    Parameters:
        eval_loop: the original eval loop that returns the MCTS output.
    Returns:
        a function that returns (policy, value) pair.
    """
    def mcts_wrapped_policy(x: jnp.ndarray, params: FrozenDict, opp_params: FrozenDict, key: jnp.ndarray) -> jnp.ndarray:
        policy_output = eval_loop(key, x, (params,), (opp_params,))
        return clip_log(policy_output.action_weights), policy_output.search_tree.node_values[:, 0]
    return mcts_wrapped_policy


def get_name(obj):
    if hasattr(obj, "__name__"):
        return obj.__name__
    elif hasattr(obj, "func"):  # wrapped by `partial`
        return get_name(obj.func)


def select_sample_after_sim(role: str, rollout: Rollout, dimension: int, mix_random_terminal_states=True, key=None) -> Rollout:
    """
    After samples are generated by simulations, some states are still of ongoing games, and some states are terminal.
    We select those that are unfinished, plus optionally a random collection of terminal states in the following way:
        Say there are N unfinished states. To shuffle in a balanced amount of terminal states while not exceeding the
        total sample size, we uniformly sample N indices with replacement out of all indices, and include those in the
        sample set. So when N is small, we guarantee a minimum of 1:1 ongoing and terminal states. When N is large, we
        roughly get diminishing terminal states.
    If mix_random_terminal_states is un-toggled, we will only look at unfinished games.
    Parameters:
        rollout: the raw collection of rollouts. Shape: (sample_size, input_dim).
        role: host or agent.
        dimension: the dimension.
        mix_random_terminal_states: (Optional) whether to mix in a bunch of random terminal states in the end.
        key: (Optional) the PRNGKey. If None, use `time.time_ns()`.
    Returns:
        a mask of rollouts corresponding to the ones that are chosen. Shape: (sample_size,).
    """
    key = jnp.where(key is None, jax.random.PRNGKey(time.time_ns()), key)
    size = rollout[0].shape[0]
    offset = jnp.where(role == "agent", dimension, 0)

    undone_idx = jnp.sum(rollout[0] >= 0, axis=-1) > (dimension + offset)
    # jit does not allow variable array size (like this `undone_sum`). So this is a work-around to sample exactly
    #   `undone_num` indices and set them to True in the final result.
    if mix_random_terminal_states:
        undone_num = jnp.sum(undone_idx)
        random_idx = jax.random.choice(key, jnp.arange(size), (size,), False)
        selected_idx = undone_idx | (random_idx < undone_num)
    else:
        selected_idx = undone_idx
    return selected_idx


@partial(jax.pmap, static_broadcasted_argnums=(1, 2, 3, 4, 5))
def generate_pts(key: jnp.ndarray, shape: Tuple, max_value: int, dtype=jnp.float32,
                 rescale=True, reposition=True) -> jnp.ndarray:
    pts = jax.random.randint(key, shape, 0, max_value).astype(dtype)
    pts = get_newton_polytope_jax(pts)
    pts = jnp.where(reposition, reposition_jax(pts), pts)
    pts = jnp.where(rescale, rescale_jax(pts), pts)
    return pts


def rollout_sanity_tests(rollout: Rollout, spec: Tuple[int, int]) -> bool:
    """
    Given a rollout (observation, policy, value), this tests a few things:
        - whether the action masks are properly applied.
        - whether the policy *MIGHT* already be the softmax (all added up to 1 and between [0, 1]).
    Parameters:
        rollout: the rollout.
        spec: (max_num_points, dimension)
    Returns:
        True if it passes the sanity test.
    """
    obs, policy, value = rollout
    # Check masks.
    if obs.shape[-1] == (spec[0] + 1) * spec[1]:
        mask = obs[..., -spec[1]:] > 0.5
        # In unified states, host observations are padded by 0 which is impossible for agent states to have.
        is_host = jnp.expand_dims(jnp.all(jnp.isclose(mask, 0.0), axis=-1), axis=-1)
        # Sometimes, the action space is padded to certain length. Mark the extra dimensions to be False in the mask.
        mask = jnp.concatenate([mask + is_host, jnp.repeat(is_host, policy.shape[-1] - spec[1], axis=-1)], axis=-1)

        if jnp.any((policy * ~mask - mask * jnp.inf) != -jnp.inf):
            return False

    # Check softmax policy.
    # You need to have God's blessing to have logits that add up to 1 and all between 0, 1 for all states in a batch.
    if jnp.all(jnp.isclose(jnp.sum(policy, axis=-1), 1.0)) and jnp.all((policy <= 1.0) & (policy >= 0.0)):
        return False

    return True


def safe_clip_grads(grad_tree, max_norm):
    norm = l2_norm(grad_tree)
    eps = 1e-9
    normalize = lambda g: jnp.where(norm < max_norm, g, g * max_norm / (norm + eps))
    return jax.tree_util.tree_map(normalize, grad_tree)
