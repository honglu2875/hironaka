import time
from functools import partial
from typing import Callable, Tuple, Union

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import vmap

from hironaka.src import get_newton_polytope_jax, rescale_jax, shift_jax

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
    return jnp.sum(obs >= 0, axis=-1) <= dimension + (role == "agent") * (2 ** dimension - dimension - 1)


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


def get_take_actions(role: str, spec: Tuple[int, int], rescale_points: bool = True) -> Callable:
    """
    Factory function that returns a `take_actions` function to perform observation update depending on the current role.
    Parameters:
        role: 'host' or 'agent'
        spec: (max_num_points, dimension)
        rescale_points: whether to do an L0-rescale at the end.
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
        shifted_pts = get_newton_polytope_jax(shift_jax(points, coords, axis))
        maybe_rescaled = jnp.where(rescale_points, rescale_jax(shifted_pts), shifted_pts).reshape(
            (-1, spec[0] * spec[1]))
        return maybe_rescaled

    return take_actions


def get_reward_fn(role: str) -> Callable:
    """
    Parameters:
        role: host or agent.
    Returns:
        the corresponding reward function.
    """
    if role == "host":

        @jit
        def reward_fn(dones: jnp.ndarray, prev_dones: jnp.ndarray) -> jnp.ndarray:
            return (dones & (~prev_dones)).astype(jnp.float32)

    elif role == "agent":

        @jit
        def reward_fn(dones: jnp.ndarray, prev_dones: jnp.ndarray) -> jnp.ndarray:
            return -(dones & (~prev_dones)).astype(jnp.float32)

    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return reward_fn


def get_feature_fn(role: str, spec: Tuple) -> Callable:
    """
    Get the feature function on (possibly flattened) observations.
    """
    if role == "host":

        @jit
        def feature_fn(observations: jnp.ndarray) -> jnp.ndarray:
            return -flatten(vmap(partial(jnp.sort, axis=0), 0, 0)(-observations.reshape(-1, *spec)))

    elif role == "agent":
        obs_preprocess, coords_preprocess = get_preprocess_fns("agent", spec)

        @jit
        def feature_fn(observations: jnp.ndarray) -> jnp.ndarray:
            points = -flatten(vmap(partial(jnp.sort, axis=0), 0, 0)(-obs_preprocess(observations)))
            coords = coords_preprocess(observations, None)
            return make_agent_obs(points, coords)

    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return feature_fn


def calculate_value_using_reward_fn(
        done: jnp.ndarray, prev_done: jnp.ndarray, discount: float, max_length_game: int, reward_fn: Callable
) -> jnp.ndarray:
    reward = vmap(reward_fn, (0, 0), 0)(done, prev_done)
    diff = jnp.arange(max_length_game).reshape((1, -1)) - jnp.arange(max_length_game).reshape((-1, 1))
    discount_table = (discount ** diff) * (diff >= 0)
    discounted_value = vmap(jnp.matmul, (None, 0), 0)(discount_table, reward)
    return discounted_value


def apply_agent_action_mask(agent_policy: Callable, dimension: int) -> Callable:
    """
    Apply a masked agent policy wrapper on top of an `agent_policy` function.
    """

    def masked_agent_policy(x: jnp.ndarray, *args, **kwargs) -> Tuple:
        batch_size, feature_num = x.shape
        mask = lax.dynamic_slice(x, (0, feature_num - dimension), (batch_size, dimension)) > 0.5
        policy_prior, value_prior = agent_policy(x, *args, **kwargs)
        return policy_prior * mask - jnp.inf * (~mask), value_prior

    masked_agent_policy.__name__ = get_name(agent_policy)
    return masked_agent_policy


def action_wrapper(policy_value_fn: Callable, dimension: Union[int, None]) -> Callable:
    """
    Turns a policy function (returning (policy_logits, value)) into a function that returns one-hot actions.
    """
    masked_action = policy_value_fn if dimension is None else apply_agent_action_mask(policy_value_fn, dimension)

    def wrapped_action_fn(x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        out, _ = masked_action(x, *args, **kwargs)
        action_dim = out.shape[1]
        return jax.nn.one_hot(jnp.argmax(out, axis=1), action_dim)

    wrapped_action_fn.__name__ = get_name(policy_value_fn)
    return wrapped_action_fn


def get_name(obj):
    if hasattr(obj, "__name__"):
        return obj.__name__
    elif hasattr(obj, "func"):  # wrapped by `partial`
        return get_name(obj.func)


def select_sample_after_sim(rollout: Rollout, role: str, dimension: int, key=None) -> Rollout:
    """
    After samples are generated by simulations, select those that are not finished, plus a random collection of
        finished states (Say there are N unfinished states. Randomly sample N indices with replacement out of all
        indices, and include those in the sample set) to preserve a balance of samples (sometimes too many games end
        early and samples would skew heavily towards finished states).
    Parameters:
        rollout: the raw collection of rollouts.
        role: host or agent.
        dimension: the dimension.
        key: (Optional) the PRNGKey. If None, use `time.time_ns()`.
    Returns:
        a collection of processed rollouts according to the descriptions.
    """
    key = jax.random.PRNGKey(time.time_ns()) if key is None else key
    size = rollout[0].shape[0]
    offset = (2 ** dimension - dimension - 1) if role == 'agent' else 0

    undone_idx = jnp.sum(rollout[0] >= 0, axis=-1) > dimension + offset
    undone_num = jnp.sum(undone_idx)
    random_idx = jax.random.randint(key, (undone_num,), 0, size)
    index = undone_idx.at[random_idx].set(True)
    return rollout[0][index], rollout[1][index], rollout[2][index]

# ---------- Encode/decode host actions ---------- #


def decode_table(dimension: int) -> dict:
    """
    Return a decoding table. The i-th row is the corresponding multi-binary vector.
    """
    res = []
    for i in range(2 ** dimension):
        if i == 0 or i & (i - 1) == 0:
            continue
        binary_str = bin(i)[2:]
        binary_vector = []
        for j in range(dimension):
            if j >= len(binary_str):
                binary_vector.append(0)
            else:
                binary_vector.append(int(binary_str[-j - 1]))
        res.append(jnp.array(binary_vector).astype(jnp.int32))
    return jnp.array(res)


_MAX_DIM = 11
dec_table = {0: None, 1: None}
# The decoding table will be generated the first time running functions in this file.
# Remark: When `_MAX_DIM` goes up, the computation and memory become very very costly.
#   To scale up further, the only way is to predict multi-binary vectors instead of
#   having discrete action of size 2**dim-dim-1.
#   Thus, for performance reason, we recommend to cap _MAX_DIM at 10.
for i in range(2, _MAX_DIM):
    dec_table[i] = decode_table(i)


def decode_from_one_hot(one_hot: jnp.ndarray, lookup_dict: jnp.ndarray) -> jnp.ndarray:
    """
    Decode a single one hot vector into a multi-binary vector, assuming a look-up dict is given (yes, I am cheating).
    The `lookup_dict` will be locked up as static when jitted with a factory function.
    E.g., [0, 0, 1, 0] is decoded into [0, 1, 1].
    """
    cls = jnp.argmax(one_hot)
    return lookup_dict[cls]


def decode(cls: int, lookup_dict: jnp.ndarray) -> jnp.ndarray:
    """
    Decode a single encoded host action number into a multi-binary vector, assuming a look-up dict is given.
    The `lookup_dict` will be locked up as static when jitted with a factory function.
    E.g., cls=2, dimension=3 is decoded into [0, 1, 1].
    """

    return lookup_dict[cls]


def get_batch_decode(dimension: int) -> Callable:
    """
    The factory function of getting a batch decoder function with given dimension.
    """
    if dimension >= _MAX_DIM:
        raise ValueError(f"Dimension is capped at {_MAX_DIM}. Got {dimension}.")
    return jit(vmap(partial(decode, lookup_dict=dec_table[dimension]), 0, 0))


def get_batch_decode_from_one_hot(dimension: int) -> Callable:
    """
    The factory function of getting a batch decoder (from one-hot vectors) function with given dimension.
    """
    if dimension >= _MAX_DIM:
        raise ValueError(f"Dimension is capped at {_MAX_DIM}. Got {dimension}.")
    return jit(vmap(partial(decode_from_one_hot, lookup_dict=dec_table[dimension]), 0, 0))


def encode(multi_binary: jnp.ndarray) -> int:
    """
    Encode a multi-binary vector into compressed class number.
    E.g., [1,0,1] is turned into 1 (second in the permissible actions: 3, 5, 6, 7).
    Return:
        int
    """
    dimension = multi_binary.shape[0]
    naive_binary = jnp.sum(2 ** jnp.arange(dimension) * multi_binary)
    return naive_binary - jnp.floor(jnp.log2(naive_binary)) - 2


def encode_one_hot(multi_binary: jnp.ndarray) -> jnp.ndarray:
    """
    Encode a multi-binary vector into the one-hot vector of compressed class.
    E.g. [1,0,1] is turned into [0,1,0,0].
    Return:
        jnp.ndarray with type jnp.float32
    """
    dimension = multi_binary.shape[0]
    class_num = 2 ** dimension - dimension - 1
    return (jnp.arange(class_num) == encode(multi_binary)).astype(jnp.float32)


batch_encode = vmap(encode, 0, 0)
batch_encode_one_hot = vmap(encode_one_hot, 0, 0)


# ---------- Loss functions ---------- #


def compute_loss(params, apply_fn, sample, loss_fn) -> jnp.ndarray:
    obs, target_policy_logits, target_value = sample
    policy_logits, value = apply_fn(obs, params)
    return loss_fn(policy_logits, value, target_policy_logits, target_value)


def policy_value_loss(
        policy_logit: jnp.ndarray, value: jnp.ndarray, target_policy: jnp.ndarray, target_value: jnp.ndarray
) -> jnp.ndarray:
    # Shapes:
    # policy_logit, target_policy: (B, action)
    # value_logit, target_value: (B,)
    policy_loss = jnp.sum(-jax.nn.softmax(target_policy, axis=-1) * jax.nn.log_softmax(policy_logit, axis=-1), axis=1)
    value_loss = jnp.square(value - target_value)
    return jnp.mean(policy_loss + value_loss)
