from functools import partial
from typing import Tuple, Callable

import flax
import jax.numpy as jnp
from flax.core import FrozenDict

from jax import vmap, jit, lax, numpy as jnp

from hironaka.src import rescale_jax, get_newton_polytope_jax, shift_jax

flatten = vmap(jnp.ravel, 0, 0)


@jit
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


@jit
def get_dones(pts: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((pts[:, :, 0] >= 0), axis=1) < 2


'''
@partial(jit, static_argnames=['enemy_role'])
def make_obs(enemy_role: str, observations: jnp.ndarray, enemy_actions: jnp.ndarray) -> jnp.ndarray:
    """
    Create the correct format of observations when the *ENEMY* is `role`.
    It is a helper function for `get_recurrent_fn_for_role`.
    """
    return jnp.where(enemy_role == 'host',
                     make_agent_obs(observations, enemy_actions),
                     observations)
'''


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
    if role == 'host':
        def obs_preprocess(observations): return observations.reshape(-1, *spec)
        def coords_preprocess(observations, actions): return actions
    elif role == 'agent':
        def obs_preprocess(observations):
            return vmap(partial(lax.dynamic_slice, start_indices=(0,), slice_sizes=(spec[0] * spec[1],)), 0, 0)(
                observations).reshape(-1, *spec)

        def coords_preprocess(observations, actions):
            return vmap(partial(lax.dynamic_slice, start_indices=(spec[0] * spec[1],), slice_sizes=(spec[1],)), 0, 0)(
                observations)
    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return jit(obs_preprocess), jit(coords_preprocess)


def get_take_actions(role: str, spec: Tuple[int, int], rescale_points: bool = True) -> Callable:
    """
    Factory function that returns a `take_actions` function to perform observation update depending on the current role.
    Parameters:
        role: 'host' or 'agent'
        spec: (max_num_points)
        rescale_points: whether to do an L0-rescale at the end.
    Returns:
        the `take_actions` function.
    """

    obs_preprocess, coords_preprocess = get_preprocess_fns(role, spec)

    @jit
    def take_actions(observations: jnp.ndarray, actions: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
        """
        Shift a batch of points depending on the `role` in `get_recurrent_fn_for_role`. Rules are described as below:
        role == 'host',
            points = observations
            coords = actions
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
        return jnp.where(rescale_points, rescale_jax(shifted_pts), shifted_pts).reshape((-1, spec[0]*spec[1]))
    return take_actions


def get_reward_fn(role: str) -> Callable:
    """
    Parameters:
        role: host or agent.
    Returns:
        the corresponding reward function.
    """
    if role == 'host':
        @jit
        def reward_fn(dones: jnp.ndarray, prev_dones: jnp.ndarray) -> jnp.ndarray:
            return (dones & (~prev_dones)).astype(jnp.float32)
    elif role == 'agent':
        @jit
        def reward_fn(dones: jnp.ndarray, prev_dones: jnp.ndarray) -> jnp.ndarray:
            return -(dones & (~prev_dones)).astype(jnp.float32)
    else:
        raise ValueError(f"role must be either host or agent. Got {role}.")

    return reward_fn


@jit
def extra_features(role: str, observations: jnp.ndarray) -> jnp.ndarray:
    """
    Given a role and an observation, return an extra array of shape (batch_size, *) as an extra set of features.
    """
    return None


"""
# TODO
root = mctx.RootFnOutput(
        prior_logits=jnp.array([[0] * (2 ** dimension - dimension - 1)] * batch_size, dtype=jnp.float32),
        value=jnp.array([0] * batch_size, dtype=jnp.float32),
        embedding=jax.random.randint(key, (batch_size, max_num_points, dimension), 0, 20).astype(jnp.float32))
"""


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
                binary_vector.append(int(binary_str[- j - 1]))
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


@jit
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


@jit
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


