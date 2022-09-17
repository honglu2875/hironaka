"""
Fixed Host and Agent policies are implemented as functions here.
Most of them can be jitted once dtype is fixed using `partial` from functools.
List of hosts:
    random_host_fn(pts, dtype=jnp.float32, key=jnp.array([0, 0]))
    all_coord_host_fn(pts, dtype=jnp.float32, **kwargs)
    zeillinger_fn(pts, dtype=jnp.float32, **kwargs)
List of agents:
    random_agent_fn(pts, spec, dtype=jnp.float32, key=jnp.array([0, 0]))
    choose_first_agent_fn(pts, spec, dtype=jnp.float32, **kwargs)
    choose_last_agent_fn(pts, spec, dtype=jnp.float32, **kwargs)
"""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from hironaka.jax.util import encode_one_hot, get_name

sub_2_2 = vmap(vmap(jnp.subtract, (None, 0), 0), (0, None), 0)  # (n, d) - (m, d) -> (n, m, d) subtract each vector
default_key = jnp.array([0, 0], dtype=jnp.uint32)

# ---------- Host functions ---------- #


def random_host_fn(pts: jnp.ndarray, key=default_key, dtype=jnp.float32, **kwargs) -> jnp.ndarray:
    """
    Parameters:
        pts: points (batch_size, max_num_points, dimension)
        key: RNG random key
        dtype: data type
    Return:
        host action as one-hot array.
    """
    batch_size, max_num_points, dimension = pts.shape
    cls_num = 2**dimension - dimension - 1
    return jax.nn.one_hot(jax.random.randint(key, (batch_size,), 0, cls_num), cls_num, dtype=dtype)


def all_coord_host_fn(pts: jnp.ndarray, dtype=jnp.float32, **kwargs) -> jnp.ndarray:
    """
    Parameters:
        pts: points (batch_size, max_num_points, dimension)
        dtype: data type
    Return:
        host action as one-hot array.
    """
    batch_size, max_num_points, dimension = pts.shape
    cls_num = 2**dimension - dimension - 1
    return jax.nn.one_hot(jnp.full((batch_size,), cls_num - 1), cls_num, dtype=dtype)


@jit
def char_vector(v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Get characteristic vectors.
    Parameters:
        v1: vector to subtract from
        v2: vector to be subtracted
    Return:
        the characteristic vector (L, S) corresponding to the difference v1 - v2.
            L: max coordinate - min coordinate
            S: num of max coordinates + num of min coordinates (no repeating counts)
    """
    diff = v1 - v2
    maximal = jnp.max(diff)
    minimal = jnp.min(diff)
    max_count = jnp.sum(diff == maximal)
    min_count = jnp.sum(diff == minimal)
    return jnp.where(
        jnp.any(v1 < 0) | jnp.any(v2 < 0) | jnp.isclose(maximal, minimal),
        jnp.array([jnp.inf, jnp.inf]),
        jnp.array([maximal - minimal, jnp.where(maximal == minimal, max_count, max_count + min_count)]),
    )


# (n, d) - (m, d) -> (n, m, 2) characteristic vector of each pair
char_vector_of_pts = vmap(vmap(char_vector, (None, 0), 0), (0, None), 0)


@jit
def zeillinger_fn_slice(pts: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Zeillinger policy on one single set of points (batch size = 1 and ignore the batch axis).
    Parameters:
        pts: points (max_num_points, dimension)
    Return:
        host action as one-hot array.
    """
    n, d = pts.shape  # (n, d)
    char_vec = char_vector_of_pts(pts, pts).reshape(-1, 2)  # (n, n, 2)
    # Bump up the diagonal entries (only place that can have 0 entries)
    #   so that they do not show up when finding minimal.

    min_index = jnp.lexsort((char_vec[:, 1], char_vec[:, 0]))[0]
    diff = sub_2_2(pts, pts).reshape(-1, d)  # (n, n, d) -> (n*n, d)
    minimal_diff_vector = diff[min_index, :]
    # (min, max) of the minimal_diff_vector is the Zeillinger's choice
    argmin = jnp.argmin(minimal_diff_vector)
    argmax = jnp.argmax(minimal_diff_vector)
    multi_bin = (jnp.arange(d) == argmin) | (jnp.arange(d) == argmax)

    return jnp.where(argmin != argmax, encode_one_hot(multi_bin), jax.nn.one_hot(0, 2**d - d - 1))


def zeillinger_fn(pts: jnp.ndarray, dtype=jnp.float32, **kwargs) -> jnp.ndarray:
    return vmap(zeillinger_fn_slice, 0, 0)(pts).astype(dtype)


def get_host_with_flattened_obs(spec, func, dtype=jnp.float32) -> Callable:
    def func_flatten(pts, dtype=dtype, **kwargs):
        return func(pts.reshape(-1, *spec), dtype=dtype, **kwargs)

    func_flatten.__name__ = get_name(func))
    return func_flatten


# ---------- Agent functions ---------- #


@partial(jit, static_argnames=["spec", "dtype"])
def random_agent_fn(pts: jnp.ndarray, spec: Tuple, key=default_key, dtype=jnp.float32, **kwargs) -> jnp.ndarray:
    """
    Parameters:
        pts: flattened and concatenated points (batch_size, max_num_points * dimension + dimension)
        spec: tuple specifying (max_num_points, dimension)
        key: RNG random key
        dtype: data type
    Return:
        agent action as one-hot array.
    """
    (max_num_points, dimension), batch_size = spec, pts.shape[0]
    return jax.nn.one_hot(jax.random.randint(key, (batch_size,), 0, dimension), dimension, dtype=dtype)


def choose_first_agent_fn_slice(pts: jnp.ndarray, spec: Tuple) -> jnp.ndarray:
    """
    Choose the first action in the set of host coordinates:
    (Assume the convention that host coordinate is
        pts[max_num_points * dimension: max_num_points * dimension + dimension])
    Parameters:
        pts: a single set of points without batch axis. Shape (max_num_points * dimension + dimension, )
        spec: specifying (max_num_points, dimension)
    Returns:
        a one-hot vector corresponding to the first axis chosen by host
    """
    max_num_points, dimension = spec
    host_action = lax.dynamic_slice(pts, [max_num_points * dimension], [dimension])
    first_host_action = jnp.argmax(host_action)
    return (jnp.arange(dimension) == first_host_action).astype(jnp.float32)


@partial(jit, static_argnames=["spec", "dtype"])
def choose_first_agent_fn(pts: jnp.ndarray, spec: Tuple, dtype=jnp.float32, **kwargs) -> jnp.ndarray:
    """
    Parameters:
        pts: flattened and concatenated points (batch_size, max_num_points * dimension + dimension)
        spec: tuple specifying (max_num_points, dimension)
        dtype: data type
    Return:
        agent action as one-hot array.
    """
    return vmap(partial(choose_first_agent_fn_slice, spec=spec), 0, 0)(pts).astype(dtype)


def choose_last_agent_fn_slice(pts: jnp.ndarray, spec: Tuple) -> jnp.ndarray:
    """
    Choose the last action in the set of host coordinates:
    (Assume the convention that host coordinate is
        pts[max_num_points * dimension: max_num_points * dimension + dimension])
    Parameters:
        pts: a single set of points without batch axis. Shape (max_num_points * dimension + dimension, )
        spec: specifying (max_num_points, dimension)
    Returns:
        a one-hot vector corresponding to the first axis chosen by host
    """
    max_num_points, dimension = spec
    host_action = lax.dynamic_slice(pts, [max_num_points * dimension], [dimension])
    eps = 1e-5  # Assumes dimension is never higher than 1e5
    last_host_action = jnp.argmax(host_action + jnp.arange(dimension) * eps)
    return (jnp.arange(dimension) == last_host_action).astype(jnp.float32)


@partial(jit, static_argnames=["spec", "dtype"])
def choose_last_agent_fn(pts: jnp.ndarray, spec: Tuple, dtype=jnp.float32, **kwargs) -> jnp.ndarray:
    """
    Parameters:
        pts: flattened and concatenated points (batch_size, max_num_points * dimension + dimension)
        spec: tuple specifying (max_num_points, dimension)
        dtype: data type
    Return:
        agent action as one-hot array.
    """
    return vmap(partial(choose_last_agent_fn_slice, spec=spec), 0, 0)(pts).astype(dtype)
