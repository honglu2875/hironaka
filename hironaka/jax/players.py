"""
Fixed Host and Agent policies are implemented as functions here.
"""
import time
from functools import lru_cache, partial
from typing import Tuple

import jax
from jax import vmap, lax, jit
import jax.numpy as jnp

from hironaka.src import get_newton_polytope_jax, shift_jax, rescale_jax

sub_2_2 = vmap(vmap(jnp.subtract, (None, 0), 0), (0, None), 0)  # (n, d) - (m, d) -> (n, m, d) subtract each vector


def decode_table(dimension: int) -> jnp.ndarray:
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
        res.append(jnp.array(binary_vector))
    return jnp.array(res).astype(jnp.float32)


# The decoding table will be generated the first time running functions in this file.
# Remark: When `_MAX_DIM` goes up, the computation and memory become very very costly.
#   To scale up further, the only way is to predict multi-binary vectors instead of
#   having discrete action of size 2**dim-dim-1.

_MAX_DIM = 10
dec_table = [None, None]
for i in range(2, _MAX_DIM):
    dec_table.append(decode_table(i))


def decode(one_hot: jnp.ndarray) -> jnp.ndarray:
    """
    Decode a single one hot vector into a multi-binary vector.
    E.g., [0, 0, 1, 0] is decoded into [0, 1, 1].
    """
    cls = jnp.argmax(one_hot)
    cls_num = one_hot.shape[0]
    dimension = lax.cond(cls_num < 4,
                         lambda: 2,
                         lambda: jnp.log2(cls_num).astype(jnp.int32) + 1)
    return dec_table[dimension][cls]


batch_decode = vmap(decode, 0, 0)


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


# ---------- Host functions ---------- #
# List of hosts:
#   random_host_fn(pts, key=0, dtype=jnp.float32)
#   all_coord_host_fn(pts, dtype=jnp.float32)
#   zeillinger_fn(pts, dtype=jnp.float32)


def random_host_fn(pts: jnp.ndarray, key=0, dtype=jnp.float32) -> jnp.ndarray:
    """
    Parameters:
        pts: points (batch_size, max_num_points, dimension)
        key: RNG random key
        dtype: data type
    Return:
        host action as one-hot array.
    """
    batch_size, max_num_points, dimension = pts.shape
    key = lax.cond(key == 0,
                   lambda: jax.random.PRNGKey(time.time_ns()),
                   lambda: jax.random.PRNGKey(key))
    cls_num = 2 ** dimension - dimension - 1
    return jax.nn.one_hot(jax.random.randint(key, (batch_size,), 0, cls_num), cls_num, dtype=dtype)


def all_coord_host_fn(pts: jnp.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    """
    Parameters:
        pts: points (batch_size, max_num_points, dimension)
        dtype: data type
    Return:
        host action as one-hot array.
    """
    batch_size, max_num_points, dimension = pts.shape
    cls_num = 2 ** dimension - dimension - 1
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
            S: num of max coordinate + num of min coordinate
    """
    diff = v1 - v2
    maximal = jnp.max(diff)
    minimal = jnp.min(diff)
    max_count = jnp.sum(diff == maximal)
    min_count = jnp.sum(diff == minimal)
    return jnp.array(
        [maximal - minimal, lax.cond(maximal == minimal, lambda: max_count, lambda: max_count + min_count)])


# (n, d) - (m, d) -> (n, m, 2) characteristic vector of each pair
char_vector_of_pts = vmap(vmap(char_vector, (None, 0), 0), (0, None), 0)


def zeillinger_fn_slice(pts: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Zeillinger policy on one single set of points (batch size = 1 and ignore the batch axis).
    Parameters:
        pts: points (max_num_points, dimension)
    Return:
        host action as one-hot array.
    """
    n, d = pts.shape  # (n, d)
    char_vec = char_vector_of_pts(pts, pts)  # (n, n, 2)
    # Bump up the diagonal entries so that they do not show up when finding minimal.
    maximal = jnp.max(char_vec)
    diag = jnp.stack([jnp.diag(jnp.full((n,), maximal)),
                      jnp.diag(jnp.full((n,), maximal))],
                     axis=2)
    bumped_char_vec = (char_vec + diag).reshape(-1, 2)
    min_index = jnp.lexsort((bumped_char_vec[:, 1], bumped_char_vec[:, 0]))[0]
    diff = sub_2_2(pts, pts).reshape(-1, d)  # (n, n, d) -> (n*n, d)
    minimal_diff_vector = diff[min_index, :]
    # (min, max) of the minimal_diff_vector is the Zeillinger's choice
    argmin = jnp.argmin(minimal_diff_vector)
    argmax = jnp.argmax(minimal_diff_vector)
    multi_bin = (jnp.arange(d) == argmin) | (jnp.arange(d) == argmax)
    return lax.cond(argmin != argmax,
                    lambda: encode_one_hot(multi_bin.astype(jnp.float32)),
                    lambda: jax.nn.one_hot(0, 2 ** d - d - 1))


def zeillinger_fn(pts: jnp.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    return vmap(zeillinger_fn_slice, 0, 0)(pts).astype(dtype)


# ---------- Agent functions ---------- #
# Agent list:
#   random_agent_fn(pts, spec, key=0, dtype=jnp.float32)

def random_agent_fn(pts: jnp.ndarray, spec: Tuple, key=0, dtype=jnp.float32) -> jnp.ndarray:
    """
    Parameters:
        pts: flattened and concatenated points (batch_size, max_num_points * dimension + dimension)
        spec: tuple specifying (max_num_points, dimension)
        key: RNG random key
        dtype: data type
    Return:
        agent action as one-hot array.
    """
    max_num_points, dimension = spec
    batch_size = pts.shape[0]
    key = lax.cond(key == 0,
                   lambda: jax.random.PRNGKey(time.time_ns()),
                   lambda: jax.random.PRNGKey(key))

    return jax.nn.one_hot(jax.random.randint(key, (batch_size,), 0, dimension), dimension, dtype=dtype)


