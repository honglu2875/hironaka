import functools
from functools import partial
from typing import Callable

from jax import numpy as jnp, vmap


def decode_table(dimension: int) -> dict:
    """
    Return a decoding table. The i-th row is the corresponding multi-binary vector.
    """
    res = []
    for i in range(2**dimension):
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


@functools.lru_cache()
def get_batch_decode(dimension: int) -> Callable:
    """
    The factory function of getting a batch decoder function with given dimension.
    """
    if dimension >= _MAX_DIM:
        raise ValueError(f"Dimension is capped at {_MAX_DIM}. Got {dimension}.")
    return vmap(partial(decode, lookup_dict=dec_table[dimension]), 0, 0)


@functools.lru_cache()
def get_batch_decode_from_one_hot(dimension: int) -> Callable:
    """
    The factory function of getting a batch decoder (from one-hot vectors) function with given dimension.
    """
    if dimension >= _MAX_DIM:
        raise ValueError(f"Dimension is capped at {_MAX_DIM}. Got {dimension}.")
    return vmap(partial(decode_from_one_hot, lookup_dict=dec_table[dimension]), 0, 0)


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
    class_num = 2**dimension - dimension - 1
    return (jnp.arange(class_num) == encode(multi_binary)).astype(jnp.float32)


batch_encode = vmap(encode, 0, 0)
batch_encode_one_hot = vmap(encode_one_hot, 0, 0)
