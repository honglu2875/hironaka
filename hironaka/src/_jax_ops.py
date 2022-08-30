from functools import partial

import jax.numpy as jnp
from jax import vmap, jit


def get_newton_polytope_approx_jax(points: jnp.ndarray, padding_value: float = -1.) -> jnp.ndarray:
    """
    A JAX implementation of the approximation of Newton polytopes.
    Note: we cannot do inplace operations due to being JAX arrays.
    """
    @jit
    def update(points_slice) -> jnp.ndarray:
        res = []
        max_num_points = points_slice.shape[0]
        for j in range(max_num_points):
            diff = vmap(partial(jnp.subtract, x2=points_slice[j, :]), 0, 0)(points_slice)
            res.append(jnp.any(jnp.all(diff < 0, axis = 1)))
        return jnp.array(res)

    for i in range(batch_size):


    return points if inplace else copied_points


def get_newton_polytope_jax(points: jnp.ndarray, padding_value: float = -1.):
    return get_newton_polytope_approx_jax(points, padding_value=padding_value)


