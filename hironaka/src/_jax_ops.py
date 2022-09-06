from functools import partial

import jax.numpy as jnp
from jax import vmap, jit, lax

mul_2_1 = vmap(lax.mul, (0, None), 0)  # (m, d) * (d,) -> (m, d)
mul_kronecker = vmap(vmap(lax.mul, (None, 0), 0), (0, None), 0)  # (m,) * (n,) -> (m, n) Kronecker product
and_kronecker = vmap(vmap(jnp.logical_and, (None, 0), 0), (0, None), 0)  # (m,) * (n,) -> (m, n) Kronecker bool
mul_3_2 = vmap(vmap(lax.mul, (1, None), 1), (0, 0), 0)  # (b, m, d) * (b, m) -> (b, m, d)
add_3_2 = vmap(vmap(lax.add, (1, None), 1), (0, 0), 0)  # (b, m, d) + (b, m) -> (b, m, d)
sub_2_2 = vmap(jnp.subtract, (None, 0), 1)  # (m, d) - (n, d) -> (m, n, d)

@jit
def get_equal(x, y):
    """
    (m, d) -> (m, m) comparing whether each pair of rows are equal
    Note: We still use == on floats. Floating error is not a problem as remove_repeated is used to get rid of bugs
        when comparing points in `get_newton_polytope_jax`.
    """
    return jnp.all(vmap(vmap(jnp.equal, (None, 0), 0), (0, None), 0)(x, y), axis=2)


@jit
def is_repeated(x):
    """
    (m, d) -> (m) whether there exists the same row whose index is less
    """
    n = x.shape[0]
    return jnp.any(get_equal(x, x) & ~jnp.triu(jnp.ones((n, n)).astype(bool), k=0), axis=1)


@jit
def remove_repeated_jax(points: jnp.ndarray, padding_value: float = -1.0) -> jnp.ndarray:
    batch_size, max_num_points, dimension = points.shape
    dtype = points.dtype

    mask = ~vmap(is_repeated, 0, 0)(points)
    return add_3_2(mul_3_2(points, mask.astype(dtype)), ((~mask) * padding_value).astype(dtype))


@jit
def get_interior(points_slice: jnp.ndarray) -> jnp.ndarray:
    """
    Intermediate function in `get_newton_polytope_approx_jax` to get the closure of interior of cubes.
    Parameter:
        points_slice: 2d array (max_num_points, dimension).
    Return:
        a 1-dim boolean jax.numpy array indicating whether the point (row) is in the closure of interior.
    """
    max_num_points = points_slice.shape[0]
    # Find all available points that are not removed (not marked by `padding_value`)
    available = jnp.all(points_slice >= 0, axis=1)
    available_mask = and_kronecker(available, available)
    diff = sub_2_2(points_slice, points_slice)  # (max_num_points, max_num_points, dimension)
    res = ~jnp.any(jnp.all(diff >= 0, axis=2) & (~jnp.diag(jnp.ones(max_num_points, dtype=bool))) & available_mask, axis=1)
    return res


@jit
def get_newton_polytope_approx_jax(points: jnp.ndarray, padding_value: float = -1.0) -> jnp.ndarray:
    """
    A JAX implementation of the approximation of Newton polytopes.
    Note: we cannot do inplace operations due to being JAX arrays.
    """
    points = remove_repeated_jax(points)
    dtype = points.dtype
    get_interior_batch = vmap(get_interior, 0, 0)
    mask = get_interior_batch(points)
    return add_3_2(mul_3_2(points, mask.astype(dtype)), ((~mask) * padding_value).astype(dtype))


def get_newton_polytope_jax(points: jnp.ndarray, padding_value: float = -1.0):
    return get_newton_polytope_approx_jax(points, padding_value=padding_value)


@jit
def shift_single_batch(points_slice: jnp.ndarray, coord_slice: jnp.ndarray, axis_slice: int) -> jnp.ndarray:
    max_num_points, dimension = points_slice.shape
    dtype = points_slice.dtype
    axis_binary = jnp.arange(dimension) == axis_slice
    # Get the raw computation of linear transformation
    return mul_kronecker(jnp.sum(mul_2_1(points_slice, coord_slice.astype(dtype)), axis=1),
                         axis_binary.astype(dtype)) + mul_2_1(points_slice, (~axis_binary).astype(dtype))


@jit
def shift_jax(points: jnp.ndarray, coord: jnp.ndarray, axis: jnp.ndarray, padding_value: float = -1.0):
    dtype = points.dtype
    shift_batch = vmap(shift_single_batch, (0, 0, 0), 0)(points, coord, axis)
    available = jnp.any(points >= 0, axis=2)
    return add_3_2(mul_3_2(shift_batch, available.astype(dtype)), ((~available) * padding_value).astype(dtype))


def calculate_rescale(points: jnp.ndarray) -> jnp.array:
    maximum = jnp.max(points)  # (m, d)
    # Use eps to stay away from division by zero (or very small).
    # But ideally, maximum should always be larger than 1 if the batch points keep rescaling.
    eps = 1e-8
    return lax.cond(maximum <= eps,
                    lambda: points,
                    lambda: points / maximum)


@jit
def rescale_jax(points: jnp.ndarray, padding_value: float = -1.0) -> jnp.array:
    """
    Calculate rescaling. Abort and return `points` if the maximal entry is less than or equal to zero (`eps`).
    Parameters:
        points: batch points. (b, m, d).
        padding_value: the negative value used to fill in deleted points.
    """
    available = jnp.any(points >= 0, axis=2)
    dtype = points.dtype
    raw_rescale = vmap(calculate_rescale, 0, 0)(points)
    return add_3_2(mul_3_2(raw_rescale, available.astype(dtype)), ((~available) * padding_value).astype(dtype))


@jit
def subtract_min(vector: jnp.ndarray, padding_value: float = -1.0) -> jnp.array:
    available = vector >= 0
    dtype = vector.dtype
    modified = vector * available + (~available * jnp.max(vector)).astype(dtype)
    minimal = jnp.min(modified)
    return lax.cond(minimal <= 0.0,
                    lambda: vector,
                    lambda: (vector - minimal) * available + (~available * padding_value).astype(dtype))


@jit
def reposition_jax(points: jnp.ndarray, padding_value: float = -1.0) -> jnp.array:
    return vmap(vmap(partial(subtract_min, padding_value=padding_value), 1, 1), 0, 0)(points)
