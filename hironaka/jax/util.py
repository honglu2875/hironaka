import jax.numpy as jnp

from jax import vmap, jit

flatten = vmap(jnp.ravel, 0, 0)


@jit
def make_agent_obs(pts: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Combine points observation and coordinates into a flattened and concatenated agent observation.
    Parameters:
        pts: jax array of shape (batch_size, max_num_points, dimension)
        coords: jax multi-binary array of shape (batch_size, dimension)
    """
    return jnp.concatenate([flatten(pts), coords], axis=1)
