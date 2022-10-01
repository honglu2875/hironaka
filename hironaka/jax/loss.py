import jax
from jax import numpy as jnp


def compute_loss(params, apply_fn, sample, loss_fn, weight=None) -> jnp.ndarray:
    obs, target_policy_logits, target_value = sample
    weight = jnp.where(weight is None, jnp.ones_like(target_value), weight)
    policy_logits, value = apply_fn(obs, params)
    return loss_fn(policy_logits, value, target_policy_logits, target_value, weight=weight, params=params)


def policy_value_loss(
        policy_logit: jnp.ndarray, value: jnp.ndarray, target_policy: jnp.ndarray, target_value: jnp.ndarray,
        weight=None) -> jnp.ndarray:
    # Shapes:
    # policy_logit, target_policy: (B, action)
    # value_logit, target_value: (B,)
    weight = jnp.where(weight is None, jnp.ones_like(value), weight)
    policy_loss = jnp.sum(
        -jax.nn.softmax(target_policy, axis=-1) * jax.nn.log_softmax(policy_logit, axis=-1),
        axis=1
    )
    value_loss = jnp.square(value - target_value)
    return jnp.mean((policy_loss + value_loss) * weight)


def clip_log(x: jnp.ndarray, a_min=1e-8) -> jnp.ndarray:
    return jnp.log(jnp.clip(x, a_min=a_min))
