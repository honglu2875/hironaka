import jax
from jax import numpy as jnp


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


def clip_log(x: jnp.ndarray, a_min=1e-8) -> jnp.ndarray:
    return jnp.log(jnp.clip(x, a_min=a_min))
