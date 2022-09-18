from functools import partial
from typing import Any, Callable, List, Tuple

import flax
import jax
from flax import linen as nn
from jax import lax
from jax import numpy as jnp
from jax import vmap

from hironaka.jax.util import get_feature_fn

ModuleDef = Any


class DenseResidueBlock(nn.Module):
    features: int
    dtype: jnp.dtype
    norm: ModuleDef
    activation: Callable

    @nn.compact
    def __call__(self, x):
        original = x
        y = nn.Dense(self.features, dtype=self.dtype)(x)
        y = self.norm()(y)
        y = self.activation(y)
        y = nn.Dense(self.features, dtype=self.dtype)(y)
        y = self.norm()(y)

        if original.shape != y.shape:
            original = nn.Dense(self.features, dtype=self.dtype, name="res_proj")(original)
            original = self.norm(name="norm_proj")(original)

        return self.activation(original + y)


class DenseBlock(nn.Module):
    features: int
    dtype: jnp.dtype
    norm: ModuleDef
    activation: Callable

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.features, dtype=self.dtype)(x)
        return self.activation(y)


class DenseResNet(nn.Module):
    output_size: int
    net_arch: List[int]
    norm: ModuleDef = nn.LayerNorm
    block_cls: ModuleDef = DenseResidueBlock
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = vmap(jnp.ravel, 0, 0)(x)  # Flatten
        for _, size in enumerate(self.net_arch):
            x = self.block_cls(features=size, dtype=self.dtype, norm=self.norm, activation=self.activation)(x)
        x = nn.Dense(self.output_size, dtype=self.dtype)(x)
        return x


DenseNet = partial(DenseResNet, block_cls=DenseBlock)

CustomNet = DenseResNet

DResNetMini = partial(DenseResNet, net_arch=[32] * 2)
DResNet18 = partial(DenseResNet, net_arch=[256] * 18)


class PolicyWrapper:
    def __init__(
            self,
            key: jnp.ndarray,
            role: str,
            batch_spec: Tuple,
            model: flax.linen.Module,
            value_model: flax.linen.Module = None,
            separate_policy_value_models: bool = False,
    ):
        """
        Parameters:
            key: the PRNG random key.
            role: 'host' or 'agent'.
            batch_spec: specify the point properties including the batch size: (batch_size, max_num_points, dimension).
            model: the policy model. If `separate_policy_value_models` is False, the last logit of the model output
                is assumed to be the value.
            value_model: if `separate_policy_value_models` is True, it is the separate value model.
            separate_policy_value_models: use separate policy model and value model.
        """
        self.role = role
        self.batch_spec = batch_spec
        self.model = model
        self.separate_policy_value_models = separate_policy_value_models
        if self.separate_policy_value_models:
            if value_model is None:
                raise ValueError("If policy and value models are separated, 'value_model' must be set.")
        self.value_model = value_model

        self.init_key, self.input_shape, self.output_shape = None, None, None
        self.parameters, self.value_parameters = None, None

        self.init(key, batch_spec)

    def init(self, key: jnp.ndarray, batch_spec: Tuple):
        """
        (Re-)initialize the model.
        """
        if self.separate_policy_value_models:
            key, value_key = jax.random.split(key)
        else:
            key, value_key = key, None

        self.init_key = key
        self.input_shape = (
            (batch_spec[0], batch_spec[1] * batch_spec[2])
            if self.role == "host"
            else (batch_spec[0], batch_spec[1] * batch_spec[2] + batch_spec[2])
        )
        self.parameters = self.model.init(key, jnp.ones(self.input_shape))
        self.output_shape = self.model.apply(self.parameters, jnp.ones(self.input_shape)).shape
        if self.separate_policy_value_models:
            self.value_parameters = self.value_model.init(value_key, jnp.ones(self.input_shape))

    def get_apply_fn(self, new_batch_size=None, feature_fn=None) -> Callable:
        batch_size = new_batch_size if new_batch_size is not None else self.input_shape[0]
        _, logit_length = self.output_shape
        if feature_fn is None:
            feature_fn = get_feature_fn(self.role, self.batch_spec[1:])

        if self.separate_policy_value_models:
            raise NotImplementedError()
        else:
            def apply_fn(x: jnp.ndarray, params, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
                output = self.model.apply(params, feature_fn(x))
                policy_logits = lax.dynamic_slice(output, (0, 0), (batch_size, logit_length - 1))
                value_logits = jnp.ravel(lax.dynamic_slice(output, (0, logit_length - 1), (batch_size, 1)))
                return policy_logits, value_logits

        return apply_fn
