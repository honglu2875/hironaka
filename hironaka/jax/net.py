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
    """
    A wrapper class over the flax.linen.Module used for the policy/value predictions. It creates model parameters and
        apply functions. It also hosts metadata related to the particular policy/value model in the life-cycle of
        a JAXTrainer object, including:
        - model, the module.
        - batch_spec, the shape of the input (batch_size, max_num_points, dimension).
        - separate_policy_value_models, a bool value about whether to use separate models for policy and value.
        - output_shape, the output shape of the given model, under the input shape of `batch_spec`
    Note that it does not save the model parameters. Parameters are sent out and saved as TrainState externally. This
        class is only responsible for initializing and re-initializing them (perhaps with a different batch size).
    """
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

        self.init(key, batch_spec)

    def init(self, key: jnp.ndarray, batch_spec: Tuple, input_shape=None):
        """
        (Re-)initialize the model, update the output shape of the model (`self.output_shape`), and return the newly
            initialized parameters.
        Parameters:
            key: the PRNG key.
            batch_spec: shape specifications including batch size, (batch_size, max_num_points, dimension).
        Returns:
            tuple, in the form of
                (policy_net_parameters, value_net_parameters).
            when self.separate_policy_value_models==False, value_net_parameters is None
        """
        key, value_key = jax.random.split(key)

        self.init_key = key
        if input_shape is None:
            # When the feature function is not customized: we simply flatten the point states,
            # resulting in the following numbers:
            self.input_shape = (
                (batch_spec[0], batch_spec[1] * batch_spec[2])
                if self.role == "host"
                else (batch_spec[0], batch_spec[1] * batch_spec[2] + batch_spec[2])
            )
        else:
            self.input_shape = input_shape

        parameters = self.model.init(key, jnp.ones(self.input_shape))
        self.output_shape = self.model.apply(parameters, jnp.ones(self.input_shape)).shape

        if self.separate_policy_value_models:
            value_parameters = self.value_model.init(value_key, jnp.ones(self.input_shape))
        else:
            value_parameters = None
        return parameters, value_parameters

    def get_apply_fn(self, new_batch_size=None, feature_fn=None) -> Callable:
        """
        The factory function for the `apply_fn` which evaluates on the point states and network parameters.
        Note that the first input of apply_fn is the batch of points: (batch_size, max_num_points, dimension). It needs
            to be later fed into a feature function which can be customized in `feature_fn` parameter. If not, we simply
            carry out the default process: flatten for host, flatten and concatenate with coordinates for agent.
        Parameters:
            new_batch_size: (Optional) the potential new batch size used to override the original setup. The override
                is one-time only and does not rewrite `self.input_shape`.
            feature_fn: (Optional) the custom feature function.
        Returns:
            the `apply_fn`.
        """
        batch_size = new_batch_size if new_batch_size is not None else self.input_shape[0]
        _, logit_length = self.output_shape
        if feature_fn is None:
            feature_fn = get_feature_fn(self.role, self.batch_spec[1:])

        if self.separate_policy_value_models:
            raise NotImplementedError()  # Not implemented yet. Will do when the need arises.
        else:
            def apply_fn(x: jnp.ndarray, params, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
                output = self.model.apply(params, feature_fn(x))
                policy_logits = lax.dynamic_slice(output, (0, 0), (batch_size, logit_length - 1))
                value_logits = jnp.ravel(lax.dynamic_slice(output, (0, logit_length - 1), (batch_size, 1)))
                return policy_logits, value_logits

        return apply_fn
