from functools import partial
from typing import Any, Callable, List, Tuple, Optional

from flax import linen as nn

from hironaka.jax.util import get_feature_fn
from jax import numpy as jnp
from jax import vmap

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
    spec: Optional[Tuple[int]] = None
    norm: ModuleDef = partial(nn.GroupNorm, num_groups=32)
    block_cls: ModuleDef = DenseResidueBlock
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = vmap(jnp.ravel, 0, 0)(x)  # Flatten
        for _, size in enumerate(self.net_arch):
            x = self.block_cls(features=size, dtype=self.dtype, norm=self.norm, activation=self.activation)(x)
        # Policy head and value head
        p = nn.Dense(self.output_size, dtype=self.dtype)(x)
        v = nn.Dense(1, dtype=self.dtype)(x)
        # Regulate the value output between -1 and 1
        v = nn.tanh(v)
        return p, v


DenseNet = partial(DenseResNet, block_cls=DenseBlock)


class CustomNet(nn.Module):
    output_size: int
    net_arch: List[int]
    spec: Tuple[int]
    norm: ModuleDef = partial(nn.GroupNorm, num_groups=32)
    block_cls: ModuleDef = DenseResNet
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.reshape((*x.shape[:-1], -1, self.spec[1]))
        available = x[..., :self.spec[0], 0] >= 0

        outs = []
        for i in range(self.spec[0]):
            out = x[..., :i, :].reshape((*x.shape[:-2], -1))
            for size in self.net_arch:
                out = self.block_cls(features=size, dtype=self.dtype, norm=self.norm, activation=self.activation)(out)
            outs.append(out)

        out = jnp.sum(jnp.stack(outs, axis=-1) * available[..., None, self.spec[0]], axis=-1)  # (b, size)
        extra = x[..., self.spec[0]:, :].reshape((*x.shape[:-2], -1))

        features = jnp.concatenate([out, extra], axis=-1)
        # Policy head and value head
        p = nn.Dense(self.output_size, dtype=self.dtype)(features)
        v = nn.Dense(1, dtype=self.dtype)(features)
        # Regulate the value output between -1 and 1
        v = nn.tanh(v)
        return p, v


DResNetMini = partial(DenseResNet, net_arch=[32] * 2)
DResNet18 = partial(DenseResNet, net_arch=[256] * 18)


def get_apply_fn(role: str, model: nn.Module, spec: Tuple[int, int], feature_fn=None) -> Callable:
    """
    The factory function for the `apply_fn` which evaluates on the point states and network parameters.
    Note that the first input of apply_fn is the batch of points: (batch_size, max_num_points, dimension).
    The only difference than just using model.apply is that the first input needs to be fed into a feature
        function which can be customized in `feature_fn` parameter. If not, we simply carry out the default
        process: flatten for host, flatten and concatenate with coordinates for agent.
    Parameters:
        role: host or agent.
        model: the flax.linen Module.
        spec: (max_num_points, dimension).
        feature_fn: (Optional) a custom feature function that transforms the default flattened arrays to custom
            feature arrays.
    Returns:
        the `apply_fn`.
    """
    if feature_fn is None:
        feature_fn = get_feature_fn(role, spec)

    def apply_fn(x: jnp.ndarray, params, *args, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        policy_prior, value_prior = model.apply(params, feature_fn(x))
        return policy_prior, value_prior.squeeze(-1)

    return apply_fn
