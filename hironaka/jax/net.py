from typing import List, Callable, Any

import jax.numpy as jnp
from jax import vmap
from flax import linen as nn

ModuleDef = Any


class DenseResidueBlock(nn.Module):
    features: int
    dtype: jnp.dtype
    norm: ModuleDef
    activation: Callable

    @nn.compact
    def __call__(self, x, ):
        residual = x
        y = nn.Dense(self.features, dtype=self.dtype)(x)
        y = self.norm()(y)
        y = self.activation(y)
        y = nn.Dense(self.features, dtype=self.dtype)(x)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = nn.Dense(self.features, dtype=self.dtype, name='res_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.activation(residual + y)


class DenseResNet(nn.Module):
    net_arch: List[int]
    output_size: int
    norm: ModuleDef = nn.LayerNorm
    block_cls: ModuleDef = DenseResidueBlock
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = vmap(jnp.ravel, 0, 0)(x)  # Flatten
        for i, size in enumerate(self.net_arch):
            x = self.block_cls(features=size,
                               dtype=self.dtype,
                               norm=self.norm,
                               activation=self.activation)(x)
        x = nn.Dense(self.output_size, dtype=self.dtype)(x)
        return self.activation(x)
