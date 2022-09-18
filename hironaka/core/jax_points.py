from functools import partial
from typing import List, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import vmap

from hironaka.src import get_batched_padded_array, get_newton_polytope_jax, reposition_jax, rescale_jax, shift_jax
from .points_base import PointsBase


class JAXPoints(PointsBase):
    subcls_config_keys = ["value_threshold", "device", "padding_value", "dtype"]
    running_attributes = ["distinguished_points"]

    def __init__(
            self,
            points: Union[jnp.ndarray, List[List[List[float]]], np.ndarray],
            value_threshold: Optional[float] = 1e8,
            device: Optional[jaxlib.xla_extension.Device] = None,
            padding_value: Optional[float] = -1.0,
            distinguished_points: Optional[List[int]] = None,
            dtype: Optional[Type] = jnp.float32,
            **kwargs,
    ):
        self.value_threshold = value_threshold
        assert padding_value <= 0.0, f"'padding_value' must be a non-positive number. Got {padding_value} instead."

        self.dtype = dtype
        if device is None:
            cpus = jax.devices(backend="cpu")
            if cpus:
                self.device = cpus[0]
            else:
                raise SystemError("Cannot find a CPU device.")
        else:
            self.device = device

        if isinstance(points, list):
            points = jnp.array(
                get_batched_padded_array(points, new_length=kwargs["max_num_points"], constant_value=padding_value),
                dtype=self.dtype,
            )
        elif isinstance(points, np.ndarray):
            points = jnp.array(points, dtype=self.dtype)
        elif isinstance(points, jnp.ndarray):
            points = points.astype(self.dtype)
        else:
            raise Exception(f"Input must be a jax numpy array, a numpy array or a nested list. Got {type(points)}.")
        jax.device_put(points, self.device)

        self.batch_size, self.max_num_points, self.dimension = points.shape
        self.padding_value = padding_value
        self.distinguished_points = distinguished_points

        super().__init__(points, **kwargs)

    def exceed_threshold(self) -> bool:
        """
        Check whether the maximal value exceeds the threshold.
        """
        return jnp.max(self.points) >= self.value_threshold

    def get_features(self) -> jnp.ndarray:
        return vmap(partial(jnp.sort, axis=0), 0, 0)(self.points)

    def type(self, t: Type):
        self.dtype = t
        self.points = self.points.astype(t)

    def _shift(
            self, points: jnp.ndarray, coords: jnp.ndarray, axis: jnp.ndarray, inplace: Optional[bool] = True, **kwargs
    ) -> jnp.ndarray:
        new_pts = shift_jax(points, coords, axis, padding_value=self.padding_value)
        if inplace:
            self.points = new_pts
        return new_pts

    def _get_newton_polytope(self, points: jnp.ndarray, inplace: Optional[bool] = True, **kwargs) -> jnp.ndarray:
        new_pts = get_newton_polytope_jax(points, padding_value=self.padding_value)
        if inplace:
            self.points = new_pts
        return new_pts

    def _get_shape(self, points: jnp.ndarray) -> Tuple:
        return points.shape

    def _reposition(self, points: jnp.ndarray, inplace: Optional[bool] = True, **kwargs) -> jnp.ndarray:
        new_pts = reposition_jax(points, padding_value=self.padding_value)
        if inplace:
            self.points = new_pts
        return new_pts

    def _rescale(self, points: jnp.ndarray, inplace: Optional[bool] = True, **kwargs) -> jnp.ndarray:
        new_pts = rescale_jax(points, padding_value=self.padding_value)
        if inplace:
            self.points = new_pts
        return new_pts

    @staticmethod
    def _points_copy(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.copy(points)

    def _add_batch_axis(self, points: jnp.ndarray) -> jnp.ndarray:
        return jnp.expand_dims(points, axis=0)

    def _get_batch_ended(self, points: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((points[:, :, 0] >= 0), axis=1) < 2

    def __repr__(self) -> str:
        return str(self.points)

    def __hash__(self) -> int:
        return hash(self.points.round(8).tostring())
