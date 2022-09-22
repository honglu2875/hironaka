from typing import List, Optional, Type, Union

import numpy as np
import torch

from hironaka.src import get_batched_padded_array, get_newton_polytope_torch, reposition_torch, rescale_torch, shift_torch

from .points_base import PointsBase


class TensorPoints(PointsBase):
    subcls_config_keys = ["value_threshold", "device", "padding_value", "dtype"]
    running_attributes = ["distinguished_points"]

    def __init__(
        self,
        points: Union[torch.Tensor, List[List[List[float]]], np.ndarray],
        value_threshold: Optional[float] = 1e8,
        device: Optional[Union[str, torch.device]] = "cpu",
        padding_value: Optional[float] = -1.0,
        distinguished_points: Optional[List[int]] = None,
        dtype: Optional[Union[Type, torch.dtype]] = torch.float32,
        **kwargs,
    ):
        self.value_threshold = value_threshold

        assert padding_value <= 0.0, f"'padding_value' must be a non-positive number. Got {padding_value} instead."

        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        if isinstance(points, list):
            points = torch.tensor(
                get_batched_padded_array(points, new_length=kwargs["max_num_points"], constant_value=padding_value),
                device=self.device,
                dtype=self.dtype,
            )
        elif isinstance(points, np.ndarray):
            points = torch.tensor(points, device=self.device, dtype=self.dtype)
        elif isinstance(points, torch.Tensor):
            points = points.type(self.dtype).to(self.device)
        else:
            raise Exception(f"Input must be a Tensor, a numpy array or a nested list. Got {type(points)}.")

        self.batch_size, self.max_num_points, self.dimension = points.shape
        self.padding_value = padding_value
        self.distinguished_points = distinguished_points

        super().__init__(points, **kwargs)

        # Legacy support
        if "device_key" in kwargs:
            self.device = torch.device(kwargs["device_key"])
            self.points.to(self.device)
            self.logger.warning("'device_key' is a legacy parameter. Will be deprecated in future version.")

    def exceed_threshold(self) -> bool:
        """
        Check whether the maximal value exceeds the threshold.
        """
        if self.value_threshold is not None:
            return torch.max(self.points) >= self.value_threshold
        return False

    def get_num_points(self) -> torch.Tensor:
        """
        The number of points for each batch.
        """
        num_points = torch.sum(self.points[:, :, 0].ge(0), dim=1)
        return num_points

    def get_features(self) -> torch.Tensor:
        sorted_args = torch.argsort(self.points[:, :, 0], dim=1, descending=True)
        return self.points.gather(1, sorted_args.unsqueeze(-1).repeat(1, 1, self.dimension)).clone()

    def type(self, t: Union[Type, torch.dtype]):
        self.dtype = t
        self.points = self.points.type(t)

    def _shift(
        self,
        points: torch.Tensor,
        coords: Union[torch.Tensor, List[List[int]]],
        axis: Union[torch.Tensor, List[int]],
        inplace: Optional[bool] = True,
        ignore_ended_games: Optional[bool] = True,
        **kwargs,
    ) -> Union[torch.Tensor, None]:
        return shift_torch(
            points, coords, axis, inplace=inplace, padding_value=self.padding_value, ignore_ended_games=ignore_ended_games
        )

    def _get_newton_polytope(
        self, points: torch.Tensor, inplace: Optional[bool] = True, **kwargs
    ) -> Union[torch.Tensor, None]:
        return get_newton_polytope_torch(points, inplace=inplace, padding_value=self.padding_value)

    def _get_shape(self, points: torch.Tensor) -> torch.Size:
        return points.shape

    def _reposition(self, points: torch.Tensor, inplace: Optional[bool] = True, **kwargs) -> Union[torch.Tensor, None]:
        return reposition_torch(points, inplace=inplace, padding_value=self.padding_value)

    def _rescale(self, points: torch.Tensor, inplace: Optional[bool] = True, **kwargs) -> Union[torch.Tensor, None]:
        return rescale_torch(points, inplace=inplace, padding_value=self.padding_value)

    @staticmethod
    def _points_copy(points: torch.Tensor) -> torch.Tensor:
        return points.clone().detach()

    def _add_batch_axis(self, points: torch.Tensor) -> torch.Tensor:
        return points.unsqueeze(0)

    def _get_batch_ended(self, points: torch.Tensor) -> torch.Tensor:
        num_points = torch.sum(points[:, :, 0].ge(0), 1)
        return num_points.le(1).detach()

    @property
    def ended_batch_in_tensor(self) -> torch.Tensor:
        return torch.sum(self.points[:, :, 0].ge(0), 1).le(1)

    def __repr__(self) -> str:
        return str(self.points)

    def __hash__(self) -> int:
        return hash(self.points.detach().cpu().numpy().round(8).tostring())
