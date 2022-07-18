from typing import List, Any, Dict, Optional, Union

import numpy as np
import torch

from .PointsBase import PointsBase
from hironaka.src import get_batched_padded_array, rescale_torch
from hironaka.src import shift_torch, get_newton_polytope_torch, reposition_torch


class PointsTensor(PointsBase):
    subcls_config_keys = ['value_threshold', 'device_key', 'padding_value']
    copied_attributes = ['distinguished_points']

    def __init__(self,
                 points: Union[torch.Tensor, List[List[List[int]]], np.ndarray],
                 value_threshold: Optional[int] = 1e8,
                 device_key: Optional[str] = 'cpu',
                 padding_value: Optional[float] = -1.0,
                 distinguished_points: Optional[List[int]] = None,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        config = kwargs if config_kwargs is None else {**config_kwargs, **kwargs}
        self.value_threshold = value_threshold

        assert padding_value < 0, f"'padding_value' must be a negative number. Got {padding_value} instead."

        if isinstance(points, list):
            points = torch.FloatTensor(
                    get_batched_padded_array(points,
                                             new_length=config['max_num_points'],
                                             constant_value=padding_value))
        elif isinstance(points, np.ndarray):
            points = torch.FloatTensor(points)
        elif isinstance(points, torch.Tensor):
            points = points.type(torch.float32)
        else:
            raise Exception(f"Input must be a Tensor, a numpy array or a nested list. Got {type(points)}.")

        self.batch_size, self.max_num_points, self.dimension = points.shape

        self.device_key = device_key
        self.padding_value = padding_value
        self.distinguished_points = distinguished_points

        super().__init__(points, **config)
        self.device = torch.device(self.device_key)
        self.points.to(self.device)

    def exceed_threshold(self) -> bool:
        """
            Check whether the maximal value exceeds the threshold.
        """
        if self.value_threshold is not None:
            return torch.max(self.points) >= self.value_threshold
        return False

    def get_num_points(self) -> List[int]:
        """
            The number of points for each batch.
        """
        num_points = torch.sum(self.points[:, :, 0].ge(0), dim=1)
        return num_points.cpu().tolist()

    def _shift(self,
               points: torch.Tensor,
               coords: List[List[int]],
               axis: List[int],
               inplace: Optional[bool] = True):
        return shift_torch(points, coords, axis, inplace=inplace, padding_value=self.padding_value)

    def _get_newton_polytope(self, points: torch.Tensor, inplace: Optional[bool] = True):
        return get_newton_polytope_torch(points, inplace=inplace, padding_value=self.padding_value)

    def _get_shape(self, points: torch.Tensor):
        return points.shape

    def _reposition(self, points: torch.Tensor, inplace: Optional[bool] = True):
        return reposition_torch(points, inplace=inplace, padding_value=self.padding_value)

    def _rescale(self, points: torch.Tensor, inplace: Optional[bool] = True):
        r = rescale_torch(points, inplace=inplace, padding_value=self.padding_value)

    def _points_copy(self, points: torch.Tensor):
        return points.clone().detach()

    def _add_batch_axis(self, points: torch.Tensor):
        return points.unsqueeze(0)

    def _get_batch_ended(self, points: torch.Tensor):
        num_points = torch.sum(points[:, :, 0].ge(0), 1)
        return num_points.le(1).cpu().detach().tolist()

    def __repr__(self):
        return str(self.points)
