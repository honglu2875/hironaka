from typing import List, Any, Dict, Optional, Union

import numpy as np
import torch

from hironaka.abs.PointsBase import PointsBase
from hironaka.src import shift_lst, get_newton_polytope_lst, get_shape, scale_points, get_batched_padded_array
from hironaka.src._torch_ops import shift_torch, get_newton_polytope_torch, reposition_torch


class PointsTensor(PointsBase):
    config_keys = ['value_threshold', 'device_key', 'padded_value']

    def __init__(self,
                 points: Union[torch.Tensor, List[List[List[int]]], np.ndarray],
                 value_threshold: Optional[int] = 1e8,
                 device_key: Optional[str] = 'cpu',
                 padded_value: Optional[float] = -1.0,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        config = kwargs if config_kwargs is None else {**config_kwargs, **kwargs}
        self.value_threshold = value_threshold

        # It's better to require a fixed shape of the tensor implementation.

        if isinstance(points, list):
            points = torch.FloatTensor(
                    get_batched_padded_array(points,
                                             new_length=config['max_num_points'],
                                             constant_value=config.get('padded_value', -1)))
        elif isinstance(points, (torch.Tensor, np.ndarray)):
            points = torch.FloatTensor(points)
        else:
            raise Exception(f"Input must be a Tensor, a numpy array or a nested list. Got {type(points)}.")

        self.batch_size, self.max_num_points, self.dimension = points.shape

        self.device_key = device_key
        self.padded_value = padded_value

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
        return shift_torch(points, coords, axis, inplace=inplace)

    def _get_newton_polytope(self, points: torch.Tensor, inplace: Optional[bool] = True):
        return get_newton_polytope_torch(points, inplace=inplace)

    def _get_shape(self, points: torch.Tensor):
        return points.shape

    def _reposition(self, points: torch.Tensor, inplace: Optional[bool] = True):
        return reposition_torch(points, inplace=inplace)

    def _rescale(self, points: torch.Tensor, inplace: Optional[bool] = True):
        return points / torch.reshape(torch.amax(points, (1, 2)), (-1, 1, 1))

    def _points_copy(self, points: torch.Tensor):
        return points.clone().detach()

    def _add_batch_axis(self, points: torch.Tensor):
        return points.unsqueeze(0)

    def _get_batch_ended(self, points: torch.Tensor):
        num_points = torch.sum(points[:, :, 0].ge(0), 1)
        return num_points.le(1).cpu().detach().tolist()

    def __repr__(self):
        return str(self.points)
