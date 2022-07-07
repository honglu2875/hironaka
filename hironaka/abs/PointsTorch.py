from typing import List, Any, Dict, Optional, Union

import numpy as np
import torch

from hironaka.abs.PointsBase import PointsBase
from hironaka.src import shift_lst, get_newton_polytope_lst, get_shape, scale_points, get_batched_padded_array


class PointsTorch(PointsBase):
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
        if 'max_num_points' not in config:
            raise Exception("Must have `max_num_points` in parameters.")

        if isinstance(points, list):
            points = torch.FloatTensor(
                    get_batched_padded_array(points,
                                             new_length=config['max_num_points'],
                                             constant_value=config.get('padded_value', -1)))
        elif isinstance(points, (torch.Tensor, np.ndarray)):
            points = torch.FloatTensor(points)
        else:
            raise Exception(f"Input must be a Tensor, a numpy array or a nested list. Got {type(points)}.")

        self.device_key = device_key
        self.padded_value = padded_value

        super().__init__(points, **config)
        self.device = torch.device(self.device_key)
        self.points.to(self.device)

    # TODO:
    def exceed_threshold(self) -> bool:
        """
            Check whether the maximal value exceeds the threshold.
        """
        if self.value_threshold is not None:
            if torch.max(self.points) >= self.value_threshold:
                return True
        return False

    def get_num_points(self) -> List[int]:
        """
            The number of points for each batch.
        """
        first_slice = self.points[:, :, 0].ge(0)
        # TODO:

        return [len(batch) for batch in self.points]

    def _shift(self,
               points: Any,
               coords: List[List[int]],
               axis: List[int],
               inplace: Optional[bool] = True):
        return shift_lst(points, coords, axis, inplace=inplace)

    def _get_newton_polytope(self, points: Any, inplace: Optional[bool] = True):
        return get_newton_polytope_lst(points, inplace=inplace, get_ended=False)

    def _get_shape(self, points: Any):
        return get_shape(points)

    def _rescale(self, points: Any, inplace: Optional[bool] = True):
        return scale_points(points, inplace=inplace)

    def _points_copy(self, points: Any):
        return [[point.copy() for point in batch] for batch in points]

    def _add_batch_axis(self, points: Any):
        return [points]

    def _get_batch_ended(self, points: Any):
        ended_each_batch = []
        for p in points:
            assert len(p) != 0
            ended_each_batch.append(True if len(p) <= 1 else False)
        return ended_each_batch

    def __repr__(self):
        return str(self.points)
