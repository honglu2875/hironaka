from typing import List, Any, Optional, Union, Tuple

import numpy as np
import torch

from hironaka.src import shift_lst, get_newton_polytope_lst, get_shape, scale_points, reposition_lst, \
    get_newton_polytope_approx_lst
from .PointsBase import PointsBase

PointsAsNestedLists = List[List[List[Any]]]


class ListPoints(PointsBase):
    """
    An abstraction of (batches of) collection of points. Point data is saved as python lists.
    Rmk: When dealing with small batches, small dimension and small point numbers, list is much better than numpy.
    """
    subcls_config_keys = ['value_threshold', 'use_precise_newton_polytope']
    running_attributes = ['distinguished_points']

    def __init__(self,
                 points: Union[PointsAsNestedLists, List[List[float]], np.ndarray],
                 value_threshold: Optional[float] = 1e8,
                 use_precise_newton_polytope: Optional[bool] = False,
                 distinguished_points: Optional[Union[List[int], None]] = None,
                 **kwargs):
        """
        Parameters:
            points: representation of points.
            value_threshold: the threshold above which the game will stop.
            use_precise_newton_polytope: use a proper convex cone algorithm instead of the approximate version.
            distinguished_points: a distinguished point which we keep track of (call )
        """
        self.value_threshold = value_threshold
        self.use_precise_newton_polytope = use_precise_newton_polytope
        self.distinguished_points = distinguished_points

        # Be lenient and (try to) allow numpy array/tensor as input.
        # The input might already be a padded arrays. Thus, we do a thorough check to clean that up.
        # WARNING: padding value must be *STRICTLY* negative for the numpy array!
        if isinstance(points, np.ndarray) or isinstance(points, torch.Tensor):
            points = points.tolist()
            for p in points:
                while p[-1][0] < 0:
                    p.pop()
                    assert len(p) != 0  # Should not be empty batch

        super().__init__(points, **kwargs)

    def exceed_threshold(self) -> bool:
        """
        Check whether the maximal value exceeds the threshold.
        """
        if self.value_threshold is not None:
            for b in range(self.batch_size):
                for i in range(len(self.points[b])):
                    if max(self.points[b][i]) > self.value_threshold:
                        return True
            return False
        return False

    def get_num_points(self) -> List[int]:
        """
        The number of points for each batch.
        """
        return [len(batch) for batch in self.points]

    def _shift(self,
               points: PointsAsNestedLists,
               coords: List[List[int]],
               axis: List[int],
               inplace: Optional[bool] = True,
               **kwargs) -> Union[PointsAsNestedLists, None]:
        return shift_lst(points, coords, axis, inplace=inplace)

    def _reposition(self, points: PointsAsNestedLists,
                    inplace: Optional[bool] = True, **kwargs) -> Union[PointsAsNestedLists, None]:
        return reposition_lst(points, inplace=inplace)

    def _get_newton_polytope(self, points: PointsAsNestedLists,
                             inplace: Optional[bool] = True, **kwargs) -> Union[PointsAsNestedLists, None]:
        # Mark distinguished points
        if self.distinguished_points is not None:
            # Apply marks to the distinguished points before the operation
            for b in range(self.batch_size):
                if self.distinguished_points[b] is None:
                    continue
                self.points[b][self.distinguished_points[b]].append('d')

        if self.use_precise_newton_polytope:
            result = get_newton_polytope_lst(points, inplace=inplace)
        else:
            result = get_newton_polytope_approx_lst(points, inplace=inplace, get_ended=False)

        # Recover the locations of distinguished points
        if self.distinguished_points is not None:
            transformed_points = points if inplace else result
            for b in range(self.batch_size):
                if self.distinguished_points[b] is None:
                    continue
                distinguished_point_index = None
                for i in range(len(transformed_points[b])):
                    if transformed_points[b][i][-1] == 'd':
                        distinguished_point_index = i
                        transformed_points[b][i].pop()
                        break
                self.distinguished_points[b] = distinguished_point_index

        return result

    def _get_shape(self, points: PointsAsNestedLists) -> Tuple[int, int, int]:
        return get_shape(points)

    def _rescale(self, points: PointsAsNestedLists,
                 inplace: Optional[bool] = True, **kwargs) -> Union[PointsAsNestedLists, None]:
        return scale_points(points, inplace=inplace)

    def _add_batch_axis(self, points: PointsAsNestedLists) -> PointsAsNestedLists:
        return [points]

    def _get_batch_ended(self, points: PointsAsNestedLists) -> List[bool]:
        ended_each_batch = []
        for p in points:
            assert len(p) != 0
            ended_each_batch.append(True if len(p) <= 1 else False)
        return ended_each_batch

    def __repr__(self) -> str:
        return str(self.points)
