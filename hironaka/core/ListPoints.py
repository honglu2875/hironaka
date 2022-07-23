from typing import List, Any, Dict, Optional, Union

import numpy as np

from .PointsBase import PointsBase
from hironaka.src import shift_lst, get_newton_polytope_lst, get_shape, scale_points, reposition_lst, \
    get_newton_polytope_approx_lst


class ListPoints(PointsBase):
    """
        When dealing with small batches, small dimension and small point numbers, list is much better than numpy.
    """
    subcls_config_keys = ['value_threshold', 'use_precise_newton_polytope']
    copied_attributes = ['distinguished_points']

    def __init__(self,
                 points: Union[List[List[List[float]]], List[List[float]], np.ndarray],
                 value_threshold: Optional[float] = 1e8,
                 use_precise_newton_polytope: Optional[bool] = False,
                 distinguished_points: Optional[Union[List[int], None]] = None,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        config = kwargs if config_kwargs is None else {**config_kwargs, **kwargs}
        self.value_threshold = value_threshold
        self.use_precise_newton_polytope = use_precise_newton_polytope
        self.distinguished_points = distinguished_points

        # Be lenient and allow numpy array as input.
        # The input might already be a padded arrays. Thus, we do a thorough check to clean that up.
        # WARNING: padding value must be *STRICTLY* negative for the numpy array!
        if isinstance(points, np.ndarray):
            points = points.tolist()
            assert isinstance(points, list)
            for p in points:
                while p[-1][0] < 0:
                    p.pop()
                    assert len(p) != 0  # Should not be empty batch

        super().__init__(points, **config)

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
               points: Any,
               coords: List[List[int]],
               axis: List[int],
               inplace: Optional[bool] = True,
               **kwargs):
        return shift_lst(points, coords, axis, inplace=inplace)

    def _reposition(self, points: Any, inplace: Optional[bool] = True, **kwargs):
        return reposition_lst(points, inplace=inplace)

    def _get_newton_polytope(self, points: Any, inplace: Optional[bool] = True, **kwargs):
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

    def _get_shape(self, points: Any):
        return get_shape(points)

    def _rescale(self, points: Any, inplace: Optional[bool] = True, **kwargs):
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

    def get_sym_features(self):
        """
            [currently not in-use]
            Say the points are ((x_1)_1, ...,(x_1)_n), ...,((x_k)_1, ...,(x_k)_n)
            We generate the Newton polynomials of each coordinate and output the new array as features.
            The output becomes
                ((sum_i (x_i)_1^1), ..., (sum_i (x_i)_n^1)),
                ...,
                ((sum_i (x_i)_1^length), ..., (sum_i (x_i)_n^length))
        """
        features = [
            [
                [
                    sum([
                        x[i] ** j for x in batch
                    ]) for i in range(self.dimension)
                ] for j in range(1, self.max_num_points + 1)
            ] for batch in self.points
        ]

        return features

    def __repr__(self):
        return str(self.points)
