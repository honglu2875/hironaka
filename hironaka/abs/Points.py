from typing import List

import numpy as np

from hironaka.src import shift_lst, shift_np, get_newton_polytope_lst, get_newton_polytope_np, get_shape


class Points:
    """
        Class wrapper of points.

        Internally, it is List[List[List[int]]] containing batches of lists of points. When use_np is True, it uses
        3d numpy array instead(size will be unchanged and removed points will be marked as [-1,...,-1] for easier c++
        compatibility). (batch, number_of_point, coordinate). For each batch, the number of points is either the
        pre-determined number or smaller.
    """
    points: List[List[List[int]]]
    ended: bool  # updated after every call of get_newton_polytope()

    def __init__(self, pts, ended=False, use_np=False, value_threshold=None):
        """
        In any case, the three dimensions will be (self.batchNum, self.m, self.dim). When use_np=True, everything
        will be saved and manipulated as numpy array (a lot of room for improvement!!!). Otherwise, it will be nested
        list-like mutable objects.
        """
        self.use_np = use_np
        if self.use_np:
            pts = np.array(pts)
            shape = pts.shape
            self._shift = shift_np
            self._getNewtonPolytope = get_newton_polytope_np
        else:
            if isinstance(pts, np.ndarray):
                pts = pts.tolist()
                # If the input is already numpy array, the input might already be -1 padded arrays.
                # Thus, we do a thorough check to clean that up.
                for b in range(len(pts)):
                    while pts[b][-1][0] == -1:
                        pts[b].pop()
                        assert len(pts[b]) != 0

            shape = get_shape(pts)
            self._shift = shift_lst
            self._getNewtonPolytope = get_newton_polytope_lst

        if len(shape) == 2:
            print("Input is required to be 3-dimensional: batch, number_of_points, coordinate")
            print("A batch dimension is automatically added.")
            shape = (1, *shape)
            if self.use_np:
                pts = np.reshape(pts, shape)
            else:
                pts = [pts]
        if len(shape) != 3:
            raise Exception("input dimension must be 2 or 3.")

        self.points = pts
        self.batchNum, self.m, self.dim = shape
        self.ended = ended
        self.value_threshold = value_threshold

        if self.use_np:
            self.numPoints = np.full(self.batchNum, self.m)
        else:
            self.numPoints = [self.m] * self.batchNum

    def shift(self, coords: List[List[int]], axis: List[int], inplace=True):
        if inplace:
            self._shift(self.points, coords, axis)
        else:
            r = self._shift(self.points, coords, axis, inplace=False)
            return Points(r.copy(), ended=self.ended, use_np=self.use_np)

    def get_newton_polytope(self, inplace=True):
        if inplace:
            self.ended = self._getNewtonPolytope(self.points)
            self._update_num_points()
        else:
            r, self.ended = self._getNewtonPolytope(self.points, inplace=False)
            return Points(r.copy(), ended=self.ended, use_np=self.use_np)

    def copy(self):
        p = [[point.copy() for point in batch] for batch in self.points]
        return Points(p, ended=self.ended, use_np=self.use_np, value_threshold=self.value_threshold)

    def get_batch(self, b: int):
        return self.points[b][:self.numPoints[b]]

    def exceed_threshold(self):
        if self.value_threshold is not None:
            if self.use_np:
                return np.max(self.points) > self.value_threshold
            else:
                for b in range(self.batchNum):
                    for i in range(len(self.points[b])):
                        if max(self.points[b][i]) > self.value_threshold:
                            return True
                return False
        return False

    def get_features(self):
        """
            Get the list of points.
        """
        return self.points

    def get_sym_features(self):
        """
            Say the points are ((x_1)_1, ...,(x_1)_n), ...,((x_k)_1, ...,(x_k)_n)
            We generate the Newton polynomials of each coordinate and output the new array as feature.
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
                    ]) for i in range(self.dim)
                ] for j in range(1, self.m + 1)
            ] for batch in self.points
        ]

        if self.use_np:
            return np.array(features)  # TODO: could directly optimize using vectorization
        else:
            return features

    def _update_num_points(self):
        """
            When use_np=True, removed points will become [-1,...,-1] and the numPoints array will need to be updated.
        """
        if self.use_np:
            for i in range(self.batchNum):
                for j in range(self.m):
                    if self.points[i][j][0] == -1:
                        self.numPoints[i] = j
                        break

    def __repr__(self):
        return str(self.points)
