from typing import List, Tuple
import numpy as np
from .src import shift, getNewtonPolytope


class Points:
    """
        Class wrapper of points.

        Internally, it is a 3d numpy array whose size is unchanged.
        (batch, number_of_point, coordinate).
        For each batch, the number of points is either the pre-determined number or smaller.
        In case of a smaller number of points, the end of points is marked as "-1" on the next location
    """

    def __init__(self, pts: List[List[Tuple[int]]], ended=False):
        """
            Can be initialized by the naiive representation of points:
                list of tuples where each tuple represents a point.
        """
        self.points = np.array(pts)

        if len(self.points.shape) == 2:
            print("Input is required to be 3-dimensional: batch, number_of_points, coordinate")
            print("A batch dimension is automatically added.")
            self.points = np.reshape(self.points, (1, *self.points.shape))
        if len(self.points.shape) != 3:
            raise Exception("input dimension must be 2 or 3.")

        self.batchNum, self.m, self.dim = self.points.shape
        self.ended = ended
        self.numPoints = np.full((self.batchNum), self.m)

    def shift(self, coords: List[List[int]], axis: List[int], inplace=True):
        if inplace:
            self.ended = shift(self.points, coords, axis)
        else:
            r, self.ended = shift(self.points, coords, axis, inplace=False)
            return Points(r, ended=self.ended)

    def getNewtonPolytope(self, inplace=True):
        if inplace:
            self.ended = getNewtonPolytope(self.points)
            self._updateNumPoints()
        else:
            r, self.ended = getNewtonPolytope(self.points, inplace=False)
            self._updateNumPoints()
            return Points(r, ended=self.ended)

    def getBatch(self, b: int):
        return self.points[b][:self.numPoints[b]]

    def _updateNumPoints(self):
        for i in range(self.batchNum):
            for j in range(self.m):
                if self.points[i][j][0] == -1:
                    self.numPoints[i] = j
                    break

    def __repr__(self):
        return str(self.points)
