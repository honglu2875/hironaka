import ctypes
from typing import List

import numpy as np

cppUtil = ctypes.cdll.LoadLibrary("build/cppUtil.so")

cppUtil.getNewtonPolytope_approx.argtypes = [
    ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
cppUtil.getNewtonPolytope_approx.restype = None


def get_newton_polytope_approx_py_np(points: np.ndarray, inplace=True):
    """
        A simple-minded quick-and-dirty method to obtain an approximation of Newton Polytope disregarding convexity.
    """

    assert len(points.shape) == 3

    batch_num, m, n = points.shape
    new_points = np.full(points.shape, -1)
    ended = True

    for b in range(batch_num):
        counter = 0
        for i in range(m):
            if points[b][i][0] == -1:
                break
            contained = False

            for j in range(m):
                if points[b][j][0] == -1:
                    break
                if i != j and np.all(points[b][i] < points[b][j]):
                    contained = True
                    break

            if not contained:
                for k in range(n):
                    new_points[b][counter][k] = points[b][i][k]
                counter += 1
                if counter > 1:
                    ended = False

    if inplace:  # TODO: can be optimized
        np.copyto(points, new_points)
        return ended
    else:
        return new_points, ended


def get_newton_polytope_approx_np(points: np.ndarray, inplace=True):
    """
        The corresponding C++ implementation of the brute force approximation of Newton Polytope.
    """
    assert len(points.shape) == 3

    new_points = np.full(points.shape, -1)

    _c_points = points.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    _c_newPoints = new_points.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    cppUtil.getNewtonPolytope_approx(_c_points, *points.shape, _c_newPoints)

    # TODO: may embed the ending check into the C++ function
    ended = True
    for b in range(points.shape[0]):
        for i in range(points.shape[1]):
            if new_points[b][i][0] == -1:
                break
            if i >= 1:
                ended = False
                break

    if inplace:  # TODO: can be optimized
        np.copyto(points, new_points)
        return ended
    else:
        return new_points, ended


def get_newton_polytope_np(points: np.ndarray, inplace=True):
    """
        Get the Newton Polytope for a set of points.
    """
    return get_newton_polytope_approx_np(points, inplace)
    # TODO: change to a more precise algo to obtain Newton Polytope


def shift_np(points: np.ndarray, coords: List[List[int]], axis: List[int], inplace=True):
    """
        Shift a set of points according to the rule of Hironaka game.
        Directly modify the data of "points".

        points, coords, axis all have a batch dimension. shifting operation is applied batch by batch.

        return:
            inplace=True:
                None
            inplace=False:
                a new numpy array consisting of new point locations and a boolean
    """
    for i in range(len(axis)):
        assert axis[i] in coords[i]
    assert len(points.shape) == 3  # 3-dim numpy array
    assert sum([s == 0 for s in points.shape]) == 0  # no coordinate is zero

    if not inplace:
        new_points = points.copy()

    batch_num, m, dim = points.shape

    # ended = True
    for i in range(batch_num):
        if i >= len(axis):
            break
        for j in range(m):
            if points[i][j][0] == -1:
                break

            if inplace:
                points[i][j][axis[i]] = np.sum(points[i][j][coords[i]])
            else:
                new_points[i][j][axis[i]] = np.sum(points[i][j][coords[i]])

    if not inplace:
        return new_points
