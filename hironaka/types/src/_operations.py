from typing import List, Tuple
import numpy as np
import ctypes

stdc = ctypes.cdll.LoadLibrary("libc.so.6")
stdcpp = ctypes.cdll.LoadLibrary("libstdc++.so.6")
cppUtil = ctypes.cdll.LoadLibrary("build/cppUtil.so")

cppUtil.getNewtonPolytope_approx.argtypes = [
    ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
cppUtil.getNewtonPolytope_approx.restype = None


def getNewtonPolytope_approx_py(points: np.ndarray, inplace=True):
    """
        A simple-minded quick-and-dirty method to obtain an approximation of Newton Polytope disregarding convexity.
    """

    assert len(points.shape) == 3

    batchNum, m, n = points.shape
    newPoints = np.full(points.shape, -1)
    ended = True

    for b in range(batchNum):
        for i in range(m):
            if points[b][i][0] == -1:
                break
            contained = False
            counter = 0
            if i >= 1:
                ended = False

            for j in range(m):
                if sum([points[b][i][k] > points[b][j][k] for k in range(n)]) == 0:
                    contained = True
                    break
            if not contained:
                for k in range(n):
                    newPoints[b][counter][k] = points[b][i][k]
                counter += 1

    if inplace:  # TODO: can be optimized
        np.copyto(points, newPoints)
        return ended
    else:
        return newPoints, ended


def getNewtonPolytope_approx(points: np.ndarray, inplace=True):
    """
        The corresponding C++ implementation of the brute force approximation of Newton Polytope.
    """
    assert len(points.shape) == 3

    newPoints = np.full(points.shape, -1)

    _c_points = points.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    _c_newPoints = newPoints.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

    cppUtil.getNewtonPolytope_approx(_c_points, *points.shape, _c_newPoints)

    # TODO: may embed the ending check into the C++ function
    ended = True
    for b in range(points.shape[0]):
        for i in range(points.shape[1]):
            if newPoints[b][i][0] == -1:
                break
            if i >= 1:
                ended = False
                break

    if inplace:  # TODO: can be optimized
        np.copyto(points, newPoints)
        return ended
    else:
        return newPoints, ended


def getNewtonPolytope(points: np.ndarray, inplace=True):
    """
        Get the Newton Polytope for a set of points.
    """
    return getNewtonPolytope_approx(points, inplace)  # TODO: change to a more precise algo to obtain Newton Polytope


def shift(points: np.ndarray, coords: List[List[int]], axis: List[int], inplace=True):
    """
        Shift a set of points according to the rule of Hironaka game.
        Directly modify the data of "points".

        points, coords, axis all have a batch dimension. shifting operation is applied batch by batch.

        return:
            inplace=True:
                True if number of points in each batch is less than or equal to 1. False otherwise.
            inplace=False:
                a new numpy array consisting of new point locations and a boolean showing if the game has ended
    """
    for i in range(len(axis)):
        assert axis[i] in coords[i]
    assert len(points.shape) == 3  # 3-dim numpy array
    assert sum([s == 0 for s in points.shape]) == 0  # no coordinate is zero

    if not inplace:
        newPoints = points.copy()

    batchNum, m, dim = points.shape

    ended = True
    for i in range(batchNum):
        if i >= len(axis):
            break
        for j in range(m):
            if points[i][j][0] == -1:
                break
            if j >= 1:
                ended = False  # there exists at least one batch not terminal
            # print(points)
            if inplace:
                points[i][j][axis[i]] = sum([points[i][j][k] for k in coords[i]])
            else:
                newPoints[i][j][axis[i]] = sum([points[i][j][k] for k in coords[i]])
            # print(points)

    if inplace:
        return ended
    else:
        return newPoints, ended
