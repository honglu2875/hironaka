import ctypes
import unittest
import numpy as np
import time
from hironaka.util import getNewtonPolytope_approx_py, getNewtonPolytope_approx

stdc = ctypes.cdll.LoadLibrary("libc.so.6")  # or similar to load c library
stdcpp = ctypes.cdll.LoadLibrary("libstdc++.so.6")  # or similar to load c++ library
cppUtil = ctypes.cdll.LoadLibrary("build/cppUtil.so")

cppUtil.getNewtonPolytope_approx.argtypes = [
    ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_long)]
cppUtil.getNewtonPolytope_approx.restype = None


class TestGame(unittest.TestCase):
    def test_getNewtonPolytope_approx(self):
        points = np.array([(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
                           (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6), (17, 18, 20, 30)])
        newPoints = np.full(points.shape, -1)

        _c_points = points.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        _c_newPoints = newPoints.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

        cppUtil.getNewtonPolytope_approx(_c_points, 1, len(points), len(points[0]), _c_newPoints)

        print("test1")
        print(newPoints)

    def test_compare(self):
        points = [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
                  (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6), (17, 18, 20, 30)]

        NUM = 10000

        print("test2")
        t0 = time.time()
        for _ in range(NUM):
            getNewtonPolytope_approx_py(points)
        print(time.time()-t0)

        t0 = time.time()
        for _ in range(NUM):
            getNewtonPolytope_approx(np.array(points))
        print(time.time()-t0)
