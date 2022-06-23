import unittest
from hironaka.types.Points import Points
import numpy as np
from hironaka.util import generatePoints


class TestPoints(unittest.TestCase):
    def test_define_point(self):
        points = generatePoints(5)
        print(points)
        points = Points(points)
        print(points)

    def test_operations(self):
        #host = Zeillinger()
        points = Points(
            [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        )
        p2 = Points([(0, 1, 0, 1), (0, 2, 0, 0), (1, 0, 0, 1),
                     (1, 0, 1, 0), (1, 1, 0, 0), (2, 0, 0, 0)])

        result = Points(
            [[[7, 5, 11, 8],
              [8, 1, 26, 18],
                [8, 3, 25, 8],
                [11, 11, 20, 19],
                [11, 12, 24, 6],
                [16, 11, 11, 6]]]
        )

        assert str(result) == str(points.shift([[2, 3]], [2], inplace=False))
        points.shift([[2, 3]], [2])
        assert points.__repr__() == result.__repr__()
        #p2.shift([[1, 3]], [3])
        # print(p2)

    def test_operations2(self):
        p = Points(
            [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        )
        q = Points(
            [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        )
        r = Points(
            [[[7, 5, 3, 8],
              [8, 3, 17, 8],
                [11, 11, 1, 19],
                [11, 12, 18, 6],
                [16, 11, 5, 6],
              [-1, -1, -1, -1]]]
        )
        q.getNewtonPolytope()
        assert str(q) == str(p.getNewtonPolytope(inplace=False))
        assert str(q) == str(r)
