import unittest

import numpy as np

from hironaka.core.Points import Points
from hironaka.src import make_nested_list, generate_points


class TestPoints(unittest.TestCase):
    def test_define_point(self):
        points = generate_points(5)
        print(points)
        points = Points(points)
        print(points)

    def test_operations(self):
        # host = Zeillinger()
        points = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ))
        p2 = Points(make_nested_list(
            [(0, 1, 0, 1), (0, 2, 0, 0), (1, 0, 0, 1),
             (1, 0, 1, 0), (1, 1, 0, 0), (2, 0, 0, 0)]))

        result = Points(make_nested_list(
            [[[7, 5, 11, 8],
              [8, 1, 26, 18],
              [8, 3, 25, 8],
              [11, 11, 20, 19],
              [11, 12, 24, 6],
              [16, 11, 11, 6]]]
        ))

        assert str(result) == str(points.shift([[2, 3]], [2], inplace=False))
        points.shift([[2, 3]], [2])
        assert points.__repr__() == result.__repr__()
        # p2.shift([[1, 3]], [3])
        # print(p2)

    def test_operations2(self):
        p = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ))
        q = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ))
        """
        # For numpy Points test
        r = Points(make_nested_list(
            [[[7, 5, 3, 8],
              [8, 3, 17, 8],
              [11, 11, 1, 19],
              [11, 12, 18, 6],
              [16, 11, 5, 6],
              [-1, -1, -1, -1]]]
        ))
        """
        r2 = Points(make_nested_list(
            [[[16, 11, 5, 6], [11, 12, 18, 6], [11, 11, 1, 19], [8, 3, 17, 8], [7, 5, 3, 8]]]
        ))
        q.get_newton_polytope()

        assert str(q) == str(p.get_newton_polytope(inplace=False))
        assert str(q) == str(r2)

    def test_features(self):
        p = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ))
        r = [[[61, 51, 52, 65], [675, 501, 712, 885], [8125, 5271, 11410, 14147], [105411, 57285, 193300, 246081],
              [1453021, 633351, 3345562, 4446755], [20962275, 7076901, 58428292, 81675705]]]
        print(p.get_sym_features())
        assert str(p.get_sym_features()) == str(r)

    def test_get_batch(self):
        p = Points(make_nested_list(
            [[(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
              (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)],
             [(0, 1, 0, 1), (0, 2, 0, 0), (1, 0, 0, 1),
              (1, 0, 1, 0), (1, 1, 0, 0), (2, 0, 0, 0)]]
        ))
        r = [[0, 1, 0, 1], [0, 2, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [2, 0, 0, 0]]

        assert str(p.get_batch(1)) == str(r)

    def test_numpy_input_without_use_np(self):
        p = Points(np.array(
            [[[7, 5, 11, 8],
              [8, 1, 26, 18],
              [8, 3, 25, 8],
              [11, 11, 20, 19],
              [-1, -1, -1, -1],
              [-1, -1, -1, -1]]]
        ))

        r = [[[7, 5, 11, 8],
              [8, 1, 26, 18],
              [8, 3, 25, 8],
              [11, 11, 20, 19]]]

        assert str(p.points) == str(r)

    def test_scale(self):
        points = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ))

        r = [[[0.3684210526315789, 0.2631578947368421, 0.15789473684210525, 0.42105263157894735],
              [0.42105263157894735, 0.05263157894736842, 0.42105263157894735, 0.9473684210526315],
              [0.42105263157894735, 0.15789473684210525, 0.8947368421052632, 0.42105263157894735],
              [0.5789473684210527, 0.5789473684210527, 0.05263157894736842, 1.0],
              [0.5789473684210527, 0.631578947368421, 0.9473684210526315, 0.3157894736842105],
              [0.8421052631578947, 0.5789473684210527, 0.2631578947368421, 0.3157894736842105]]]

        points.rescale()
        assert str(points) == str(r)
        # p2.shift([[1, 3]], [3])
        # print(p2)

    def test_value_threshold(self):
        points = Points([[[5e7, 5e7+1, 1e7], [1, 1, 2]]], value_threshold=int(1e8))
        assert not points.exceed_threshold()
        points.shift([[0, 1]], [0])
        assert points.exceed_threshold()

    def test_reposition(self):
        points = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ))
        r = Points([[[0, 4, 2, 2], [1, 0, 7, 12], [1, 2, 16, 2], [4, 10, 0, 13], [4, 11, 17, 0], [9, 10, 4, 0]]])
        points.reposition()
        a = points.reposition(inplace=False)
        assert str(points) == str(r)
        assert str(points) == str(a)

    def test_distinguished_elements(self):
        points = Points(make_nested_list(
            [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        ), distinguished_points=[2])

        points.get_newton_polytope()
        d_ind = points.distinguished_points[0]
        assert tuple(points.points[0][d_ind]) == (8, 3, 17, 8)

        points.shift([[0, 1]], [0])
        points.get_newton_polytope()
        d_ind = points.distinguished_points[0]
        assert tuple(points.points[0][d_ind]) == (11, 3, 17, 8)

        points.shift([[0, 2]], [0])
        points.shift([[2, 3]], [2])
        points.shift([[0, 1]], [1])
        points.get_newton_polytope()
        d_ind = points.distinguished_points[0]
        assert d_ind is None
