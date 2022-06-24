import unittest

from hironaka.types.Points import Points
from hironaka.types.src import make_nested_list
from hironaka.util import generate_points


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
        for i in [True, False]:
            p = Points(make_nested_list(
                [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
                 (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
            ), use_np=i)
            q = Points(make_nested_list(
                [(7, 5, 3, 8), (8, 9, 8, 18), (8, 3, 17, 8),
                 (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
            ), use_np=i)
            r = Points(make_nested_list(
                [[[7, 5, 3, 8],
                  [8, 3, 17, 8],
                  [11, 11, 1, 19],
                  [11, 12, 18, 6],
                  [16, 11, 5, 6],
                  [-1, -1, -1, -1]]]
            ), use_np=i)
            r2 = Points(make_nested_list(
                [[[16, 11, 5, 6], [11, 12, 18, 6], [11, 11, 1, 19], [8, 3, 17, 8], [7, 5, 3, 8]]]
            ), use_np=i)
            q.get_newton_polytope()
            assert str(q) == str(p.get_newton_polytope(inplace=False))
            if i:
                assert str(q) == str(r)
            else:
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
