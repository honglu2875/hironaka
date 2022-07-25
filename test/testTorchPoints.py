import unittest

import torch

from hironaka.core import TensorPoints
from hironaka.src import get_newton_polytope_torch, shift_torch, reposition_torch, remove_repeated


class testTorchPoints(unittest.TestCase):
    r = torch.Tensor([[[1., 2., 3., 4.],
                       [-1., -1., -1., -1.],
                       [4., 1., 2., 3.],
                       [1., 6., 7., 3.]],

                      [[0., 1., 3., 5.],
                       [1., 1., 1., 1.],
                       [-1., -1., -1., -1.],
                       [-1., -1., -1., -1.]]])

    r2 = torch.Tensor([[[1., 5., 3., 4.],
                        [-1., -1., -1., -1.],
                        [4., 3., 2., 3.],
                        [1., 13., 7., 3.]],

                       [[0., 1., 3., 8.],
                        [1., 1., 1., 3.],
                        [-1., -1., -1., -1.],
                        [-1., -1., -1., -1.]]])

    r3 = torch.Tensor([[[0., 2., 1., 1.],
                        [-1., -1., -1., -1.],
                        [3., 0., 0., 0.],
                        [0., 10., 5., 0.]],

                       [[0., 0., 2., 5.],
                        [1., 0., 0., 0.],
                        [-1., -1., -1., -1.],
                        [-1., -1., -1., -1.]]])

    rs = torch.Tensor([[[0.0000, 0.2000, 0.1000, 0.1000],
                        [-1.0000, -1.0000, -1.0000, -1.0000],
                        [0.3000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 1.0000, 0.5000, 0.0000]],

                       [[0.0000, 0.0000, 0.4000, 1.0000],
                        [0.2000, 0.0000, 0.0000, 0.0000],
                        [-1.0000, -1.0000, -1.0000, -1.0000],
                        [-1.0000, -1.0000, -1.0000, -1.0000]]])

    def test_functions(self):
        p = torch.FloatTensor(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        )
        assert torch.all(get_newton_polytope_torch(p, inplace=False).eq(self.r))
        get_newton_polytope_torch(p, inplace=True)
        assert torch.all(p.eq(self.r))
        assert torch.all(shift_torch(p, [[1, 2], [0, 2, 3]], [1, 3], inplace=False).eq(self.r2))
        shift_torch(p, [[1, 2], [0, 2, 3]], [1, 3], inplace=True)
        assert torch.all(p.eq(self.r2))
        assert torch.all(reposition_torch(p, inplace=False).eq(self.r3))
        reposition_torch(p, inplace=True)
        assert torch.all(p.eq(self.r3))

    def test_pointstensor(self):
        p = torch.FloatTensor(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        )

        pts = TensorPoints(p)

        pts.get_newton_polytope()
        assert str(pts) == str(self.r)
        pts.shift([[1, 2], [0, 2, 3]], [1, 3])
        assert str(pts) == str(self.r2)
        pts.reposition()
        assert str(pts) == str(self.r3)
        pts.rescale()
        assert str(pts) == str(self.rs)

    def test_invalid_actions(self):
        p = torch.FloatTensor(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        )
        pts = TensorPoints(p)

        pts.get_newton_polytope()
        pts.shift([[1], [0, 2, 3]], [0, 1])
        assert str(pts) == str(self.r)  # Both actions are invalid. No actions should have been taken.

        p = torch.FloatTensor(
            [[[1, 0, 0, 1], [-1, -1, -1, -1]]]
        )
        q = torch.FloatTensor(
            [[[1, 1, 0, 1], [-1, -1, -1, -1]]]
        )
        pts = TensorPoints(p).copy()
        pts.shift([[0, 1]], [1], ignore_ended_games=True)
        assert str(pts) == str(TensorPoints(p))  # Game is already over. No action should be taken.
        pts.shift([[0, 1]], [1], ignore_ended_games=False)
        assert str(pts) == str(TensorPoints(q))  # Force operations on ended games.

    def test_remove_repeated(self):
        p = torch.Tensor([[[0., 0., 0.],
                           [0., 0., 0.]]])

        r = torch.Tensor([[[0., 0., 0.],
                           [-1., -1., -1.]]])

        remove_repeated(p)
        assert p.eq(r).all()

    def test_functions_2(self):
        p = TensorPoints(
            torch.FloatTensor([[[4., 2., 4.],
                                [4., 0., 3.],
                                [3., 2., 4.],
                                [3., 3., 3.],
                                [3., 4., 2.],
                                [3., 0., 0.],
                                [3., 1., 2.],
                                [3., 0., 1.],
                                [3., 3., 3.],
                                [3., 0., 3.],
                                [2., 0., 1.],
                                [2., 1., 3.],
                                [2., 1., 1.],
                                [1., 4., 2.],
                                [1., 4., 1.],
                                [1., 4., 3.],
                                [1., 0., 4.],
                                [1., 3., 4.],
                                [0., 0., 0.],
                                [0., 0., 0.]]])
        )

        r = torch.FloatTensor([[[-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [-1., -1., -1.],
                                [0., 0., 0.],
                                [-1., -1., -1.]]])

        p.get_newton_polytope()

        assert p.points.eq(r).all()
