import unittest

import torch

import jax.numpy as jnp
from hironaka.core import TensorPoints
from hironaka.core.JAXPoints import JAXPoints
from hironaka.src import get_newton_polytope_torch, shift_torch, reposition_torch, remove_repeated
from hironaka.src._jax_ops import get_newton_polytope_jax, shift_jax, rescale_jax, reposition_jax


class testTorchPoints(unittest.TestCase):
    r = jnp.array([[[1., 2., 3., 4.],
                    [-1., -1., -1., -1.],
                    [4., 1., 2., 3.],
                    [1., 6., 7., 3.]],

                   [[0., 1., 3., 5.],
                    [1., 1., 1., 1.],
                    [-1., -1., -1., -1.],
                    [-1., -1., -1., -1.]]])

    r2 = jnp.array([[[1., 5., 3., 4.],
                     [-1., -1., -1., -1.],
                     [4., 3., 2., 3.],
                     [1., 13., 7., 3.]],

                    [[0., 1., 3., 8.],
                     [1., 1., 1., 3.],
                     [-1., -1., -1., -1.],
                     [-1., -1., -1., -1.]]])

    r3 = jnp.array([[[0., 2., 1., 1.],
                     [-1., -1., -1., -1.],
                     [3., 0., 0., 0.],
                     [0., 10., 5., 0.]],

                    [[0., 0., 2., 5.],
                     [1., 0., 0., 0.],
                     [-1., -1., -1., -1.],
                     [-1., -1., -1., -1.]]])

    rs = jnp.array([[[0.0000, 0.2000, 0.1000, 0.1000],
                        [-1.0000, -1.0000, -1.0000, -1.0000],
                        [0.3000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 1.0000, 0.5000, 0.0000]],

                       [[0.0000, 0.0000, 0.4000, 1.0000],
                        [0.2000, 0.0000, 0.0000, 0.0000],
                        [-1.0000, -1.0000, -1.0000, -1.0000],
                        [-1.0000, -1.0000, -1.0000, -1.0000]]])

    def test_functions(self):
        p = jnp.array(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        ).astype(jnp.float32)
        extreme = jnp.array(
            [
                [[1, 1, 1], [1, 1, 1]]
            ]
        ).astype(jnp.float32)
        extreme_a = jnp.array(
            [
                [[1, 1, 1], [-1, -1, -1]]
            ]
        ).astype(jnp.float32)
        s = jnp.array([
            [[1, 2, 3], [2, 3, 4], [-1, -1, -1]],
            [[0, 0, 0], [-1, -1, -1], [-1, -1, -1]]
        ]).astype(float)
        sr = jnp.array(
            [[[0.25, 0.5, 0.75],
              [0.5, 0.75, 1.],
              [-1., - 1., - 1.]],

             [[0., 0., 0.],
              [-1., - 1., - 1.],
              [-1., - 1., - 1.]]]

        )
        assert jnp.all(get_newton_polytope_jax(p) == self.r)
        assert jnp.all(get_newton_polytope_jax(extreme) == extreme_a)
        p = shift_jax(p, jnp.array([[0, 1, 1, 0], [1, 0, 1, 1]]), jnp.array([1, 3]))
        assert jnp.all(get_newton_polytope_jax(p) == self.r2)
        assert jnp.all(rescale_jax(s) == sr)
        assert jnp.all(reposition_jax(get_newton_polytope_jax(p)) == self.r3)


    def test_points_jax(self):
        p = jnp.array(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        )

        pts = JAXPoints(p)

        pts.get_newton_polytope()
        assert str(pts) == str(self.r)
        pts.shift(jnp.array([[0, 1, 1, 0], [1, 0, 1, 1]]), jnp.array([1, 3]))
        assert str(pts) == str(self.r2)
        pts.reposition()
        assert str(pts) == str(self.r3)
        pts.rescale()
        assert jnp.all(jnp.isclose(pts.points, self.rs))

    """
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

    def test_rescale_by_0(self):
        p = torch.FloatTensor(
            [
                [[0, 0, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        )
        point = TensorPoints(p)
        point.rescale()
        assert point.points.isfinite().all()

    def test_points_hash_is_value_based(self):
        p = TensorPoints(torch.rand(100, 20, 3))
        assert hash(p) == hash(p.copy())

    def test_type(self):
        p = torch.FloatTensor(
            [
                [[0, 0, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]]
            ]
        )
        point = TensorPoints(p, dtype=torch.float32)
        assert point.points.dtype == torch.float32
        point.type(torch.float16)
        assert point.dtype == torch.float16
        assert point.points.dtype == torch.float16
    """
