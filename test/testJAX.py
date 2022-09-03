import unittest

import torch

import jax.numpy as jnp
from hironaka.core import TensorPoints, JAXPoints
from hironaka.jax.players import decode_table, batch_encode, batch_encode_one_hot, all_coord_host_fn, random_host_fn, \
    char_vector_of_pts, zeillinger_fn_slice, zeillinger_fn, random_agent_fn, decode, batch_decode
from hironaka.jax.util import flatten, make_agent_obs

from hironaka.trainer.MCTSJAXTrainer.MCTSJAXTrainer import JAXObs
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

    def test_jax_obs(self):
        host_obs = jnp.array([
            [[1, 2, 3], [2, 3, 4], [0, 1, 0], [-1, -1, -1]],
            [[4, 2, 2], [-1, -1, -1], [0, 0, 1], [-1, -1, -1]]
        ]).astype(jnp.float32)
        agent_obs = {'points': jnp.copy(host_obs),
                     'coords': jnp.array([[0, 1, 1], [1, 1, 1]])}
        combined = jnp.concatenate([agent_obs['points'].reshape(2, -1), agent_obs['coords']],
                                   axis=1)

        h_o = JAXObs('host', host_obs)
        assert jnp.all(h_o.get_features() == host_obs.reshape(2, -1))
        assert jnp.all(h_o.get_points() == host_obs)
        assert h_o.get_coords() is None
        a_o = JAXObs('agent', agent_obs)
        assert jnp.all(a_o.get_features() == combined)
        assert jnp.all(a_o.get_points() == host_obs)
        assert jnp.all(a_o.get_coords() == agent_obs['coords'])
        a_o2 = JAXObs('agent', combined, dimension=3)
        assert jnp.all(a_o2.get_features() == combined)
        assert jnp.all(a_o2.get_points() == host_obs)
        assert jnp.all(a_o2.get_coords() == agent_obs['coords'])

        with self.assertRaises(Exception) as context:
            a_o = JAXObs('agent', host_obs)
        with self.assertRaises(Exception) as context:
            a_o = JAXObs('agent', host_obs, dimension=7)

    def test_encode_decode(self):
        decode_3 = jnp.array([
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ])
        assert jnp.all(decode_table(3) == decode_3)
        encode_in = jnp.array([
            [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        encode_out = jnp.array([
            1, 3, 2
        ])
        encode_one_hot_out = jnp.array([
            [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]
        ])
        assert jnp.all(batch_encode(encode_in) == jnp.array(encode_out))
        assert jnp.all(batch_encode_one_hot(encode_in) == jnp.array(encode_one_hot_out))
        assert jnp.all(decode(jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])) == jnp.array([0, 1, 1, 0]))
        assert jnp.all(batch_decode(encode_one_hot_out) == encode_in)

    def test_hosts(self):
        obs = jnp.array([
            [[1, 2, 3], [2, 3, 4]],
            [[0, 1, 2], [-1, -1, -1]]
        ])
        all_coord_out = jnp.array([
            [[0, 0, 0, 1], [0, 0, 0, 1]]
        ])
        random_host_fn(obs)
        assert jnp.all(all_coord_host_fn(obs) == all_coord_out)
        pts = jnp.array([
            [[0, 0, 4], [5, 0, 1], [1, 5, 1], [0, 25, 0]]
        ])
        r = jnp.array([0, 1, 0, 0])
        assert jnp.all(zeillinger_fn_slice(pts[0]) == r)
        obs2 = jnp.array([[[19, 15, 0, 10],
                           [12, 0, 14, 9],
                           [8, 14, 8, 18],
                           [3, 18, 17, 12],
                           [19, 6, 1, 13]],

                          [[17, 3, 6, 9],
                           [19, 1, 13, 12],
                           [14, 0, 6, 7],
                           [2, 15, 3, 16],
                           [0, 16, 1, 5]],

                          [[19, 0, 8, 6],
                           [8, 9, 17, 1],
                           [2, 3, 7, 14],
                           [6, 19, 9, 12],
                           [0, 19, 19, 14]]], dtype=jnp.float32)
        r = batch_encode_one_hot(jnp.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]]))
        assert jnp.all(zeillinger_fn(obs2) == r)

    def test_agent(self):
        obs = jnp.array([
            [[1, 2, 3], [2, 3, 4]],
            [[0, 1, 2], [-1, -1, -1]]
        ])
        coords = jnp.array([
            [1, 1, 0], [0, 1, 1]
        ])
        random_agent_fn(make_agent_obs(obs, coords), (2, 3))

