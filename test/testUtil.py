import unittest

import numpy as np

from hironaka.src import get_batched_padded_array, batched_coord_list_to_binary, get_newton_polytope_lst, get_shape


class TestUtil(unittest.TestCase):
    def test_batch_padded_array(self):
        pt = [
            [[1, 2, 3], [2, 3, 4]],
            [[1, 1, 1], [4, 4, 4], [9, 8, 7]]
        ]
        r = np.array([
            [[1, 2, 3], [2, 3, 4], [-1, -1, -1], [-1, -1, -1]],
            [[1, 1, 1], [4, 4, 4], [9, 8, 7], [-1, -1, -1]]
        ])
        r2 = get_batched_padded_array(pt, 4, constant_value=-1)
        print(r)
        print(r2)
        assert (r == r2).all()

    def test_batched_coord_to_bin(self):
        coords = [[1, 2, 3], [2, 0, 1]]
        r = np.array([[0, 1, 1, 1], [1, 1, 1, 0]])
        assert (batched_coord_list_to_binary(coords, 4) == r).all()

    def test_true_newton_polytope(self):
        p = [[[1., 0.], [0.9, 0.9], [0., 1.]]]
        r = [[[1., 0.], [0., 1.]]]

        assert str(get_newton_polytope_lst(p, inplace=False)) == str(r)

        p = [[[0.37807224, 0.60967653, 0.50641324]]]
        assert str(get_newton_polytope_lst(p, inplace=False)) == str(p)

        p = [
            [[0.11675344, 0.39038985, 0.55826897, 0.06529552],
             [0.9846373, 0.45638349, 0.70517085, 0.90032522],
             [0.01027646, 0.11461289, 0.89243383, 0.634063],
             [0.58811481, 0.99114348, 0.61889408, 0.59967777],
             [0.91356043, 0.62654142, 0.69501398, 0.68474988],
             [0.88135114, 0.30110585, 0.04229966, 0.03769748],
             [0.37982495, 0.17156216, 0.33440668, 0.48339728],
             [0.12123305, 0.15986878, 0.11907919, 0.59999993],
             [0.9496461, 0.16063278, 0.42188375, 0.66339718],
             [0.59075721, 0.17488182, 0.89326396, 0.01449242]],
            [[0.64929492, 0.8896327, 0.98860123, 0.52941554],
             [0.25994605, 0.03554693, 0.43534583, 0.19954576],
             [0.62238657, 0.33769715, 0.2672676, 0.67115147],
             [0.23643443, 0.51686672, 0.72861238, 0.0351913],
             [0.3788386, 0.67130138, 0.87033132, 0.4363841],
             [0.30030881, 0.11823987, 0.20820786, 0.49078142],
             [0.25722259, 0.32548102, 0.97916295, 0.0842389],
             [0.06561767, 0.55689435, 0.70502167, 0.27102844],
             [0.38096357, 0.59775385, 0.97628977, 0.60265799],
             [0.28909349, 0.08945314, 0.80995294, 0.63317]]
        ]

        r = [
            [[0.88135114, 0.30110585, 0.04229966, 0.03769748],
             [0.59075721, 0.17488182, 0.89326396, 0.01449242],
             [0.12123305, 0.15986878, 0.11907919, 0.59999993],
             [0.11675344, 0.39038985, 0.55826897, 0.06529552],
             [0.01027646, 0.11461289, 0.89243383, 0.634063]],
            [[0.30030881, 0.11823987, 0.20820786, 0.49078142],
             [0.25994605, 0.03554693, 0.43534583, 0.19954576],
             [0.25722259, 0.32548102, 0.97916295, 0.0842389],
             [0.23643443, 0.51686672, 0.72861238, 0.0351913],
             [0.06561767, 0.55689435, 0.70502167, 0.27102844]]]

        assert str(get_newton_polytope_lst(p, inplace=False)) == str(r)

    def test_get_shape_extra_character(self):
        p = [[1, 2, 3, 'd'], [2, 3, 4]]
        assert get_shape(p) == (2, 3)
