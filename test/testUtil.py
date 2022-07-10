import unittest

import numpy as np

from hironaka.src import get_batched_padded_array, batched_coord_list_to_binary


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
