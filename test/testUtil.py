import unittest

import numpy as np
import torch
from torch import nn

from hironaka.src import get_batched_padded_array, batched_coord_list_to_binary, get_newton_polytope_lst, get_shape, \
    merge_experiences, HostActionEncoder
from hironaka.trainer.nets import expand_net_list, create_mlp


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

    def test_create_mlp(self):
        net_arch = [12, 13, 'b',
                    {'repeat': 3,
                     'net_arch':
                         [14, 15,
                          {'repeat': 5,
                           'net_arch': ['b', 16]}
                          ]}
                    ]

        expanded = [12, 13, 'b', 14, 15, 'b', 16, 'b', 16, 'b', 16, 'b', 16, 'b', 16, 14, 15, 'b', 16, 'b', 16, 'b', 16,
                    'b', 16, 'b', 16, 14, 15, 'b', 16, 'b', 16, 'b', 16, 'b', 16, 'b', 16]

        assert expand_net_list(net_arch) == expanded

        net = create_mlp(nn.Flatten(),
                         [16, 12, {'repeat': 5, 'net_arch': [3, 'b', 4, 'b', {'repeat': 2, 'net_arch': [5]}]}], 60, 8)

        net.train(False)
        print(net(torch.Tensor([[1] * 60])))  # Make sure all networks connect and the evaluation is successful.

    def test_merge_exp(self):
        exps = []
        for i in range(5):
            obs = {'a': torch.full((2, 10), 4 * i, dtype=torch.float),
                   'b': torch.full((2, 5), 4 * i + 1, dtype=torch.float)}
            next_obs = {'a': torch.full((2, 10), 4 * i + 2, dtype=torch.float),
                        'b': torch.full((2, 5), 4 * i + 3, dtype=torch.float)}
            actions = torch.ones((2, 1)).type(torch.int32)
            rewards = torch.ones((2, 1)).type(torch.float)
            dones = torch.ones((2, 1)).type(torch.bool)
            exps.append((obs, actions, rewards, dones, next_obs))

        r = ({'a': torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                      [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                                      [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                      [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                                      [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                                      [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                                      [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]]),
              'b': torch.FloatTensor([[1, 1, 1, 1, 1],
                                      [1, 1, 1, 1, 1],
                                      [5, 5, 5, 5, 5],
                                      [5, 5, 5, 5, 5],
                                      [9, 9, 9, 9, 9],
                                      [9, 9, 9, 9, 9],
                                      [13, 13, 13, 13, 13],
                                      [13, 13, 13, 13, 13],
                                      [17, 17, 17, 17, 17],
                                      [17, 17, 17, 17, 17]])},
             torch.IntTensor([[1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1]]), torch.FloatTensor([[1.],
                                                        [1.],
                                                        [1.],
                                                        [1.],
                                                        [1.],
                                                        [1.],
                                                        [1.],
                                                        [1.],
                                                        [1.],
                                                        [1.]]), torch.BoolTensor([[True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True],
                                                                                  [True]]),
             {'a': torch.FloatTensor([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                      [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                                      [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                                      [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                      [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                                      [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                                      [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
                                      [18, 18, 18, 18, 18, 18, 18, 18, 18, 18],
                                      [18, 18, 18, 18, 18, 18, 18, 18, 18, 18]]),
              'b': torch.FloatTensor([[3, 3, 3, 3, 3],
                                      [3, 3, 3, 3, 3],
                                      [7, 7, 7, 7, 7],
                                      [7, 7, 7, 7, 7],
                                      [11, 11, 11, 11, 11],
                                      [11, 11, 11, 11, 11],
                                      [15, 15, 15, 15, 15],
                                      [15, 15, 15, 15, 15],
                                      [19, 19, 19, 19, 19],
                                      [19, 19, 19, 19, 19]])})

        assert str(merge_experiences(exps)) == str(r)

    def test_HostActionEncoder(self):
        encoder = HostActionEncoder(3)
        assert encoder.encode([0, 1]) == 0
        print(encoder.encode_tensor(torch.tensor([[1., 1., 0.]])))
        assert encoder.encode_tensor(torch.tensor([[1., 1., 0.]])).eq(torch.tensor([0], dtype=torch.int64)).all()
        assert encoder.encode([0, 1, 2]) == 3
        assert encoder.encode_tensor(torch.tensor([[1., 1., 1.]])).eq(torch.tensor([3], dtype=torch.int64)).all()
        encoder = HostActionEncoder(4)
        for i in range(11):
            assert encoder.encode(encoder.decode(i)) == i
        assert encoder.encode_tensor(encoder.decode_tensor(torch.arange(11))).eq(torch.arange(11)).all()
