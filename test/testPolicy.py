import unittest

from torch import nn

from hironaka.abs import Points
from hironaka.gameHironaka import GameHironaka
from hironaka.policy.NNPolicy import NNPolicy
from hironaka.policy_players.PolicyAgent import PolicyAgent
from hironaka.policy_players.PolicyHost import PolicyHost
from hironaka.util import generate_batch_points


class NN(nn.Module):
    def __init__(self, input_size: int, dimension: int):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, dimension),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TestUtil(unittest.TestCase):
    def test_policy(self):
        pt = [
            [[1, 2, 3, 5], [2, 3, 4, 1]],
            [[1, 1, 1, 1], [4, 4, 4, 3], [9, 8, 7, 10]]
        ]
        coords = [[0, 1], [1, 2]]
        for _ in range(50):
            nnet = NN(4 * 11, 4)
            nn_a = NN(4 * 11 + 4, 4)
            pl_h = NNPolicy(nnet, mode='host', eval_mode=True, max_number_points=11, dimension=4)
            pl_a = NNPolicy(nn_a, mode='agent', eval_mode=True, max_number_points=11, dimension=4)
            out_h = pl_h.predict(pt)
            out_a = pl_a.predict((pt, coords))
            for i in range(2):
                assert sum(out_h[i]) >= 2
            for i in range(2):
                assert out_a[i] in coords[i]

    def test_policy_agents(self):
        nn = NN(4 * 11, 4)
        nn_a = NN(4 * 11 + 4, 4)
        pl_h = NNPolicy(nn, mode='host', eval_mode=True, max_number_points=11, dimension=4)
        pl_a = NNPolicy(nn_a, mode='agent', eval_mode=True, max_number_points=11, dimension=4)

        agent = PolicyAgent(pl_a)
        host = PolicyHost(pl_h)

        game = GameHironaka(Points(generate_batch_points(11, dim=4)), host, agent)

        print(game.state)
        for _ in range(10):
            if not game.step():
                break
            print(game.state)
