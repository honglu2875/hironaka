import unittest

from hironaka.agent import RandomAgent, ChooseFirstAgent
from hironaka.game import GameHironaka
from hironaka.host import Zeillinger
from hironaka.core import Points
from hironaka.src import generate_points, generate_batch_points


class TestGame(unittest.TestCase):
    def test_random_games(self):
        points = Points(generate_points(5))
        agent = RandomAgent()
        host = Zeillinger()
        game = GameHironaka(points, host, agent)

        print(game.state)
        while game.step():
            print(game.state)

        game.print_history()

    def test_random_batch_games(self):
        points = Points(generate_batch_points(5, batch_num=5))
        agent = RandomAgent()
        host = Zeillinger()
        game = GameHironaka(points, host, agent)

        print(game.state)
        while game.step():
            print(game.state)

        game.print_history()

    def test_agent_ignore_batch(self):
        points = Points([[0, 1], [1, 0]])
        agent = ChooseFirstAgent(ignore_batch_dimension=True)
        assert agent.move(points, [0, 1], inplace=False) == 0
        agent.move(points, [0, 1])
        assert str(points.points) == "[[[1, 0]]]"
