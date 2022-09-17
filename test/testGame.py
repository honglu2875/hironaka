import logging
import sys
import unittest

from hironaka.agent import ChooseFirstAgent, RandomAgent
from hironaka.core import ListPoints
from hironaka.game import GameHironaka
from hironaka.host import RandomHost, WeakSpivakovsky, Zeillinger
from hironaka.points import Points
from hironaka.src import generate_batch_points, generate_points


class TestGame(unittest.TestCase):
    def test_random_games(self):
        points = ListPoints(generate_points(5))
        agent = RandomAgent()
        host = RandomHost()
        game = GameHironaka(points, host, agent)

        print(game.state)
        while game.step():
            print(game.state)

        game.print_history()

    def test_random_batch_games(self):
        points = ListPoints(generate_batch_points(5, batch_num=5))
        agent = RandomAgent()
        host = RandomHost()
        game = GameHironaka(points, host, agent)

        print(game.state)
        while game.step():
            print(game.state)

        game.print_history()

    def test_agent_host_ignore_batch(self):
        points = ListPoints([[0, 1], [1, 0]])
        agent = ChooseFirstAgent(ignore_batch_dimension=True)
        assert agent.move(points, [0, 1], inplace=False) == 0
        agent.move(points, [0, 1])
        assert str(points.points) == "[[[1, 0]]]"

        points = ListPoints([[0, 1, 2], [2, 1, 0]])
        host = Zeillinger(ignore_batch_dimension=True)
        assert str(host.select_coord(points)) == "[0, 2]"

    def test_spivakovsky_host(self):
        points = ListPoints([[0, 1, 2], [2, 1, 0]])
        host = WeakSpivakovsky(ignore_batch_dimension=False)
        assert len(host.select_coord(points)[0]) > 1

    def test_wrappers(self):
        random_agent = RandomAgent(ignore_batch_dimension=True)
        random_host = RandomHost(ignore_batch_dimension=True)
        zeillinger = Zeillinger(ignore_batch_dimension=True)
        choose_first_agent = ChooseFirstAgent(
            ignore_batch_dimension=True)  # this guy always chooses the first coordinate

        for host in [random_host, zeillinger]:
            for agent in [random_agent, choose_first_agent]:
                points = Points(generate_points(5))

                game = GameHironaka(points, host, agent, scale_observation=False)
                game.logger.setLevel(logging.INFO)
                if not game.logger.hasHandlers():
                    game.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
                print(f"{host.__class__.__name__} is playing against {agent.__class__.__name__}")
                while game.step(verbose=1):
                    print(game.state)
                game.print_history()
