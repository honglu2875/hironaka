import unittest

from hironaka.agent import RandomAgent
from hironaka.gameHironaka import GameHironaka
from hironaka.host import Zeillinger
from hironaka.util import *


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
