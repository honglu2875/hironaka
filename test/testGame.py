import unittest
from typing import List
from hironaka.util import *
from hironaka.host import Zeillinger
from hironaka.agent import RandomAgent
from hironaka.gameHironaka import GameHironaka


class TestGame(unittest.TestCase):
    def test_random_games(self):
        points = Points(generatePoints(5))
        agent = RandomAgent()
        host = Zeillinger()
        game = GameHironaka(points, host, agent)

        print(game.state)
        while game.step():
            print(game.state)

        game.printHistory()
