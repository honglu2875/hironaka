from hironaka.host import Zeillinger, RandomHost
from hironaka.agent import RandomAgent
from hironaka.agentThom import AgentThom, AgentMorin
from hironaka.game import Game
from hironaka.gameThom import GameThom, GameMorin
from hironaka.util import getNewtonPolytope
from hironaka.util.geomThom import ThomPoints, ThomPointsHomogeneous
from hironaka.util import searchDepth, searchTree
from hironaka.util.searchThom import searchTreeMorin
from treelib import Node, Tree
from dataclasses import dataclass
from typing import List
from collections import deque
from typing import List, Tuple
import numpy as np

from hironaka.util.search import searchDepth


class TestThom(unittest.TestCase):
    def test_ThomPoints(self):

        host = Zeillinger()
        agent = AgentMorin()
        points = ThomPointsHomogeneous(4)
        opoints = ThomPoints(4)
        game = GameMorin(points, host, agent)
        print('Original Points:', opoints)
        print('Homogeneous Points:', points)
        for i in range(100):
            print('Game state:', game.state)
            game.step()
            if game.stopped:
                break

        print("-------")
        print(game.coordHistory)
        print(game.moveHistory)



    def test_ThomTree(self):

        agent = AgentMorin()
        tree = Tree()
        tree.create_node(0, 0, data=Points(points))
        MAX_SIZE=10000
        host = Zeillinger()
        weights = np.ones(len(points[0]))
        searchTreeMorin(points, tree, 0, weights, host, MAX_SIZE=MAX_SIZE)
        tree.show(data_property="data", idhidden=False)
        tree.depth()

print(ThomPoints(3))
#print(ThomPoints(4)[-1])
#print(ThomPointsHomogeneous(4)[-1])
#for k in range(2,4):
#   print(len(ThomPoints(k)[0]),ThomPoints(k))
