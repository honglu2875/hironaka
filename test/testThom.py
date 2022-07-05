
from hironaka.host import Zeillinger, RandomHost
from hironaka.agent import RandomAgent, AgentThom, AgentMorin
from hironaka.game import Game, GameThom, GameMorin
from hironaka.util import getNewtonPolytope, ThomPoints, ThomPointsHomogeneous
from hironaka.util import searchDepth, searchTree
from treelib import Node, Tree
from dataclasses import dataclass
from typing import List
from collections import deque
from typing import List, Tuple

from hironaka.util.search import searchDepth


host = Zeillinger()
agent = AgentMorin()
points = ThomPointsHomogeneous(3)
opoints = ThomPoints(3)
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

#@title full search under Zeillinger policy
@dataclass
class Points:
    data:List

tree = Tree()
tree.create_node(0, 0, data=Points(points))
MAX_SIZE=1000
host = Zeillinger()
searchTree(points, tree, 0, host, MAX_SIZE=MAX_SIZE)
tree.show(data_property="data", idhidden=False)
tree.depth()