
from hironaka.host import Zeillinger, RandomHost
from hironaka.agent import RandomAgent, AgentThom
from hironaka.game import Game, GameThom
from hironaka.util import getNewtonPolytope, ThomPoints, ThomPointsHomogeneous
from hironaka.util.search import searchDepth


host = Zeillinger()
agent = AgentThom()
points = getNewtonPolytope(ThomPointsHomogeneous(3))
game = GameThom(points, host, agent)
print(points)
for i in range(10):
    game.step()
    print('Game state:', game.state)
    if game.stopped:
        break

print("-------")
print(game.coordHistory)
print(game.moveHistory)