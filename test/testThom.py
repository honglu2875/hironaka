import unittest

from treelib import Tree

from hironaka.core import Points
from hironaka.agent import AgentMorin
from hironaka.game import GameMorin
from hironaka.host import ZeillingerLex, WeakSpivakovsky
from hironaka.src import thom_monomial_ideal, thom_points, thom_points_homogeneous
from hironaka.util import search_tree_morin


class TestThom(unittest.TestCase):
    def test_game(self):
        N = 3
        host = WeakSpivakovsky()
        agent = AgentMorin()
        points = thom_points_homogeneous(N)
        opoints = thom_points(N)
        game = GameMorin(Points([points], distinguished_points=[len(points) - 1]), host, agent)

        ro = [(4, 1, 0, 1, 0, 0, 0), (4, 1, 0, 0, 0, 1, 0), (4, 1, 0, 0, 0, 0, 1), (4, 0, 2, 0, 0, 0, 0),
              (4, 0, 1, 0, 1, 0, 0), (4, 0, 0, 0, 2, 0, 0), (3, 2, 1, 0, 0, 0, 0), (3, 2, 0, 0, 1, 0, 0),
              (3, 0, 1, 0, 0, 1, 0), (3, 0, 1, 0, 0, 0, 1), (3, 0, 0, 1, 1, 0, 0), (3, 0, 0, 0, 1, 0, 1),
              (2, 4, 0, 0, 0, 0, 0), (2, 2, 0, 0, 0, 1, 0), (2, 2, 0, 0, 0, 0, 1), (2, 1, 1, 0, 1, 0, 0),
              (2, 1, 0, 0, 2, 0, 0), (1, 3, 0, 0, 1, 0, 0), (1, 1, 0, 0, 1, 0, 1)]
        rhom = [[2, 1, 0, 1, 0, 0, 0], [2, 1, 0, 0, 0, 1, 0], [2, 1, 0, 0, 0, 0, 1], [2, 0, 2, 0, 0, 0, 0],
                [2, 0, 1, 0, 1, 0, 0], [2, 0, 0, 0, 2, 0, 0], [1, 2, 1, 0, 0, 0, 0], [1, 2, 0, 0, 1, 0, 0],
                [2, 0, 1, 0, 0, 1, 0], [2, 0, 1, 0, 0, 0, 1], [2, 0, 0, 1, 1, 0, 0], [2, 0, 0, 0, 1, 0, 1],
                [0, 4, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 1, 0], [1, 2, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 2, 0, 0], [0, 3, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1]]
        assert str(opoints) == str(ro)
        assert str(rhom) == str(points)

        for i in range(100):
            print('Game state:', game.state)
            game.step()
            if game.stopped:
                break

        print("-------")
        print(game.coord_history)
        print(game.move_history)

    def test_ThomTreeHom(self):
        points = thom_points_homogeneous(4)
        print(f"Points: {points}")
        dimension = len(points[0])
        initial_points = Points([points], distinguished_points=[len(points) - 1])

        tree = Tree()
        tree.create_node(0, 0, data=initial_points)
        MAX_SIZE = 10000

        host = WeakSpivakovsky()
        weights = [1]*dimension
        search_tree_morin(initial_points, tree, 0, weights, host, max_size=MAX_SIZE)
        tree.show(data_property="points", idhidden=False)
        tree.depth()

    def test_ThomTreeOriginal(self):
        points = [list(np.array(thom_points(4)[i])+[np.dot([0,1,1,1,1,1,1],np.array(thom_points(4)[i]))-4,0,0,0,0,0,0]) for i in range(len(thom_points(4)))]
        print(f"Points: {points}")
        dimension = len(points[0])
        initial_points = Points([points], distinguished_points=[len(points) - 1])

        tree = Tree()
        tree.create_node(0, 0, data=initial_points)
        MAX_SIZE = 10000

        host = WeakSpivakovsky()
        weights = [1,1,2,2,3,3,3]
        search_tree_morin(initial_points, tree, 0, weights, host, max_size=MAX_SIZE)
        tree.show(data_property="points", idhidden=False)
        tree.depth()

    def test_thom_points(self):
        thom_points_homogeneous_2 = "[[0, 1]]"
        assert str(thom_points_homogeneous(2)) == thom_points_homogeneous_2

        thom_points_2 = "[(1, 1)]"
        assert str(thom_points(2)) == thom_points_2

        thom_monomial_idea_3 = \
            "[-b[0, 0]**3*b[1, 2], 2*b[0, 0]**2*b[1, 1]**2, -b[0, 0]**3*b[2, 2], 0, 0, 0, b[0, 0]*b[1, 1]*b[2, 2]]"
        assert str(thom_monomial_ideal(3)) == thom_monomial_idea_3

        thom_points_homogeneous_3 = "[[1, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 0], [0, 1, 0, 1]]"
        assert str(thom_points_homogeneous(3)) == thom_points_homogeneous_3

        thom_points_homogeneous_4 = \
            "[[2, 1, 0, 1, 0, 0, 0], [2, 1, 0, 0, 0, 1, 0], [2, 1, 0, 0, 0, 0, 1], [2, 0, 2, 0, 0, 0, 0], [2, 0, 1, 0, 1, 0, 0], [2, 0, 0, 0, 2, 0, 0], [1, 2, 1, 0, 0, 0, 0], [1, 2, 0, 0, 1, 0, 0], [2, 0, 1, 0, 0, 1, 0], [2, 0, 1, 0, 0, 0, 1], [2, 0, 0, 1, 1, 0, 0], [2, 0, 0, 0, 1, 0, 1], [0, 4, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 1, 0], [1, 2, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 2, 0, 0], [0, 3, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1]]"
        assert str(thom_points_homogeneous(4)) == thom_points_homogeneous_4
