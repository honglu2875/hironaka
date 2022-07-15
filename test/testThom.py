import unittest

import numpy as np
from treelib import Tree

from hironaka.abs import Points
from hironaka.agentThom import AgentMorin
from hironaka.gameThom import GameMorin
from hironaka.host import Zeillinger
from hironaka.util.geomThom import thom_points, thom_points_homogeneous, thom_monomial_ideal
from hironaka.util.searchThom import search_tree_morin


class TestThom(unittest.TestCase):
    def test_game(self):
        N = 3
        host = Zeillinger()
        agent = AgentMorin()
        points = thom_points_homogeneous(N)
        opoints = thom_points(N)
        game = GameMorin(Points([points], distinguished_points=[len(points) - 1]), host, agent)
        print('Original Points:', opoints)
        print('Homogeneous Points:', points)
        for i in range(100):
            print('Game state:', game.state)
            game.step()
            if game.stopped:
                break

        print("-------")
        print(game.coord_history)
        print(game.move_history)

    def test_ThomTree(self):
        points = thom_points_homogeneous(3)
        print(f"Points: {points}")
        dimension = len(points[0])
        initial_points = Points([points], distinguished_points=[len(points) - 1])

        tree = Tree()
        tree.create_node(0, 0, data=initial_points)
        MAX_SIZE = 10000

        host = Zeillinger()
        weights = [1] * dimension
        search_tree_morin(initial_points, tree, 0, weights, host, max_size=MAX_SIZE)
        tree.show(data_property="points", idhidden=False)
        tree.depth()

    def test_thom_points(self):
        thom_points_homogeneous_2 = "[[0, 1]]"
        assert str(thom_points_homogeneous(2)) == thom_points_homogeneous_2

        thom_points_2 = "[(1, 1)]"
        assert str(thom_points(2)) == thom_points_2

        thom_monomial_idea_3 = \
            "[-b[0, 0]**3*b[1, 2], 2*b[0, 0]**2*b[1, 1]**2, -b[0, 0]**3*b[2, 2], b[0, 0]*b[1, 1]*b[2, 2]]"
        assert str(thom_monomial_ideal(3)) == thom_monomial_idea_3

        thom_points_homogeneous_3 = "[[1, 0, 1, 0], [1, 0, 0, 1], [0, 2, 0, 0], [0, 1, 0, 1]]"
        assert str(thom_points_homogeneous(3)) == thom_points_homogeneous_3

        thom_points_homogeneous_4 = "[[2, 1, 0, 1, 0, 0, 0], [2, 0, 2, 0, 0, 0, 0], [2, 0, 0, 0, 2, 0, 0], [1, 2, 1, 0, 0, 0, 0], [2, 0, 1, 0, 0, 1, 0], [2, 0, 1, 0, 0, 0, 1], [2, 0, 0, 1, 1, 0, 0], [2, 0, 0, 0, 1, 0, 1], [0, 4, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 1, 0], [1, 2, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 0, 0], [1, 1, 0, 0, 2, 0, 0], [0, 3, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 1]]"
        assert str(thom_points_homogeneous(4)) == thom_points_homogeneous_4


