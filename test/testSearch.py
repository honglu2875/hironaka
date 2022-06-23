import unittest
from treelib import Tree, Node
from typing import List
from dataclasses import dataclass
from hironaka.util import searchDepth, searchTree
from hironaka.host import Zeillinger
from hironaka.types import Points


class TestSearch(unittest.TestCase):
    def test_search_depth(self):
        host = Zeillinger()
        points = Points(
            [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
             (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]
        )
        # points = Points([(0, 1, 0, 1), (0, 2, 0, 0), (1, 0, 0, 1),
        #                (1, 0, 1, 0), (1, 1, 0, 0), (2, 0, 0, 0)])
        r = searchDepth(points, Zeillinger())
        print(r)
        #assert r == 5551

    def test_search(self):

        host = Zeillinger()
        points = [(0, 0, 4), (5, 0, 1), (1, 5, 1), (0, 25, 0)]

        tree = Tree()
        tree.create_node(0, 0, data=Points(points))
        MAX_SIZE = 1000
        searchTree(points, tree, 0, host)
        tree.show(data_property="data", idhidden=False)
