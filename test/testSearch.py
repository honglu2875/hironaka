import unittest

from treelib import Tree

from hironaka.core import ListPoints
from hironaka.host import Zeillinger
from hironaka.src import make_nested_list
from hironaka.util import search_depth, search_tree


class TestSearch(unittest.TestCase):
    """
    def test_search_depth(self):
        points = ListPoints(make_nested_list(
            [[(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
              (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]]
        ))
        pt_lst = [(7, 5, 3, 8), (8, 1, 8, 18), (8, 3, 17, 8),
                  (11, 11, 1, 19), (11, 12, 18, 6), (16, 11, 5, 6)]

        r = search_depth(points, Zeillinger())
        # r = searchDepth_2(pt_lst, Zeillinger())
        print(r)
        assert r == 5552
    """

    def test_search_depth_small(self):
        points = ListPoints(make_nested_list([(0, 1, 0, 1), (0, 2, 0, 0), (1, 0, 0, 1),
                                              (1, 0, 1, 0), (1, 1, 0, 0), (2, 0, 0, 0)]))
        r = search_depth(points, Zeillinger())
        print(r)
        assert r == 6

    def test_search_special_char_vec(self):
        points = ListPoints([[[3, 5, 8], [5, 2, 6], [1, 3, 6], [3, 5, 3], [7, 3, 3], [1, 6, 3], [1, 0, 6], [5, 1, 6],
                              [7, 3, 0], [6, 7, 7]]])
        r = search_depth(points, Zeillinger())
        print(r)

    def test_search(self):
        host = Zeillinger()
        points = ListPoints(make_nested_list([(0, 0, 4), (5, 0, 1), (1, 5, 1), (0, 25, 0)]))

        tree = Tree()
        tree.create_node(0, 0, data=points)

        search_tree(points, tree, 0, host)
        tree.show(data_property="points", idhidden=False)
