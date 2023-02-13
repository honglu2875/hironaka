"""
This script search for the trees of certain hardcoded hosts.
"""
from hironaka.core import ListPoints
from hironaka.src import thom_points
import numpy as np
from treelib import Tree

from hironaka.util import search_tree_morin
from hironaka.host import WeakSpivakovsky


def main():
    points = [
        list(
            np.array(thom_points(4)[i])
            + [np.dot([0, 1, 1, 1, 1, 1, 1], np.array(thom_points(4)[i])) - 4, 0, 0, 0, 0, 0, 0]
        )
        for i in range(len(thom_points(4)))
    ]
    print(f"Points: {points}")
    dimension = len(points[0])
    initial_points = ListPoints([points], distinguished_points=[len(points) - 1])

    tree = Tree()
    tree.create_node(0, 0, data=initial_points)
    MAX_SIZE = 10000

    host = WeakSpivakovsky()
    weights = [1, 1, 2, 3, 2, 3, 3]
    search_tree_morin(initial_points, tree, 0, weights, host, max_size=MAX_SIZE)
    tree.show(data_property="points", idhidden=False)


if __name__ == "__main__":
    main()
