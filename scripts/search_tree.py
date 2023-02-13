"""
This script search for the trees of certain hardcoded hosts.
"""
from hironaka.core import ListPoints
from hironaka.src import thom_points, thom_points_homogeneous
import numpy as np
from treelib import Tree

from hironaka.util import search_tree_morin
from hironaka.host import WeakSpivakovsky, WeakSpivakovskyRandom


def main():
    d = input("Please specify: d=")
    MAX_SIZE = input("Please specify (default MAX_SIZE=10000): MAX_SIZE=")
    MAX_SIZE = int(MAX_SIZE) if MAX_SIZE else 10000

    d = int(d)
    points = thom_points_homogeneous(d)
    print(f"Points: {points}")
    dimension = len(points[0])
    initial_points = ListPoints([points], distinguished_points=[len(points) - 1])

    tree = Tree()
    tree.create_node(0, 0, data=initial_points)

    host = WeakSpivakovskyRandom()
    weights = [1] * dimension
    search_tree_morin(initial_points, tree, 0, weights, host, max_size=MAX_SIZE)
    tree.show(data_property="points", idhidden=False)


if __name__ == "__main__":
    main()
