from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from hironaka.abs import Points
from hironaka.src import shift_lst, get_newton_polytope_lst, reposition_lst


def search_tree_morin(points: Points, tree, curr_node, curr_weights, host, max_size=100):
    """
        Perform a full tree search and store the full result in a Tree object.
    """
    NoCont = namedtuple('NoCont', ['points'])

    if isinstance(curr_weights, np.ndarray):
        curr_weights = curr_weights.tolist()

    if points.ended or tree.size() > max_size:
        if tree.size() > max_size:
            node_id = tree.size()
            tree.create_node(node_id, node_id, parent=curr_node, data=NoCont('...more...'))
        return tree
    shifts = []
    coords = host.select_coord(points)[0]

    print(f"Weight {points}, {curr_weights}, {coords}")

    for action in coords:
        if curr_weights[action] > min([curr_weights[i] for i in coords]):
            continue

        changing_coordinate = [coord for coord in coords if coord != action]
        next_weights = [curr_weights[i] if i not in changing_coordinate else 0 for i in range(len(curr_weights))]

        new_points = points.shift([coords], [action], inplace=False)
        new_points.reposition()
        new_points.get_newton_polytope()

        if new_points.distinguished_points[0] is not None:
            shifts.append(new_points)
        else:
            node_id = tree.size()
            tree.create_node(node_id, node_id, parent=curr_node, data=NoCont('No contribution'))
            return tree
        node_id = tree.size()
        tree.create_node(node_id, node_id, parent=curr_node, data=shifts[-1])

        search_tree_morin(shifts[-1], tree, node_id, next_weights, host, max_size)
    return tree
