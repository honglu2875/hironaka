from collections import deque, namedtuple

import numpy as np

from hironaka.core import ListPoints
from hironaka.host import Host


def search_depth(points: ListPoints, host: Host, debug=False):
    """
        Given the host, return the maximal length of the game that an agent can achieve.
    """
    assert points.batch_size == 1  # Only search a single starting case.

    states = deque([(points, 0)])

    max_depth = 0
    while states:
        if debug:
            print(states)
        current, depth = states.pop()
        max_depth = max(max_depth, depth)
        coords = host.select_coord(current, debug=debug)

        for i in coords[0]:
            nxt = current.copy()
            nxt.shift(coords, [i])
            nxt.get_newton_polytope()
            if not nxt.ended:
                states.append((nxt, depth + 1))

    return max_depth + 1


def search_tree(points, tree, curr_node, host, max_size=100):
    """
        Perform a full tree search and store the full result in a Tree object.
    """

    if points.ended or tree.size() > max_size:
        return

    shifts = []
    coords = host.select_coord(points)
    for i in coords[0]:
        shifts.append(
            points.shift(coords, [i], inplace=False).get_newton_polytope(inplace=False)
        )
        node_id = tree.size()
        tree.create_node(node_id, node_id, parent=curr_node, data=shifts[-1])
        search_tree(shifts[-1], tree, node_id, host, max_size=max_size)
    return tree


def search_tree_morin(points: ListPoints, tree, curr_node, curr_weights, host, max_size=100):
    """
        Perform a full tree search and store the full result in a Tree object.
    """
    Node = namedtuple('Node', ['points'])

    if isinstance(curr_weights, np.ndarray):
        curr_weights = curr_weights.tolist()

    if points.ended or tree.size() > max_size:
        if tree.size() > max_size:
            node_id = tree.size()
            tree.create_node(node_id, node_id, parent=curr_node, data=Node('...more...'))
        return tree
    coords = host.select_coord(points)[0]
    # print('Coords:',coords, 'Weights:', curr_weights)

    for action in coords:
        if curr_weights[action] > min([curr_weights[i] for i in coords]):
            continue

        changing_coordinate = [coord for coord in coords if coord != action]
        next_weights = [curr_weights[i] if i not in changing_coordinate else curr_weights[i] - curr_weights[action] for
                        i in range(len(curr_weights))]

        new_points = points.shift([coords], [action], inplace=False)

        new_points.reposition()
        new_points.get_newton_polytope()

        if new_points.distinguished_points[0] is None:
            node_id = tree.size()
            tree.create_node(node_id, node_id, parent=curr_node, data=Node('No contribution'))
            continue
        node_id = tree.size()
        tree.create_node(node_id, node_id, parent=curr_node,
                         data=Node(str(new_points) + f", {new_points.distinguished_points}"))
        search_tree_morin(new_points, tree, node_id, next_weights, host, max_size)
    return tree
