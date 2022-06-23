from treelib import Node, Tree
from typing import Tuple, List
from collections import deque
from hironaka.types import Points
from hironaka.host import Host


def searchDepth(points: Points, host: Host, debug=False):
    """
        Fixing the host, return the maximal length of the game that an agent can achieve.
    """
    assert points.batchNum == 1  # Only search a single starting case.

    states = deque([(points, 0)])

    maxDepth = 0
    while states:
        if debug:
            print(states)
        current, depth = states.pop()
        maxDepth = max(maxDepth, depth)
        coords = host.selectCoord(current, debug=debug)

        # print(depth)
        # print(current)

        for i in coords[0]:
            next = current.shift(coords, [i], inplace=False)
            next.getNewtonPolytope()
            if not next.ended:
                states.append((next, depth+1))

    return maxDepth


def searchTree(points, tree, curr_node, host, MAX_SIZE=100):
    """
        Perform a full tree search and store the full result in a Tree object.
    """

    if len(points) == 1 or tree.size() > MAX_SIZE:
        return

    shifts = []
    coords = host.selectCoord(points)
    for i in coords:
        shifts.append(
            getNewtonPolytope(
                shift(points, coords, i)
            )
        )
        node_id = tree.size()
        tree.create_node(node_id, node_id, parent=curr_node, data=Points(shifts[-1]))
        searchTree(shifts[-1], tree, node_id, host)
    return tree
