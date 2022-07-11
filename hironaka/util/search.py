from typing import Tuple, List
from collections import deque
from dataclasses import dataclass
from hironaka.util.geom import getNewtonPolytope, shift


def searchDepth(points, host, debug=False):
    """
        Fixing the host, return the maximal length of the game that an agent can achieve.
    """
    if len(points) == 1:
        return 0

    states = deque([(points, 0)])

    maxDepth = 0
    while states:
        current, depth = states.pop()
        maxDepth = max(maxDepth, depth)
        coords = host.selectCoord(current, debug=debug)
        for i in coords:
            next = getNewtonPolytope(shift(current, coords, i))
            if len(next) > 1:
                states.append((next, depth+1))

    return maxDepth


def searchTree(points, tree, curr_node, host, MAX_SIZE):
    """
        Perform a full tree search and store the full result in a Tree object.
    """

    @dataclass
    class Points:
        """
            a wrapper of a set of points.
        """
        data: List[Tuple[int]]

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

