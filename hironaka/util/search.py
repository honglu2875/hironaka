from collections import deque

from hironaka.host import Host
from hironaka.abs import Points


def search_depth(points: Points, host: Host, debug=False):
    """
        Given the host, return the maximal length of the game that an agent can achieve.
    """
    assert points.batchNum == 1  # Only search a single starting case.

    states = deque([(points, 0)])

    max_depth = 0
    while states:
        if debug:
            print(states)
        current, depth = states.pop()
        max_depth = max(max_depth, depth)
        coords = host.select_coord(current, debug=debug)

        # print(current, depth)

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
        search_tree(shifts[-1], tree, node_id, host)
    return tree
