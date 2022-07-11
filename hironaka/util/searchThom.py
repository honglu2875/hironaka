from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from hironaka.util import shift, getNewtonPolytope


def searchTreeMorin(points, tree, curr_node, curr_weights, host, MAX_SIZE):
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
        print('Contr')
        return
    shifts = []
    coords = host.selectCoord(points)
    dim = len(points[0])
    for action in coords:
        if curr_weights[action] > np.amin([curr_weights[coords]]):
            return
        else:
            changingcoordinate = coords[np.where(coords != action)[0][0]]
            curr_weights[changingcoordinate] = 0
            ShiftedState = shift(points, coords, action)
            newState = list(map(tuple, ShiftedState - np.amin(ShiftedState, axis=0)))
            newNewtonPolytope = getNewtonPolytope(newState)
            if newState[-1] in newNewtonPolytope:
                A = newNewtonPolytope.pop(newNewtonPolytope.index(newState[-1]))
                newNewtonPolytope.append(A)
                shifts.append(newNewtonPolytope)
            else:
                print('No contr')
                return
        node_id = tree.size()
        tree.create_node(node_id, node_id, parent=curr_node, data=Points(shifts[-1]))
        searchTreeMorin(shifts[-1], tree, node_id, curr_weights, host, MAX_SIZE)
    return tree