import abc

import numpy as np

from hironaka.abs import Points
from hironaka.agent import Agent
from hironaka.src import shift_lst, get_newton_polytope_lst


class TAgent(abc.ABC):
    @abc.abstractmethod
    def move(self, points: Points, weights, coords):
        pass


class AgentThom(TAgent):
    def move(self, points: Points, weights, coords):
        assert points.batch_size == 1  # Temporarily only support batch size 1. TODO: generalize!
        weights = weights[0]
        coords = coords[0]

        if weights[coords[0]] == weights[coords[1]]:
            action = np.random.choice(coords, size=1)[0]
        else:
            action = coords[np.argmin([weights[coords[0]], weights[coords[1]]])]
        changing_coordinate = [coord for coord in coords if coord != action]
        next_weights = [weights[i] if i not in changing_coordinate else 0 for i in range(len(weights))]
        points.shift([coords], [action])
        points.reposition()
        points.get_newton_polytope()
        return points, action, [next_weights]


class AgentMorin(TAgent):
    def move(self, points, weights, coords):
        assert points.batch_size == 1  # Temporarily only support batch size 1. TODO: generalize!
        weights = weights[0]
        coords = coords[0]

        if weights[coords[0]] == weights[coords[1]]:
            action = np.random.choice(coords, size=1)[0]
        else:
            action = coords[np.argmin([weights[coords[0]], weights[coords[1]]])]
        changing_coordinate = [coord for coord in coords if coord != action]
        next_weights = [weights[i] if i not in changing_coordinate else 0 for i in range(len(weights))]
        points.shift([coords], [action])
        points.reposition()
        points.get_newton_polytope()

        ## NEED WORK
        if points.distinguished_points[0] is not None:
            return points, action, [next_weights]
        else:
            return False