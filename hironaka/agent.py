import abc

import numpy as np

from hironaka.core import Points
from hironaka.policy.Policy import Policy


class Agent(abc.ABC):
    """
        An agent can either modify the points in-place, or just return the action (the chosen coordinate)
    """

    @abc.abstractmethod
    def move(self, points: Points, coords, weights, inplace=True):
        pass


class RandomAgent(Agent):
    def move(self, points: Points, coords, weights=None, inplace=True):
        assert weights is None, "Only support agents without weights."

        actions = [np.random.choice(coord, size=1)[0] if len(coord) > 1 else None for coord in coords]
        if not inplace:
            return actions
        points.shift(coords, actions)
        points.get_newton_polytope()
        return actions


class ChooseFirstAgent(Agent):
    def move(self, points: Points, coords, weights=None, inplace=True):
        assert weights is None, "Only support agents without weights."

        actions = [min(coord) if len(coord) > 1 else None for coord in coords]
        if not inplace:
            return actions
        points.shift(coords, actions)
        points.get_newton_polytope()
        return actions


class PolicyAgent(Agent):
    def __init__(self, policy: Policy):
        self._policy = policy

    def move(self, points: Points, coords: List[List[int]], inplace=True):
        assert len(
            coords) == points.batch_size  # TODO: wrap the move method for the abstract "Agent" with sanity checks?

        features = points.get_features()

        actions = self._policy.predict((features, coords))

        if inplace:
            points.shift(coords, actions)
            points.get_newton_polytope()

        return actions


class AgentMorin(Agent):
    def move(self, points: Points, coords, weights, inplace=True):
        assert points.batch_size == 1, "Temporarily only support batch size 1."  # TODO: generalize!
        weights_unravelled = weights[0]
        coords_unravelled = coords[0]

        if weights_unravelled[coords_unravelled[0]] == weights_unravelled[coords_unravelled[1]]:
            action = np.random.choice(coords_unravelled, size=1)[0]
        else:
            action = coords_unravelled[np.argmin(
                [weights_unravelled[coords_unravelled[0]], weights_unravelled[coords_unravelled[1]]]
            )]
        changing_coordinate = [coord for coord in coords_unravelled if coord != action]
        next_weights = [weights_unravelled[i] if i not in changing_coordinate else 0
                        for i in range(len(weights_unravelled))]

        if inplace:
            points.shift([coords_unravelled], [action])
            points.reposition()
            points.get_newton_polytope()
            weights[:] = next_weights
            return [action]
        return [action], [next_weights]
