import abc
from typing import List

import numpy as np

from .core import Points
from .policy import Policy


class Agent(abc.ABC):
    """
        An agent can either modify the points in-place, or just return the action (the chosen coordinate)
    """

    @abc.abstractmethod
    def move(self, points: Points, coords, inplace=True):
        pass


class RandomAgent(Agent):
    def move(self, points: Points, coords, inplace=True):
        actions = [np.random.choice(coord, size=1)[0] if len(coord) > 1 else None for coord in coords]
        if not inplace:
            return actions
        points.shift(coords, actions)
        points.get_newton_polytope()
        return actions


class ChooseFirstAgent(Agent):
    def move(self, points: Points, coords, inplace=True):
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
        assert len(coords) == points.batch_size  # TODO: wrap the move method for the abstract "Agent" with sanity checks?

        features = points.get_features()

        actions = self._policy.predict((features, coords))

        if inplace:
            points.shift(coords, actions)
            points.get_newton_polytope()

        return actions
