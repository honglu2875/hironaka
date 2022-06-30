import abc

import numpy as np

from .abs import Points


class Agent(metaclass=abc.ABCMeta):
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
