import abc

import numpy as np

from .types import Points


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move(self, points: Points, coords):
        pass


class RandomAgent(Agent):
    def move(self, points: Points, coords):
        actions = [np.random.choice(coord, size=1)[0] for coord in coords]
        points.shift(coords, actions)
        points.get_newton_polytope()
        return actions
