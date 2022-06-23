import abc
import numpy as np
from typing import List, Tuple
from .util import getNewtonPolytope, shift


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move(self, points, restrictAxis):
        pass


class TAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move(self, points, weights, restrictAxis):
        pass


class RandomAgent(Agent):
    def move(self, points, restrictAxis):
        action = np.random.choice(restrictAxis, size=1)[0]
        return (getNewtonPolytope(shift(points, restrictAxis, action)), action)


class AgentThom(TAgent):
    def move(self, points, weights, restrictAxis):
        if weights[restrictAxis[0]] == weights[restrictAxis[1]]:
            action = np.random.choice(restrictAxis, size=1)[0]
        else:
            action = restrictAxis[np.argmin([weights[restrictAxis[0]], weights[restrictAxis[1]]])]
        changingcoordinate = restrictAxis[np.where(restrictAxis != action)]
        newweights = weights
        newweights[changingcoordinate] = weights[action] - weights[changingcoordinate]
        return getNewtonPolytope(shift(points, restrictAxis, action)), action, newweights
