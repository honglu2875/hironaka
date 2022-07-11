import abc
import numpy as np
from hironaka.util import getNewtonPolytope, shift


class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move(self, points, restrictAxis):
        pass

class RandomAgent(Agent):
    def move(self, points, restrictAxis):
        action = np.random.choice(restrictAxis, size=1)[0]
        return (getNewtonPolytope(shift(points, restrictAxis, action)), action)


