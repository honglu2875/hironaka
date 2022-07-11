import abc

import numpy as np

from hironaka.util import shift, getNewtonPolytope


class TAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move(self, points, weights, restrictAxis):
        pass


class MAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move(self, points, weights, restrictAxis):
        pass


class AgentThom(TAgent):
    def move(self, points, weights, restrictAxis):
        dim = len(points[0])
        #print(restrictAxis, restrictAxis[0],restrictAxis[1])
        if weights[restrictAxis[0]] == weights[restrictAxis[1]]:
            action = np.random.choice(restrictAxis, size=1)[0]
            #print(action)
        else:
            action = restrictAxis[np.argmin([weights[restrictAxis[0]], weights[restrictAxis[1]]])]
        changingcoordinate = restrictAxis[np.where(restrictAxis != action)[0][0]]
        weights[changingcoordinate] = 0
        newState = shift(points, restrictAxis, action)
        #print(newState, list(map(tuple,newState-np.amin(newState, axis=0))))
        return (getNewtonPolytope(list(map(tuple,newState-np.amin(newState, axis=0)))), action, weights)


class AgentMorin(MAgent):
    def move(self, points, weights, restrictAxis):
        dim = len(points[0])
        #print(restrictAxis, restrictAxis[0],restrictAxis[1])
        if weights[restrictAxis[0]] == weights[restrictAxis[1]]:
            action = np.random.choice(restrictAxis, size=1)[0]
            #print(action)
        else:
            action = restrictAxis[np.argmin([weights[restrictAxis[0]], weights[restrictAxis[1]]])]
        changingcoordinate = restrictAxis[np.where(restrictAxis != action)[0][0]]
        weights[changingcoordinate] = 0
        ShiftedState = shift(points, restrictAxis, action)
        newState = list(map(tuple,ShiftedState - np.amin(ShiftedState, axis=0)))
        newNewtonPolytope = getNewtonPolytope(newState)
        #print(newState[-1])
        #print(newNewtonPolytope)
        #print(newNewtonPolytope.index(newState[-1]))
        #print(newNewtonPolytope.pop(newNewtonPolytope.index(newState[-1])))
        #print(newNewtonPolytope)
        if newState[-1] in newNewtonPolytope:
            A = newNewtonPolytope.pop(newNewtonPolytope.index(newState[-1]))
            newNewtonPolytope.append(A)
            return (newNewtonPolytope, action, weights)
        else:
            return False