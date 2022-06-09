import abc
import numpy as np
from itertools import combinations


class Host(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def selectCoord(self, points, debug=False):
        pass


class RandomHost(Host):
    def selectCoord(self, points, debug=False):
        assert points

        DIM = len(points[0])
        return list(np.random.choice(list(range(DIM)), size=2))


class Zeillinger(Host):
    def getCharVector(self, vt):
        mx = max(vt)
        mn = min(vt)
        L = mx - mn
        S = sum([vt[i] == mx for i in range(len(vt))]) + \
            sum([vt[i] == mn for i in range(len(vt))])
        return (L, S)

    def selectCoord(self, points, debug=False):
        assert points

        DIM = len(points[0])
        pairs = combinations(points, 2)
        charVectors = []
        for pair in pairs:
            vector = tuple([pair[0][i] - pair[1][i] for i in range(DIM)])
            charVectors.append((vector, self.getCharVector(vector)))
        charVectors.sort(key=(lambda x: x[1]))

        if debug:
            print(charVectors)

        return [np.argmin(charVectors[0][0]), np.argmax(charVectors[0][0])]
