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

        dim = len(points[0])
        return list(np.random.choice(list(range(dim)), size=2))


class Zeillinger(Host):
    def getCharVector(self, vt):
        '''
            Character vector (L, S),
                L: maximum coordinate - minimum coordinate
                S: sum of the numbers of maximum coordinates and minimum coordinates
            e.g., (1, 1, -1, -1) -> (L=2, S=4)
        '''
        mx = max(vt)
        mn = min(vt)
        L = mx - mn
        S = sum([vt[i] == mx for i in range(len(vt))]) + \
            sum([vt[i] == mn for i in range(len(vt))])
        return (L, S)

    def selectCoord(self, points, debug=False):
        assert points

        dim = len(points[0])
        pairs = combinations(points, 2)
        charVectors = []
        for pair in pairs:
            vector = tuple([pair[0][i] - pair[1][i] for i in range(dim)])
            charVectors.append((vector, self.getCharVector(vector)))
        charVectors.sort(key=(lambda x: x[1]))

        if debug:
            print(charVectors)

        return [np.argmin(charVectors[0][0]), np.argmax(charVectors[0][0])]
