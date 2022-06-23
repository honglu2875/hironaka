import abc
import numpy as np
from itertools import combinations
from .types import Points


class Host(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def selectCoord(self, points: Points, debug=False):
        pass


class RandomHost(Host):
    def selectCoord(self, points: Points, debug=False):

        dim = points.dim
        return [np.random.choice(list(range(dim)), size=2) for _ in range(points.batchNum)]


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

    def selectCoord(self, points: Points, debug=False):

        dim = points.dim
        result = []
        for b in range(points.batchNum):
            pts = points.getBatch(b)
            pairs = combinations(pts, 2)
            charVectors = []
            for pair in pairs:
                vector = tuple([pair[0][i] - pair[1][i] for i in range(dim)])
                charVectors.append((vector, self.getCharVector(vector)))
            charVectors.sort(key=(lambda x: x[1]))

            if debug:
                print(charVectors)

            result.append([np.argmin(charVectors[0][0]), np.argmax(charVectors[0][0])])

        return result
