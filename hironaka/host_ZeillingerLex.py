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


class ZeillingerLex(Host):
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

        coords = []
        for char_vector in charVectors:
            if char_vector[1] == charVectors[0][1]:
                r = [np.argmin(char_vector[0]), np.argmax(char_vector[0])]
                if r[0] != r[1]:
                    coords.append(r)
                else:  # if all coordinates are the same, return the first two.
                    coords.append([0, 1])
        coords.sort()
        return coords

A = [(1,0,1,0), (1,0,0,1), (0,2,0,0), (0,1,0,1)]
host = ZeillingerLex()
print(host.selectCoord(A))
B = [[3, 1]]
B.sort()
print(B)

