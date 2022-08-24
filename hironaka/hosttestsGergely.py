import abc
import numpy as np
from itertools import combinations
from collections import defaultdict


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
#print(host.selectCoord(A))

class WeakSpivakovsky(Host):
    def selectCoord(self, points, debug=False):
        assert points

        DIM = len(points[0])
        "For each point we store the subset of nonzero coordinates"
        subsets = [set(np.nonzero(point)[0]) for point in points]
        "Find a minimal hitting set, brute-force"
        U = set.union(*subsets)
        result = []
        for i in range(2, len(U)+1):
            combs = combinations(U, i)
            for c in combs:
                if all(set(c) & l for l in subsets):
                    result.append(c)
            if result:
                break
        return result


A = [[0,1,2],[2,1,0]]
host = WeakSpivakovsky()
print(host.selectCoord(A)[0])

