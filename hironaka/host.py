import abc
from itertools import combinations

import numpy as np

from .abs import Points


class Host(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def select_coord(self, points: Points, debug=False):
        pass


class RandomHost(Host):
    def select_coord(self, points: Points, debug=False):
        dim = points.dim
        return [np.random.choice(list(range(dim)), size=2) for _ in range(points.batchNum)]


class Zeillinger(Host):
    @staticmethod
    def get_char_vector(vt):
        """
            Character vector (L, S),
                L: maximum coordinate - minimum coordinate
                S: sum of the numbers of maximum coordinates and minimum coordinates
            e.g., (1, 1, -1, -1) -> (L=2, S=4)
        """
        mx = max(vt)
        mn = min(vt)
        L = mx - mn
        S = sum([vt[i] == mx for i in range(len(vt))]) + \
            sum([vt[i] == mn for i in range(len(vt))])
        return L, S

    def select_coord(self, points: Points, debug=False):
        assert not points.ended
        dim = points.dim
        result = []
        for b in range(points.batchNum):
            pts = points.get_batch(b)
            if len(pts) <= 1:
                result.append([])
                continue
            pairs = combinations(pts, 2)
            char_vectors = []
            for pair in pairs:
                vector = tuple([pair[0][i] - pair[1][i] for i in range(dim)])
                char_vectors.append((vector, self.get_char_vector(vector)))
            char_vectors.sort(key=(lambda x: x[1]))

            if debug:
                print(char_vectors)
            result.append([np.argmin(char_vectors[0][0]), np.argmax(char_vectors[0][0])])

        return result
