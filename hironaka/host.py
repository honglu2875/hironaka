import abc
from itertools import combinations
from typing import Optional

import numpy as np

from hironaka.core import Points
from hironaka.policy.Policy import Policy


class Host(abc.ABC):
    @abc.abstractmethod
    def select_coord(self, points: Points, debug=False):
        pass


class RandomHost(Host):
    def select_coord(self, points: Points, debug=False):
        dim = points.dimension
        return [np.random.choice(list(range(dim)), size=2) for _ in range(points.batch_size)]


class Zeillinger(Host):
    # noinspection PyPep8Naming
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
        dim = points.dimension
        result = []
        for b in range(points.batch_size):
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

            result.append(self._get_coord(char_vectors))
        return result

    def _get_coord(self, char_vectors):
        r = [np.argmin(char_vectors[0][0]), np.argmax(char_vectors[0][0])]
        if r[0] != r[1]:
            return r
        else:  # if all coordinates are the same, return the first two.
            return [0, 1]


class PolicyHost(Host):
    def __init__(self,
                 policy: Policy,
                 use_discrete_actions_for_host: Optional[bool] = False,
                 **kwargs):
        self._policy = policy
        self.use_discrete_actions_for_host = kwargs.get('use_discrete_actions_for_host', use_discrete_actions_for_host)

    def select_coord(self, points: Points, debug=False):
        features = points.get_features()

        coords = self._policy.predict(features)  # return multi-binary array
        result = []
        for b in range(coords.shape[0]):
            result.append(np.where(coords[b] == 1)[0])
        return result


class ZeillingerLex(Zeillinger):
    def _get_coord(self, char_vectors):  # TODO: efficiency can be improved
        coords = []
        for char_vector in char_vectors:
            if char_vector[1] == char_vectors[0][1]:
                r = [np.argmin(char_vector[0]), np.argmax(char_vector[0])]
                if r[0] != r[1]:
                    coords.append(r)
                else:  # if all coordinates are the same, return the first two.
                    coords.append([0, 1])
        coords.sort()
        return coords[0]


class WeakSpivakovsky(Host):
    def select_coord(self, points: Points, debug=False):
        assert not points.ended
        result = []
        for b in range(points.batch_size):
            pts = points.get_batch(b)
            if len(pts) <= 1:
                result.append([])
                continue
            "For each point we store the subset of nonzero coordinates"
            subsets = [set(np.nonzero(point)[0]) for point in pts]
            "Find a minimal hitting set, brute-force"
            U = set.union(*subsets)
            result = []
            for i in range(1, len(U) + 1):
                combs = combinations(U, i)
                for c in combs:
                    if all(set(c) & l for l in subsets):
                        result.append(c)
                if result:
                    break
        return result


class PolicyHost(Host):
    def __init__(self,
                 policy: Policy,
                 use_discrete_actions_for_host: Optional[bool] = False,
                 **kwargs):
        self._policy = policy
        self.use_discrete_actions_for_host = kwargs.get('use_discrete_actions_for_host', use_discrete_actions_for_host)

    def select_coord(self, points: Points, debug=False):
        features = points.get_features()

        coords = self._policy.predict(features)  # return multi-binary array
        result = []
        for b in range(coords.shape[0]):
            result.append(np.where(coords[b] == 1)[0])
        return result
