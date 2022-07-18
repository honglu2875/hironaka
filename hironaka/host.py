import abc
from itertools import combinations
from typing import Optional

import numpy as np

from .core import Points
from .policy import Policy


class Host(abc.ABC):
    @abc.abstractmethod
    def select_coord(self, points: Points, debug=False):
        pass


class RandomHost(Host):
    def select_coord(self, points: Points, debug=False):
        dim = points.dimension
        return [np.random.choice(list(range(dim)), size=2, replace=False) for _ in range(points.batch_size)]


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

            r = [np.argmin(char_vectors[0][0]), np.argmax(char_vectors[0][0])]
            if r[0] != r[1]:
                result.append(r)
            else:  # if all coordinates are the same, return the first two.
                result.append([0, 1])

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
