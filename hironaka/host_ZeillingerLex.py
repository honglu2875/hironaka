import abc
from itertools import combinations

import numpy as np


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


def select_coord(self, points: list, debug=False):
    dim = len(points[-1])
    result = []
    for pts in range(len(points)):
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

        for char_vector in char_vectors:
            r = [np.argmin(char_vector[0]), np.argmax(char_vector[0])]
            if r[0] != r[1]:
                result.append(r)
            else:  # if all coordinates are the same, return the first two.
                result.append([0, 1])

return result