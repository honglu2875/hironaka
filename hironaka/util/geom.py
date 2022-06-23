import numpy as np


def generatePoints(n: int, dim=3, MAX_ORDER=50):
    return [tuple([np.random.randint(MAX_ORDER) for _ in range(dim)]) for _ in range(n)]
