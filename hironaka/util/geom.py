import numpy as np


def generate_points(n: int, dimension=3, max_value=50):
    return [[np.random.randint(max_value) for _ in range(dimension)] for _ in range(n)]


def generate_batch_points(n: int, batch_num=1, dimension=3, max_value=50):
    return [[[np.random.randint(max_value) for _ in range(dimension)] for _ in range(n)] for _ in range(batch_num)]
