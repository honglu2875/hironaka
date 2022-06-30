import numpy as np


def generate_points(n: int, dim=3, max_number=50):
    return [[np.random.randint(max_number) for _ in range(dim)] for _ in range(n)]


def generate_batch_points(n: int, batch_num=1, dim=3, max_number=50):
    return [[[np.random.randint(max_number) for _ in range(dim)] for _ in range(n)] for _ in range(batch_num)]
