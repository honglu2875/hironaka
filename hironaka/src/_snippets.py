import numbers
import sys
from typing import List, Union, Optional

import numpy as np
import torch


def get_shape(o):
    """
        o is supposed to be a nested object consisting of list and tuple.
        It will recursively search for o[0][0]... until it cannot proceed. output len(..) at each level.
            - Example: o = [[1,2,3],[2,3,4]], it will return (2,3)

        If the nested list/tuple objects are not of uniform shape, this function becomes pointless.
        Therefore, being uniform is an assumption before using this snippet.
            - Anti-example: o = [ ([1,2,3],2,3),(2,3,4) ], it will return (2,3,3).
        It is intuitively wrong but this function is not responsible for checking the uniformity.

        For the last axis, it also removes ONE non-number entries at the end of o[0]...[0].
            - Example: o = [[1,2,3,'d'],[2,3,4]], it will return (2,3)
            - Anti-example: o = [[1,2,3,'d','d'],[2,3,4]], it will return (2,4)
        It is again intuitively tricky, but it is only designed to allow for one non-number mark
            and sanity check is not our responsibility.

        Also, if it hits a length-0 object, it will just stop.
    """

    unwrapped = o
    shape = []
    last = None
    while isinstance(unwrapped, (list, tuple)) and unwrapped:
        shape.append(len(unwrapped))
        last = unwrapped[-1]
        unwrapped = unwrapped[0]

    if not isinstance(last, numbers.Number):
        shape[-1] -= 1
    return tuple(shape)


def make_nested_list(o):
    """
        This will make a nested list-like object a nested list.
        It operates in a recursive fashion, and we do not wish to use it in standard class operations.
        ***It's only for testing and scripting purposes.***

        (comment: __dir__() is super super slow!)
    """
    if '__len__' not in o.__dir__() or len(o) == 0:
        return o

    return [make_nested_list(i) for i in o]


def lst_cpy(dest, orig):
    """
        This copies the content of orig to dest. Both need to be list-like and mutable.
        Furthermore, we assume len(dest)>=len(orig).
    """
    for i in range(len(orig)):
        dest[i] = orig[i]
    diff = len(dest) - len(orig)
    for i in range(diff):
        dest.pop()


def get_padded_array(f: Union[List[List[int]], np.ndarray], new_length, constant_value=-1e-8) -> np.ndarray:
    """
        This augments a 2d nested list (axis 1 having uniform length) on axis 0 into given length.
    """
    f_np = np.array(f).astype(float)
    f_np = np.pad(f_np, ((0, new_length - f_np.shape[0]), (0, 0)), mode='constant', constant_values=constant_value)
    return f_np


def get_batched_padded_array(f: List[List[List[int]]], new_length, constant_value=-1e-8) -> np.ndarray:
    """
        This augments a 3d nested list (axis 2 having uniform length, but not axis 1) on axis 1 into a fixed length.
    """
    assert len(get_shape(f)) == 3, f"Got {len(get_shape(f))}."

    result = []
    for f_batch in f:
        result.append(get_padded_array(f_batch, new_length, constant_value=constant_value))
    return np.stack(result, axis=0)


def coord_list_to_binary(f: List[int], dimension):
    """
        This turns a list of coordinates to a numpy array of size (dimension,) consisting of 0/1.
    """
    f_np = np.zeros(dimension)
    f_np[f] = 1
    return f_np


def batched_coord_list_to_binary(f: List[List[int]], dimension):
    """
        This turns a batched list of coordinate to a numpy array of size (batch_num, dimension) of 0/1.
    """
    f_np = np.zeros((len(f), dimension))
    for b in range(len(f)):
        f_np[b][f[b]] = 1
    return f_np


def get_gym_version_in_float():
    """
        This tries to get the gym version in float number, but will not report error.
    """
    r = 0
    try:
        import gym
        r = float(".".join(gym.__version__.split(".")[:2]))
    finally:
        return r


def get_python_version_in_float():
    """
        This tries to get the python version in float number.
    """
    r = 0
    try:
        r = float(".".join(sys.version.split(".")[:2]))
    finally:
        return r


def scale_points(points: List[List[List[int]]], inplace=True):
    """
        Apply L1 normalization to each batch.
    """
    new_points = None if inplace else [[] for _ in range(len(points))]
    for b in range(len(points)):
        m = 0

        for point in points[b]:
            m = max(m, max(point))

        if m == 0:  # All vectors are zero, nothing to scale (in fact, game is over.)
            continue

        for point in points[b]:
            if inplace:
                point[:] = [x / m for x in point]
            else:
                new_points[b].append([x / m for x in point])

    if not inplace:
        return new_points


def encode_action(binary: np.ndarray):
    assert len(binary.shape) == 1, f"Got {len(binary.shape)}."
    return np.sum(2 ** np.arange(len(binary)) * np.array(binary))


def decode_action(code: int, dimension: int):
    code = int(code)
    assert isinstance(dimension, int), f"Got {type(dimension)}."

    decoded = []
    while code:
        decoded.append(code % 2)
        code = code // 2
    result = np.zeros(dimension)
    result[:len(decoded)] = np.array(decoded)
    return result


def mask_encoded_action(dimension: int):
    assert isinstance(dimension, int), f"Got {type(dimension)}."

    result = np.ones(2 ** dimension)
    result[0] = 0
    for i in range(dimension):
        result[1 << i] = 0

    return result


def generate_points(n: int, dimension=3, max_value=50):
    return [[np.random.randint(max_value) for _ in range(dimension)] for _ in range(n)]


def generate_batch_points(n: int, batch_num=1, dimension=3, max_value=50):
    return [[[np.random.randint(max_value) for _ in range(dimension)] for _ in range(n)] for _ in range(batch_num)]


def remove_repeated(points: torch.Tensor, padding_value: Optional[float] = -1.):
    """
        A crucial tensor operation to remove extra repeating points and leave the first one.
        Always inplace.
    """
    batch_size, max_num_points, dimension = points.shape
    device = points.device

    # get the difference matrix for the second axis
    difference = points.unsqueeze(2).repeat(1, 1, max_num_points, 1) - \
                 points.unsqueeze(1).repeat(1, max_num_points, 1, 1)

    upper_tri = ~torch.triu(torch.ones(max_num_points, max_num_points).type(torch.bool), diagonal=0) \
        .unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    repeated_points = ((difference.eq(0).all(3) & upper_tri).any(2)).unsqueeze(2).repeat(1, 1, dimension)
    # Always modify inplace
    r = points * ~repeated_points + torch.full(points.shape, padding_value).to(device) * repeated_points
    points[:, :, :] = r

    return None
