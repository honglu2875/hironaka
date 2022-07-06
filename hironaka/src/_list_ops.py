from typing import List

from ._snippets import get_shape


def get_newton_polytope_approx_lst(points: List[List[List[int]]], inplace=True, get_ended=False):
    """
        A simple-minded quick-and-dirty method to obtain an approximation of Newton Polytope disregarding convexity.
        Returns:
            None, ended_each_batch: List[bool] (if inplace)
            new_points:List[List[List[int]]], ended_each_batch: List[bool] (if not inplace)
            None or new_points (if not get_ended)
    """
    batch_num, _, dim = get_shape(points)

    assert batch_num

    new_points = []
    ended_each_batch = []
    for b in range(batch_num):
        counter = 0
        r = []

        points[b] = sorted(points[b], reverse=True)
        for i in range(len(points[b])):
            contained = False
            for j in range(i + 1, len(points[b])):
                if sum([points[b][i][k] < points[b][j][k] for k in range(dim)]) == 0:
                    contained = True
                    break
            if not contained:
                if inplace:
                    points[b][counter] = points[b][i]
                    counter += 1
                else:
                    r.append(points[b][i])

        if inplace:
            if get_ended:
                ended_each_batch.append(counter <= 1)
            for _ in range(len(points[b]) - counter):
                points[b].pop()
        else:
            if get_ended:
                ended_each_batch.append(len(r) <= 1)
            new_points.append(r)

    if inplace:
        return (None, ended_each_batch) if get_ended else None
    else:
        return (new_points, ended_each_batch) if get_ended else new_points


def get_newton_polytope_lst(points: List[List[List[int]]], inplace=True, get_ended=False):
    """
        Get the Newton Polytope for a set of points.
    """
    return get_newton_polytope_approx_lst(points, inplace=inplace, get_ended=get_ended)
    # TODO: perhaps change to a more precise algo to obtain Newton Polytope


def shift_lst(points: List[List[List[int]]], coords: List[List[int]], axis: List[int], inplace=True):
    """
        Shift a set of points according to the rule of Hironaka game.
    """

    batch_num, _, dim = get_shape(points)

    assert batch_num

    if inplace:
        for b in range(batch_num):
            if axis[b] is None:
                continue
            for i in range(len(points[b])):
                points[b][i][axis[b]] = sum([points[b][i][k] for k in coords[b]])
    else:
        result = [[
            [
                sum([x[k] for k in coord]) if ax is not None and i == ax else x[i]
                for i in range(dim)
            ] for x in point
        ] for point, coord, ax in zip(points, coords, axis)]
        return result
