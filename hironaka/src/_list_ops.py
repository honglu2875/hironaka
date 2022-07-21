from typing import List

import numpy as np
from scipy.spatial import ConvexHull

from ._snippets import get_shape


def get_newton_polytope_approx_lst(points: List[List[List[float]]], inplace=True, get_ended=False):
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


def get_newton_polytope_lst(points: List[List[List[float]]], inplace=True):
    """
        Get the Newton Polytope for a set of points.
        TODO: this is perhaps a slow implementation. Must improve!
    """
    assert len(get_shape(points)) == 3

    result = []
    for pts in points:
        pts_np = np.array(pts)
        maximum = np.max(pts_np)
        dimension = pts_np.shape[1]

        # Add points that are very far-away.
        extra = np.full((dimension, dimension), maximum * 2) * (~np.diag([True] * dimension))
        enhanced_pts = np.concatenate((pts_np, extra), axis=0)

        vertices = ConvexHull(enhanced_pts).vertices
        newton_polytope_indices = vertices[vertices < len(pts_np)]
        result.append(pts_np[newton_polytope_indices, :].tolist())

    result = get_newton_polytope_approx_lst(result, inplace=False, get_ended=False)

    if inplace:
        points[:] = result
    else:
        return result


def shift_lst(points: List[List[List[float]]], coords: List[List[int]], axis: List[int], inplace=True):
    """
        Shift a set of points according to the rule of Hironaka game.
    """

    batch_num, _, dim = get_shape(points)
    assert len(coords) == batch_num
    assert len(axis) == batch_num

    assert batch_num

    if inplace:
        for b in range(batch_num):
            if axis[b] not in coords[b]:
                continue
            for i in range(len(points[b])):
                points[b][i][axis[b]] = sum([points[b][i][k] for k in coords[b]])
    else:
        result = [[
            [
                sum([x[k] for k in coord]) if ax in coord and i == ax else x[i]
                for i in range(dim)
            ] for x in point
        ] for point, coord, ax in zip(points, coords, axis)]
        return result


def reposition_lst(points: List[List[List[float]]], inplace=True):
    """
        Reposition all batches of points so that each of them hits all coordinate planes.
    """
    dim = len(points[0][0])
    new_points = []
    for b in range(len(points)):
        min_vector = []
        for i in range(dim):
            min_value = points[b][0][i]
            for point in points[b]:
                min_value = min(point[i], min_value)
            min_vector.append(min_value)

        if not inplace:
            new_points.append([])

        for point in points[b]:
            new_point = None if inplace else []
            for i in range(dim):
                if inplace:
                    point[i] -= min_vector[i]
                else:
                    new_point.append(point[i] - min_vector[i])
            if not inplace:
                new_points[-1].append(new_point)

    if inplace:
        return None
    else:
        return new_points
