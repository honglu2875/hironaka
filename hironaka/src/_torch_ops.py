from typing import Optional, Union, List

import torch

from hironaka.src import batched_coord_list_to_binary


def get_newton_polytope_approx_torch(points: torch.Tensor,
                                     inplace: Optional[bool] = True,
                                     padding_value: Optional[float] = -1.):
    assert len(points.shape) == 3
    batch_size, max_num_points, dimension = points.shape

    available_points = (points >= 0)

    # get a filter matrix
    filter_matrix = available_points.unsqueeze(2).repeat(1, 1, max_num_points, 1) & \
                    available_points.unsqueeze(1).repeat(1, max_num_points, 1, 1)
    # get the difference matrix for the point-indexing axis
    difference = points.unsqueeze(2).repeat(1, 1, max_num_points, 1) - \
                 points.unsqueeze(1).repeat(1, max_num_points, 1, 1)

    # filter the diagonal
    diag_filter = ~torch.diag(torch.ones(max_num_points)).type(torch.BoolTensor)
    diag_filter = torch.reshape(diag_filter, (1, max_num_points, max_num_points, 1)) \
        .repeat(batch_size, 1, 1, dimension)

    # get the points that need to be removed
    points_to_remove = ((difference >= 0) & diag_filter & filter_matrix).all(3).any(2)
    points_to_remove = points_to_remove.unsqueeze(2).repeat(1, 1, dimension)

    r = points * ~points_to_remove + torch.full(points.shape, padding_value) * points_to_remove

    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r


def get_newton_polytope_torch(points: torch.Tensor,
                              inplace: Optional[bool] = True,
                              padding_value: Optional[float] = -1.):
    return get_newton_polytope_approx_torch(points, inplace=inplace, padding_value=padding_value)


def shift_torch(points: torch.Tensor,
                coord: Union[torch.Tensor, List[List[int]]],
                axis: Union[torch.Tensor, List[int]],
                inplace: Optional[bool] = True,
                padding_value: Optional[float] = -1.):
    """
        note:
            If coord is a list, it is assumed to be lists of chosen coordinates.
            If coord is a tensor, it is assumed to be batches of multi-binary data according to chosen coordinates.
            They are not equivalent. E.g.,
                dimension is 3, coord = [[1,2]] is equivalent to coord = Tensor([[0,1,1]])
    """
    _TENSOR_TYPE = torch.float32

    assert len(points.shape) == 3
    batch_size, max_num_points, dimension = points.shape

    if isinstance(coord, list):
        coord = torch.FloatTensor(batched_coord_list_to_binary(coord, dimension))
    elif not isinstance(coord, torch.Tensor):
        raise Exception(f"unsupported input type for coord. Got {type(coord)}.")
    if isinstance(axis, (list, torch.Tensor)):
        axis = torch.FloatTensor(axis)
    else:
        raise Exception(f"unsupported input type for axis. Got {type(axis)},")

    # Initial sanity check
    coord = coord.type(_TENSOR_TYPE)
    assert coord.shape == (batch_size, dimension)
    assert axis.shape == (batch_size,)
    assert torch.all(torch.all(points.ge(0), 2).eq(torch.any(points.ge(0), 2)))

    # Get filter
    available_points = (points >= 0)

    # Turn each axis label into (0, 0, ... 1, ..., 0) where only the given location is 1.
    src = torch.full((batch_size, 1), 1.).type(_TENSOR_TYPE)
    index = axis.unsqueeze(1).type(torch.int64)  # index must be int64
    axis_binary = torch.scatter(torch.zeros(batch_size, dimension), 1, index, src).type(_TENSOR_TYPE)
    assert torch.all(axis_binary - coord).le(0)

    # Generate transition matrices
    trans_matrix = axis_binary.unsqueeze(2) * coord.unsqueeze(1) + \
                   torch.diag(torch.ones(dimension)).repeat(batch_size, 1, 1) - \
                   axis_binary.unsqueeze(2) * axis_binary.unsqueeze(1)
    trans_matrix = trans_matrix.unsqueeze(1).repeat(1, max_num_points, 1, 1)

    transformed_points = torch.matmul(trans_matrix, points.unsqueeze(3)).squeeze(3)
    r = (transformed_points * available_points) + torch.full(points.shape, padding_value) * ~available_points

    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r


def reposition_torch(points: torch.Tensor,
                     inplace: Optional[bool] = True,
                     padding_value: Optional[float] = -1.):
    available_points = points.ge(0)
    maximum = torch.max(points)

    preprocessed = points * available_points + torch.full(points.shape, maximum + 1) * ~available_points
    coordinate_minimum = torch.amin(preprocessed, 1)
    unfiltered_result = points - coordinate_minimum.unsqueeze(1).repeat(1, points.shape[1], 1)
    r = unfiltered_result * available_points + torch.full(points.shape, padding_value) * ~available_points
    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r


def rescale_torch(points: torch.Tensor, inplace: Optional[bool] = True, padding_value: Optional[float] = -1.):
    available_points = points.ge(0)
    r = points * available_points / torch.reshape(torch.amax(points, (1, 2)), (-1, 1, 1)) + \
        padding_value * ~available_points
    if inplace:
        points[:, :, :] = r
    else:
        return r