from typing import List, Optional, Union

import torch

from hironaka.src import batched_coord_list_to_binary, remove_repeated


def get_newton_polytope_approx_torch(
    points: torch.Tensor, inplace: Optional[bool] = True, padding_value: Optional[float] = -1.0
):
    assert len(points.shape) == 3
    remove_repeated(points, padding_value=padding_value)

    batch_size, max_num_points, dimension = points.shape
    device = points.device
    available_points = points.ge(0)

    # get a filter matrix
    filter_matrix = available_points.unsqueeze(2).repeat(1, 1, max_num_points, 1) & available_points.unsqueeze(1).repeat(
        1, max_num_points, 1, 1
    )
    # get the difference matrix for the second axis
    difference = points.unsqueeze(2).repeat(1, 1, max_num_points, 1) - points.unsqueeze(1).repeat(1, max_num_points, 1, 1)

    # filter the diagonal
    diag_filter = ~torch.diag(torch.ones(max_num_points, device=device)).type(torch.bool)
    diag_filter = torch.reshape(diag_filter, (1, max_num_points, max_num_points, 1)).repeat(batch_size, 1, 1, dimension)

    # get the points that need to be removed
    points_to_remove = (difference.ge(0) & diag_filter & filter_matrix).all(3).any(2)
    points_to_remove = points_to_remove.unsqueeze(2).repeat(1, 1, dimension)

    r = points * ~points_to_remove + torch.full(points.shape, padding_value, device=device) * points_to_remove

    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r


def get_newton_polytope_torch(points: torch.Tensor, inplace: Optional[bool] = True, padding_value: Optional[float] = -1.0):
    return get_newton_polytope_approx_torch(points, inplace=inplace, padding_value=padding_value)


def shift_torch(
    points: torch.Tensor,
    coord: Union[torch.Tensor, List[List[int]]],
    axis: Union[torch.Tensor, List[int]],
    inplace: Optional[bool] = True,
    padding_value: Optional[float] = -1.0,
    ignore_ended_games: Optional[bool] = True,
):
    """
    note:
        If coord is a list, it is assumed to be lists of chosen coordinates.
        If coord is a tensor, it is assumed to be batches of multi-binary data according to chosen coordinates.
        They are not equivalent. E.g.,
            dimension is 3, coord = [[1,2]] is equivalent to coord = Tensor([[0,1,1]])
    """
    _TENSOR_TYPE = points.dtype
    device = points.device

    assert len(points.shape) == 3
    batch_size, max_num_points, dimension = points.shape

    if isinstance(coord, list):
        coord = torch.tensor(batched_coord_list_to_binary(coord, dimension), device=device)
    elif not isinstance(coord, torch.Tensor):
        raise Exception(f"unsupported input type for coord. Got {type(coord)}.")
    if isinstance(axis, list):
        axis = torch.tensor(axis, device=device)
    elif not isinstance(axis, torch.Tensor):
        raise Exception(f"unsupported input type for axis. Got {type(axis)},")

    # Initial sanity check
    coord = coord.type(_TENSOR_TYPE)
    assert coord.shape == (batch_size, dimension)
    assert axis.shape == (batch_size,)
    assert torch.all(torch.all(points.ge(0), 2).eq(torch.any(points.ge(0), 2)))

    # Get filter
    available_points = points.ge(0)

    # Turn each axis label into (0, 0, ... 1, ..., 0) where only the given location is 1.
    src = torch.full((batch_size, 1), 1.0, device=device, dtype=_TENSOR_TYPE)
    index = axis.unsqueeze(1).type(torch.int64)  # index must be int64
    axis_binary = torch.scatter(torch.zeros((batch_size, dimension), device=device, dtype=_TENSOR_TYPE), 1, index, src)
    # For each element in the batch, record a mask for valid actions.
    valid_actions_mask = torch.all((axis_binary - coord).le(0), dim=1)
    axis_binary *= valid_actions_mask.reshape(-1, 1)  # Apply the valid action mask.
    if ignore_ended_games:
        axis_binary *= torch.sum(points[:, :, 0].ge(0), 1).ge(2).reshape(-1, 1)  # Filter by whether game ended

    # Generate transition matrices
    trans_matrix = (
        axis_binary.unsqueeze(2) * coord.unsqueeze(1)
        + torch.diag(torch.ones(dimension, device=device, dtype=_TENSOR_TYPE)).repeat(batch_size, 1, 1)
        - axis_binary.unsqueeze(2) * axis_binary.unsqueeze(1)
    )
    trans_matrix = trans_matrix.unsqueeze(1).repeat(1, max_num_points, 1, 1)

    transformed_points = torch.matmul(trans_matrix, points.unsqueeze(3)).squeeze(3)
    r = (transformed_points * available_points) + torch.full(points.shape, padding_value, device=device) * ~available_points

    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r


def reposition_torch(points: torch.Tensor, inplace: Optional[bool] = True, padding_value: Optional[float] = -1.0):
    available_points = points.ge(0)
    maximum = torch.max(points)
    _TENSOR_TYPE = points.dtype
    device = points.device

    preprocessed = (
        points * available_points
        + torch.full(points.shape, maximum + 1, device=device, dtype=_TENSOR_TYPE) * ~available_points
    )
    coordinate_minimum = torch.amin(preprocessed, 1)
    unfiltered_result = points - coordinate_minimum.unsqueeze(1).repeat(1, points.shape[1], 1)
    r = (
        unfiltered_result * available_points
        + torch.full(points.shape, padding_value, device=device, dtype=_TENSOR_TYPE) * ~available_points
    )
    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r


def rescale_torch(points: torch.Tensor, inplace: Optional[bool] = True, padding_value: Optional[float] = -1.0):
    available_points = points.ge(0)
    max_val = torch.amax(points, (1, 2))
    max_val = max_val + max_val.eq(0) * 1
    max_val = torch.reshape(max_val, (-1, 1, 1))
    r = points * available_points / max_val + padding_value * ~available_points
    if inplace:
        points[:, :, :] = r
        return None
    else:
        return r
