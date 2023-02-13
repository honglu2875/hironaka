from itertools import zip_longest
from typing import Iterable

import torch

"""
    This file consolidates helper functions directly borrowed from other repos. We consolidate important helper
        functions in this file instead of importing to prevent future unexpected changes. Credits are documented
        and given to the original repos.
"""

"""
    The followings are short snippets directly borrowed from `stable-baselines3`.
"""


def zip_equal(*iterables):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        # TODO: PEP 618, zip new feature allowing strict=True
        for param, target_param in zip_equal(params, target_params):
            target_param.data.mul_(1 - tau)  # pytype: disable=attribute-error
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)  # pytype: disable=attribute-error
