import abc

import torch
from torch import nn

"""
    The classes in this file implement those simple hosts/agents as dummy nn.Modules without trainable parameters.
"""


class DummyModule(abc.ABC):
    pass


class ChooseFirstAgentModule(nn.Module, DummyModule):
    def __init__(self, dimension: int, max_num_points: int, device: torch.device):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.device = device

    def forward(self, x):
        r = torch.zeros(x['coords'].shape, device=self.device, dtype=torch.float32)
        r.scatter_(1, x['coords'].argmax(1).unsqueeze(1), 1.)
        return r


class AllCoordHostModule(nn.Module, DummyModule):
    def __init__(self, dimension: int, max_num_points: int, device: torch.device):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.device = device

    def forward(self, x):
        r = torch.zeros((x.shape[0], 2 ** self.dimension), device=self.device, dtype=torch.float32)
        r[:, 2 ** self.dimension - 1] = 1.
        return r
