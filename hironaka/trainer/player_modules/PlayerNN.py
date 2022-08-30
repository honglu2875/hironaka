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
        return torch.nn.functional.one_hot(x['coords'].argmax(1), num_classes=self.dimension).type(torch.float32)


class ChooseLastAgentModule(nn.Module, DummyModule):
    def __init__(self, dimension: int, max_num_points: int, device: torch.device):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.device = device

    def forward(self, x):
        _DEVICE = x['coords'].device
        augmented = (x['coords'].type(torch.float32) + torch.arange(self.dimension,
                                                                    dtype=torch.float32,
                                                                    device=_DEVICE) * 1e-4)
        return torch.nn.functional.one_hot(augmented.argmax(1), num_classes=self.dimension).type(torch.float32)


class RandomAgentModule(nn.Module, DummyModule):
    def __init__(self, dimension: int, max_num_points: int, device: torch.device):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.device = device

    def forward(self, x):
        coords = x['coords']
        r = torch.rand((coords.shape[0], self.dimension), device=self.device) * coords
        return torch.nn.functional.one_hot(r.argmax(1), num_classes=self.dimension).type(torch.float32)


class RandomHostModule(nn.Module, DummyModule):
    def __init__(self, dimension: int, max_num_points: int, device: torch.device):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.device = device
        self.output_dim = 2 ** self.dimension - self.dimension - 1

    def forward(self, x):
        r = torch.randint(self.output_dim, (x.shape[0],), device=self.device)
        return torch.nn.functional.one_hot(r.long(), num_classes=self.output_dim).type(torch.float32)


class AllCoordHostModule(nn.Module, DummyModule):
    def __init__(self, dimension: int, max_num_points: int, device: torch.device):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.device = device
        self.output_dim = 2 ** self.dimension - self.dimension - 1

    def forward(self, x):
        r = torch.zeros((x.shape[0], self.output_dim), device=self.device, dtype=torch.float32)
        r[:, -1] = 1.
        return r
