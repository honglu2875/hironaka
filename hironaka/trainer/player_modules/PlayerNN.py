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

        self.action_map = torch.zeros((2**self.dimension - self.dimension - 1,), device=device)
        counter = 0
        for i in range(self.dimension):
            for j in range(2**i+1, 2**(i+1)):
                self.action_map[counter] = j
                counter += 1

    def forward(self, x):
        r = self.action_map[torch.randint(2**self.dimension - self.dimension - 1, (x.shape[0],), device=self.device)]
        return torch.nn.functional.one_hot(r.long(), num_classes=2**self.dimension).type(torch.float32)


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
