import abc
from typing import List, Any, Type, Dict, Union

import torch
from torch import nn

TensorDict = Dict[Union[str, int], torch.Tensor]


def expand_net_list(net_arch: List[Any]) -> List[Any]:
    """
        Utility function for `create_mlp`. Recursively expand the dict objects in the list.
    """
    expanded = []
    for item in net_arch:
        if isinstance(item, dict):
            for key in ['repeat', 'net_arch']:
                assert key in item, f"'{key}' must be a key in the dict."
            assert isinstance(item['net_arch'], list), f"'net_arch' must be a list. Got {type(item['net_arch'])}."
            expanded += expand_net_list(item['net_arch']) * item['repeat']
        else:
            expanded.append(item)

    return expanded


def create_mlp(head: nn.Module, net_arch: List[Any], input_dim: int, output_dim: int,
               activation_fn: Type[nn.Module] = nn.ReLU) -> nn.Module:
    """
        A basic MLP network will be constructed according to `head` followed by `net_arch`.
        `head` must output tensors with dim (-1, input_dim).
        `net_arch` does not need to specify input and output dimension as they are already in the argument.
            Numbers in `net_arch` represents network layers.
            'b' represents a BatchNorm1d layer (attached to the previous layer before activation).
            A dict represents (recursively) a repeated part of the networks:
                'repeat': number of times to repeat.
                'net_arch': a net_arch list described here (recursively).
            activation functions defaults to relu (may change in the future).
    """
    nets = [head, activation_fn()]
    network_list = expand_net_list(net_arch)

    last_layer_dim = input_dim
    last_layer_type = None
    for item in network_list:
        if isinstance(item, int):
            nets += [nn.Linear(last_layer_dim, item), activation_fn()]
            last_layer_dim = item
            last_layer_type = 'l'
        elif isinstance(item, str):
            permissible = ('b', 'r')
            assert item[0] in permissible, f"First letter in net_arch must be one of {permissible}."
            if item[0] == 'b':  # Batch norm
                if last_layer_type == 'b':
                    continue
                elif last_layer_type == 'l' or last_layer_type == 'r' or last_layer_type is None:
                    nets.pop()  # Remove the last activation function
                    nets += [nn.BatchNorm1d(last_layer_dim), activation_fn()]
                else:
                    nets.append(nn.BatchNorm1d(last_layer_dim))
                last_layer_type = 'b'
            elif item[0] == 'r':  # Residual block
                in_channels, out_channels = last_layer_dim, int(item[1:])
                assert out_channels > 0, f"Invalid description of residual block. Got {item}"
                nets += [make_residual(in_channels, out_channels), activation_fn()]
                last_layer_dim = out_channels
                last_layer_type = 'r'

    nets.pop()  # remove the last activation function
    nets.append(nn.Linear(last_layer_dim, output_dim))

    return nn.Sequential(*nets)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        return out


def make_residual(in_channels: int, out_channels: int) -> ResidualBlock:
    if in_channels == out_channels:
        return ResidualBlock(in_channels, out_channels)
    else:
        return ResidualBlock(in_channels, out_channels,
                             nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels)))


class BaseFeaturesExtractor(nn.Module, abc.ABC):
    """
        A feature extractor (inspired by stable-baselines3).
        Must assign
            self._feature_dim
        Must implement
            forward()
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @property
    def feature_dim(self) -> int:
        return self.input_dim

    @abc.abstractmethod
    def forward(self, observations) -> torch.Tensor:
        pass


class AgentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.extractors = {'points': nn.Flatten(),
                           'coords': nn.Flatten()}

    def forward(self, observations: TensorDict) -> torch.Tensor:
        tensor_list = []
        for key, extractor in self.extractors.items():
            tensor_list.append(extractor(observations[key]))
        return torch.cat(tensor_list, dim=1)


class HostFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, input_dim: int):
        super().__init__(input_dim)
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)

