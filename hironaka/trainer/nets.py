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
            Numbers in `net_arch` represents MLP layers.
            'b' represents a BatchNorm1d layer (attached to the previous layer before activation).
            A dict represents (recursively) a repeated part of the networks:
                'repeat': number of times to repeat.
                'net_arch': a net_arch list described here (recursively).
            activation functions defaults to relu (may change in the future).
    """
    nets = [head]
    network_list = expand_net_list(net_arch)

    last_layer_dim = input_dim
    begin = True
    for item in network_list:
        if isinstance(item, int):
            if not begin:
                nets.append(activation_fn())  # the activation function for the previous layer.
            nets.append(nn.Linear(last_layer_dim, item))
            last_layer_dim = item
            begin = False
        elif isinstance(item, str):
            permissible = ('b',)
            assert item in permissible, f"String item in net_arch must be one of {permissible}."
            if item == 'b':
                nets.append(nn.BatchNorm1d(last_layer_dim))
    if not begin:
        nets.append(activation_fn())  # The activation function for the last layer.

    nets.append(nn.Linear(last_layer_dim, output_dim))

    return nn.Sequential(*nets)


class BaseFeaturesExtractor(nn.Module, abc.ABC):
    """
        A feature extractor (inspired by stable-baseline3).
        Must assign
            self._feature_dim
        Must implement
            forward()
    """

    def __init__(self, dimension: int, max_num_points: int):
        super().__init__()
        self.dimension = dimension
        self.max_num_points = max_num_points

    @property
    def feature_dim(self) -> int:
        return self._get_feature_dim()

    @abc.abstractmethod
    def _get_feature_dim(self):
        pass

    @abc.abstractmethod
    def forward(self, observations) -> torch.Tensor:
        pass


class AgentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, dimension: int, max_num_points: int):
        super().__init__(dimension, max_num_points)
        self.extractors = {'points': nn.Flatten(),
                           'coords': nn.Flatten()}

    def forward(self, observations: TensorDict) -> torch.Tensor:
        tensor_list = []
        for key, extractor in self.extractors.items():
            tensor_list.append(extractor(observations[key]))
        return torch.cat(tensor_list, dim=1)

    def _get_feature_dim(self) -> int:
        return self.dimension * self.max_num_points + self.dimension


class HostFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, dimension: int, max_num_points: int):
        super().__init__(dimension, max_num_points)
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)

    def _get_feature_dim(self) -> int:
        return self.dimension * self.max_num_points




