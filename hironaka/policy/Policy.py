import abc
import logging
from typing import Any, Optional, List, Tuple

import torch

from hironaka.src import get_batched_padded_array, batched_coord_list_to_binary


class Policy(abc.ABC):
    """
        An abstract policy wrapper that makes decisions based on observations.
        It covers both agent and host scenarios (the difference shows up in input preprocessing).
        When writing host- or agent-only classes,
            please override 'allowed_modes' before calling 'super().__init__(...)'

        Need to implement:
            __init__
            predict
    """
    allowed_modes = ['host', 'agent']
    logger = None

    @abc.abstractmethod
    def __init__(self,
                 mode: str,
                 max_num_points: int,
                 padding_value: Optional[float] = -1e-8,
                 dimension: Optional[int] = 3,
                 device_key: Optional[str] = 'cpu',
                 **kwargs):
        if self.logger is None:
            self.logger = logging.getLogger(__class__.__name__)

        if mode not in self.allowed_modes:
            raise Exception(f'"mode" must be one of {self.allowed_modes}. Got {mode} instead.')
        self.mode = mode
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.padding_value = padding_value
        self.device = torch.device(device_key)

    @abc.abstractmethod
    def predict(self, features: Any, debug: Optional[bool] = False) -> Any:
        """
            Return the prediction based on the observed features.

            this method by default should be to handle both host and agent input using
            self.input_preprocess_for_host() and self.input_preprocess_for_agent()
        """
        pass

    def input_preprocess_for_host(self, features: List[List[List[int]]]):
        """
            A host will pass on a 3d batched and nested list.
            :return: a 2d Tensor with padded points flattened. Possibly normalized depending on config.
        """
        feature_tensor = torch.flatten(
            torch.tensor(
                get_batched_padded_array(features, self.max_num_points, constant_value=self.padding_value),
                dtype=torch.float32, device=self.device), start_dim=1)

        return feature_tensor

    def input_preprocess_for_agent(self, features: Tuple[List[List[List[int]]], List[List[int]]]):
        """
            An agent will pass on a tuple:
                a 3d batched and nested list for points and a 2d nested list for coordinates.
            :return: a 2d Tensor with padded points and coordinates flattened.
        """
        assert isinstance(features, Tuple)

        feature_tensor = torch.flatten(
            torch.tensor(
                get_batched_padded_array(features[0], self.max_num_points, constant_value=self.padding_value),
                dtype=torch.float32, device=self.device), start_dim=1)

        coord_tensor = torch.tensor(batched_coord_list_to_binary(features[1], self.dimension), dtype=torch.float32)

        return torch.cat([feature_tensor, coord_tensor], dim=1)
