import abc
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

    @abc.abstractmethod
    def __init__(self,
                 mode: str,
                 max_number_points: int,
                 dimension: Optional[int] = 3,
                 normalized: Optional[bool] = True):
        if mode not in self.allowed_modes:
            raise Exception(f'"mode" must be one of {self.allowed_modes}. Got {mode} instead.')
        self.mode = mode
        self.dimension = dimension
        self.max_number_points = max_number_points
        self.normalized = normalized

    @abc.abstractmethod
    def predict(self, features: Any) -> Any:
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
            torch.FloatTensor(
                get_batched_padded_array(features, self.max_number_points)
            ), start_dim=1)  # TODO: maybe add a GPU option for the process?
        if self.normalized:
            return torch.nn.functional.normalize(feature_tensor, p=1.0)
        return feature_tensor

    def input_preprocess_for_agent(self, features: Tuple[List[List[List[int]]], List[List[int]]]):
        """
            An agent will pass on a tuple:
                a 3d batched and nested list for points and a 2d nested list for coordinates.
            :return: a 2d Tensor with padded points and coordinates flattened.
        """
        assert isinstance(features, Tuple)

        feature_tensor = torch.flatten(
            torch.FloatTensor(
                get_batched_padded_array(features[0], self.max_number_points)
            ), start_dim=1)
        if self.normalized:
            feature_tensor = torch.nn.functional.normalize(feature_tensor, p=1.0)
        coord_tensor = torch.FloatTensor(batched_coord_list_to_binary(features[1], self.dimension))

        return torch.cat([feature_tensor, coord_tensor], dim=1)
