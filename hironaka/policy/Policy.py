import abc
import logging
from typing import Any, Optional, List, Tuple, Union, Type

import torch

from hironaka.src import get_batched_padded_array, batched_coord_list_to_binary

PointsDataTypes = Union[torch.Tensor, List[List[List[Any]]]]


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
                 max_num_points: int,
                 padding_value: Optional[float] = -1.0,
                 dimension: Optional[int] = 3,
                 dtype: Optional[Type] = torch.float32,
                 **kwargs):
        self.logger = logging.getLogger(__class__.__name__)

        if mode not in self.allowed_modes:
            raise ValueError(f'"mode" must be one of {self.allowed_modes}. Got {mode} instead.')
        self.mode = mode
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.padding_value = padding_value
        self.dtype = dtype

    @abc.abstractmethod
    def predict(self, features: PointsDataTypes, debug: Optional[bool] = False) -> PointsDataTypes:
        """
        Return the prediction based on the observed features.

        this method by default should be to handle both host and agent input using
        self.input_preprocess_for_host() and self.input_preprocess_for_agent()
        """
        pass

    def input_preprocess_for_host(self, features: PointsDataTypes) -> PointsDataTypes:
        """
        A host will pass on either a nested list or a tensor.
        Return:
            a 2d Tensor with padded points flattened. Possibly normalized depending on config.
        """
        if isinstance(features, List):
            feature_tensor = torch.flatten(
                torch.tensor(
                    get_batched_padded_array(features, self.max_num_points, constant_value=self.padding_value),
                    dtype=self.dtype, device=self.device), start_dim=1)
        elif isinstance(features, torch.Tensor):
            feature_tensor = torch.flatten(features, start_dim=1)
        else:
            raise TypeError(f"Unsupported input type. Got {type(features)}.")

        return feature_tensor

    def input_preprocess_for_agent(self, features: Tuple[PointsDataTypes, Union[torch.Tensor, List[List[int]]]]) -> \
            PointsDataTypes:
        """
        An agent will pass on a tuple:
            a nested list or tensor for points and a 2d nested list or tensor for coordinates.
            Recall we follow the convention for coordinates: [[1, 2]] <--equivalent--> torch.tensor[[0, 1, 1]].
        Return:
            a 2d Tensor with padded points and coordinates flattened.
        """
        assert isinstance(features, Tuple)

        feature_tensor = torch.flatten(
            torch.tensor(
                get_batched_padded_array(features[0], self.max_num_points, constant_value=self.padding_value),
                dtype=self.dtype, device=self.device), start_dim=1)

        if isinstance(features[1], torch.Tensor):
            coord_tensor = features[1]
        else:
            coord_tensor = torch.tensor(batched_coord_list_to_binary(features[1], self.dimension),
                                        device=self.device, dtype=self.dtype)

        return torch.cat([feature_tensor, coord_tensor], dim=1)
