import abc
from typing import Optional, Dict, Any, TypeVar

import gym
import numpy as np

from hironaka.abs import Points
from hironaka.src import get_padded_array, get_gym_version_in_float
from hironaka.util import generate_points

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

GYM_VERSION = get_gym_version_in_float()


class HironakaBase(gym.Env, abc.ABC):
    """
        Base gym class for both host and agent in Hironaka polyhedral game.
        As an abstract class, the subclasses need to implement:
            __init__
            _post_reset_update
            step
            _get_obs
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 1}
    reward_range = (-1, 1)

    # Implement these two
    if get_gym_version_in_float() >= 0.22:
        action_space: gym.spaces.Space[ActType]
        observation_space: gym.spaces.Space[ObsType]

    @abc.abstractmethod
    def __init__(self,
                 dimension: Optional[int] = 3,
                 max_number_points: Optional[int] = 10,
                 max_value: Optional[int] = 10,
                 max_efficiency: Optional[bool] = False):
        self.dimension = dimension
        self.max_number_points = max_number_points
        self.max_value = max_value
        self.max_efficiency = max_efficiency

        # States. Will be implemented in reset()
        self._points = None
        self._coords = []

    def reset(self,
              points=None,
              seed=None,
              return_info=False,
              options=None) -> Any:
        if GYM_VERSION >= 0.22:
            super().reset(seed=seed)

        if points is None:
            self._points = Points(
                [generate_points(self.max_number_points, dim=self.dimension, max_value=self.max_value)])
        else:
            self._points = Points(points)

        # This line guarantees that the points underwent Newton Polytope process. But in principle, the input
        # should have gone through it before being passed to reset(). For max efficiency run, we may ignore it.
        if not self.max_efficiency:
            self._points.get_newton_polytope()

        self._post_reset_update()

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    @abc.abstractmethod
    def _post_reset_update(self):
        """
            implement the action taken after the environment is reset. It may include but is not limited to:
                - update self._coord
                - clean up state-related internal attributes (if any)
        """
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    def render(self, mode='ansi'):
        print(self._points)
        print(self._coords)

    def close(self):
        pass

    @abc.abstractmethod
    def _get_obs(self):
        """
            a utility method to be implemented: return the data that matches with the gym environment observation.
        """
        pass

    def _get_padded_points(self) -> np.ndarray:
        """
            a utility method that returns -1 padded point information.
            return: numpy array of shape (self.max_number_points, self.dimension)
        """
        return get_padded_array(self._points.get_features()[0], new_length=self.max_number_points).astype(np.float32)

    def _get_coords_multi_bin(self) -> np.ndarray:
        """
            a utility method that returns a 1d numpy array of shape (self.dimension,). Chosen coordinates are marked
            as 1, and others are 0.
            If the game has ended or the length of self._coord is less than 2, return np.zeros(self.dimension).
            return: numpy array of shape (self.dimension,).
        """
        coords_multi_bin = np.zeros(self.dimension)
        if not self._points.ended and len(self._coords) >= 2:
            coords_multi_bin[self._coords] = 1
        return coords_multi_bin

    @staticmethod
    def _get_info() -> Dict:
        return dict()
