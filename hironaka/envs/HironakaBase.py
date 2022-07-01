import abc
from typing import Optional, Dict, Any

import gym
import numpy as np

from hironaka.abs import Points
from hironaka.agent import Agent
from hironaka.util import generate_points


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

    @abc.abstractmethod
    def __init__(self,
                 agent: Agent,
                 dimension: Optional[int] = 3,
                 max_number_points: Optional[int] = 10,
                 max_value: Optional[int] = 10,
                 max_efficiency: Optional[bool] = False):
        self.dimension = dimension
        self.max_number_points = max_number_points
        self.max_value = max_value
        self.max_efficiency = max_efficiency
        self.agent = agent
        self.stopped = False

        # States. Will be implemented in reset()
        self._points = None
        self._coords = None

        # Implement these two
        self.observation_space = None
        self.action_space = None

    def reset(self,
              points=None,
              seed=None,
              return_info=False,
              options=None) -> Any:
        super().reset(seed=seed)

        if points is None:
            self._points = Points(
                [generate_points(self.max_number_points, dim=self.dimension, max_number=self.max_value)])
        else:
            self._points = Points(points)

        # This line guarantees that the points underwent Newton Polytope process. But in principle, the input
        # should have gone through it before being passed to reset(). For max efficiency run, we may ignore it.
        if not self.max_efficiency:
            self._points.get_newton_polytope()
        self.stopped = False

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
        f = np.array(self._points.get_features()[0])
        f = np.pad(f, ((0, self.max_number_points - len(f)), (0, 0)), mode='constant', constant_values=-1)
        o = f.astype(np.float32)
        return o

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
