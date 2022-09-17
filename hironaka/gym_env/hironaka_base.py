import abc
from typing import Any, Dict, Optional, TypeVar

import gym
import numpy as np
from gym import spaces

from hironaka.core import ListPoints
from hironaka.src import generate_points, get_gym_version_in_float, get_padded_array

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

    metadata = {"render_modes": ["ansi"]}

    # Override these two
    config_key_for_points = ["value_threshold"]
    config_key_for_generate_points = ["dimension", "max_value"]

    if GYM_VERSION >= 0.22:
        action_space: gym.spaces.Space[ActType]
        observation_space: gym.spaces.Space[ObsType]

    @abc.abstractmethod
    def __init__(
            self,
            dimension: Optional[int] = 3,
            max_num_points: Optional[int] = 10,
            max_value: Optional[int] = 10,
            padding_value: Optional[float] = -1.0,
            value_threshold: Optional[float] = None,
            step_threshold: Optional[int] = 1000,
            fixed_penalty_crossing_threshold: Optional[int] = None,
            stop_at_threshold: Optional[bool] = True,
            improve_efficiency: Optional[bool] = False,
            scale_observation: Optional[bool] = True,
            reward_based_on_point_reduction: Optional[bool] = False,
            **kwargs
    ):
        self.dimension = dimension
        self.max_num_points = max_num_points
        self.max_value = max_value
        self.padding_value = padding_value
        self.value_threshold = value_threshold
        self.step_threshold = step_threshold
        self.fixed_penalty_crossing_threshold = fixed_penalty_crossing_threshold
        self.stop_at_threshold = stop_at_threshold
        self.improve_efficiency = improve_efficiency
        self.scale_observation = scale_observation
        self.reward_based_on_point_reduction = reward_based_on_point_reduction

        # Use self.point_observation_space in the definition of observations
        if self.scale_observation:
            self.point_observation_space = spaces.Box(
                low=-1.0, high=np.inf, shape=(self.max_num_points, self.dimension), dtype=np.float32
            )
        else:
            self.point_observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.max_num_points, self.dimension), dtype=np.float32
            )

        # Configs to pass down to other functions
        self.config_for_points = {key: getattr(self, key) for key in self.config_key_for_points}
        self.config_for_generate_points = {key: getattr(self, key) for key in self.config_key_for_generate_points}

        # States. Will be implemented in reset()
        self._points = None
        self._coords = []
        self.current_step = 0
        self.exceed_threshold = False
        self.last_action_taken = None

    def reset(self, points=None, seed=None, return_info=False, options=None) -> Any:
        if GYM_VERSION >= 0.22:
            super().reset(seed=seed)

        if points is None:
            self._points = ListPoints(
                [generate_points(self.max_num_points, **self.config_for_generate_points)],
                value_threshold=self.value_threshold
            )
        else:
            self._points = ListPoints(points, **self.config_for_points)

        self._points.get_newton_polytope()
        if self.scale_observation:
            self._points.rescale(inplace=True)

        self.current_step = 0
        self.exceed_threshold = False
        self.last_action_taken = None

        # This line guarantees that the points underwent Newton Polytope process. But in principle, the input
        # should have gone through it before being passed to reset(). For max efficiency run, we may ignore it.
        if not self.improve_efficiency:
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
        self.current_step += 1

    def render(self, mode="ansi"):
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

    def _get_padded_points(self, constant_value: Optional[float] = -1.0) -> np.ndarray:
        """
        a utility method that returns -1 padded point information.
        return: numpy array of shape (self.max_num_points, self.dimension)
        """
        return get_padded_array(
            self._points.get_features()[0], new_length=self.max_num_points, constant_value=constant_value
        ).astype(np.float32)

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

    def _get_info(self) -> Dict:
        if self.improve_efficiency:
            return {}
        else:
            return {
                "step_threshold": self.step_threshold,
                "current_step": self.current_step,
                "exceed_threshold": self.exceed_threshold,
                "last_action_taken": self.last_action_taken,
            }
