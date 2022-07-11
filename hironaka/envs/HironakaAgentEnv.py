from typing import Any, Dict, Optional

import numpy as np
from gym import spaces

from hironaka.agent import Agent
from hironaka.envs.HironakaBase import HironakaBase


class HironakaAgentEnv(HironakaBase):
    """
        The environment fixes an Agent inside, and is expected to receive actions from a host.
    """

    def __init__(self,
                 agent: Agent,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        config_kwargs = dict() if config_kwargs is None else config_kwargs
        super().__init__(**{**config_kwargs, **kwargs})

        self.agent = agent

        self.observation_space = self.point_observation_space
        self.action_space = spaces.MultiBinary(self.dimension)

    def _post_reset_update(self):
        pass

    def step(self, action: np.ndarray):
        super().step(action)  # reset self.current_step

        stopped = False
        reward = 0

        number_of_points_before = self._points.get_num_points()[0]
        self.last_action_taken = self.agent.move(self._points, [np.where(action == 1)[0]])

        stopped |= self._points.ended

        # Check whether the step and the maximal value exceeds the pre-set thresholds
        self.exceed_threshold = self._points.exceed_threshold()
        if self.stop_at_threshold:
            if self.current_step >= self.step_threshold or self.exceed_threshold:
                stopped = True
                if self.fixed_penalty_crossing_threshold is None:
                    reward -= self.step_threshold  # Massive penalty based on steps
                else:
                    reward += self.fixed_penalty_crossing_threshold  # Fixed penalty. Can be set to 0

        # Rescale the points
        if self.scale_observation:
            self._points.rescale()

        observation = self._get_obs()
        info = self._get_info()
        if self.reward_based_on_point_reduction:
            reward += number_of_points_before - self._points.get_num_points()[0]
        reward += 1 if self._points.ended else 0

        return observation, reward, stopped, info

    def _get_obs(self):
        return self._get_padded_points()
