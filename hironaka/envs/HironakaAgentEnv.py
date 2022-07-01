from typing import Any, Dict, Optional

import gym
from gym import spaces
import numpy as np

from hironaka.abs import Points
from hironaka.agent import Agent
from hironaka.envs.HironakaBase import HironakaBase
from hironaka.util import generate_points


class HironakaAgentEnv(HironakaBase):
    """
        The environment fixes an *Agent* inside, and is expected to receive actions from a host.
    """

    def __init__(self,
                 agent: Agent,
                 config_kwargs: Optional[Dict[str, Any]] = dict(),
                 **kwargs):
        super().__init__(agent, **{**config_kwargs, **kwargs})

        self.observation_space = \
            spaces.Box(low=-1.0, high=np.inf, shape=(self.max_number_points, self.dimension), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.dimension)

    def _post_reset_update(self):
        pass

    def step(self, action: np.ndarray):
        self.agent.move(self._points, [np.where(action == 1)[0]])

        observation = self._get_obs()
        info = self._get_info()
        self.stopped = self._points.ended
        reward = 1 if self._points.ended else 0

        return observation, reward, self.stopped, info

    def _get_obs(self):
        return self._get_padded_points()
