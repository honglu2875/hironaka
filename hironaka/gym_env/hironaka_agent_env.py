from typing import Any, Dict, Optional, Union

import numpy as np
from gym import spaces

from hironaka.agent import Agent
from hironaka.src import decode_action
from .hironaka_base import HironakaBase


class HironakaAgentEnv(HironakaBase):
    """
    The environment fixes an Agent inside, and is expected to receive actions from a host.
    """

    def __init__(
            self,
            agent: Agent,
            use_discrete_actions_for_host: Optional[bool] = False,
            compressed_host_output: Optional[bool] = True,
            config_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        config = kwargs if config_kwargs is None else {**kwargs, **config_kwargs}
        super().__init__(**config)

        self.agent = agent
        self.use_discrete_actions_for_host = config.get("use_discrete_actions_for_host", use_discrete_actions_for_host)
        self.compressed_host_output = compressed_host_output

        self.observation_space = self.point_observation_space
        if self.use_discrete_actions_for_host:
            if self.compressed_host_output:
                self.action_space = spaces.Discrete(2 ** self.dimension - self.dimension - 1)
            else:
                self.action_space = spaces.Discrete(2 ** self.dimension)
        else:
            self.action_space = spaces.MultiBinary(self.dimension)

    def _post_reset_update(self):
        pass

    def step(self, action: Union[np.ndarray, int]):
        # `action` for HironakaAgentEnv: a binary vector of 0/1 according to chosen coordinates by the host.

        super().step(action)  # update self.current_step
        # decode the action if discrete
        if self.use_discrete_actions_for_host:
            action = decode_action(action, self.dimension)

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
        return self._get_padded_points(constant_value=self.padding_value)
