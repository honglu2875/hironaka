from typing import Any, Dict, Optional

import numpy as np
from gym import spaces

from hironaka.gym_env.HironakaBase import HironakaBase
from hironaka.host import Host


class HironakaHostEnv(HironakaBase):
    """
    This environment fixes a Host. It receives actions from an agent (player B), resolve the action, rewards the
    agent, and returns the new observation
    """

    def __init__(self,
                 host: Host,
                 invalid_move_penalty: Optional[int] = -1e-3,
                 stop_after_invalid_move: Optional[bool] = False,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        config_kwargs = dict() if config_kwargs is None else config_kwargs
        super().__init__(**{**config_kwargs, **kwargs})

        self.observation_space = spaces.Dict(
            {
                "points": self.point_observation_space,
                "coords": spaces.MultiBinary(self.dimension)
            }
        )

        self.action_space = spaces.Discrete(self.dimension)

        self.host = host
        self.invalid_move_penalty = invalid_move_penalty
        self.stop_after_invalid_move = stop_after_invalid_move

    def _post_reset_update(self):
        self.step(action=None)

    def step(self, action):
        super().step(action)  # update self.current_step

        stopped = False
        reward = 0
        if action in self._coords:
            self._points.shift([self._coords], [action])
            self._points.get_newton_polytope()
            reward += 1. if not self._points.ended else 0.
        else:
            stopped |= self.stop_after_invalid_move
            reward += self.invalid_move_penalty

        stopped |= self._points.ended

        # Check whether the maximal value exceeds the self.value_threshold
        self.exceed_threshold = self._points.exceed_threshold()
        stopped |= self.exceed_threshold

        # After an action is already taken, now get coordinates.
        if stopped:
            self._coords = []
        else:
            self._coords = self.host.select_coord(self._points)[0]

        # Rescale the points
        if self.scale_observation:
            self._points.rescale()

        self.last_action_taken = self._coords
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, stopped, info

    def _get_obs(self):
        coords_multi_bin = self._get_coords_multi_bin()
        f = np.array(self._points.get_features()[0])
        f = np.pad(f,
                   ((0, self.max_number_points - len(f)),
                    (0, 0)),
                   mode='constant',
                   constant_values=self.padding_value)
        o = {'points': f.astype(np.float32), 'coords': coords_multi_bin}
        return o
