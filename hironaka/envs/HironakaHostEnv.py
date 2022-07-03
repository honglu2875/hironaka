from typing import Any, Dict, Optional

import numpy as np
from gym import spaces

from hironaka.envs.HironakaBase import HironakaBase
from hironaka.host import Host


class HironakaHostEnv(HironakaBase):
    """
    This environment fixes a Host. It receives actions from an agent (player B), resolve the action, rewards the
    agent, and returns the new observation
    """

    def __init__(self,
                 host: Host,
                 invalid_move_penalty: int = -1e-3,
                 stop_after_invalid_move: bool = False,
                 config_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs):
        config_kwargs = dict() if config_kwargs is None else config_kwargs
        super().__init__(**{**config_kwargs, **kwargs})
        self.observation_space = spaces.Dict(
            {
                "points": spaces.Box(low=-1.0, high=np.inf, shape=(self.max_number_points, self.dimension),
                                     dtype=np.float32),
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
        if action in self._coords:
            self._points.shift([self._coords], [action])
            self._points.get_newton_polytope()
            stopped = self._points.ended
            reward = 1. if not self._points.ended else 0.
        else:
            stopped = self.stop_after_invalid_move
            reward = self.invalid_move_penalty

        # After action is already taken. Now get coordinates.
        if self._points.ended:
            self._coords = []
        else:
            self._coords = self.host.select_coord(self._points)[0]

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, stopped, info

    def _get_obs(self):
        coords_multi_bin = self._get_coords_multi_bin()
        f = np.array(self._points.get_features()[0])
        f = np.pad(f, ((0, self.max_number_points - len(f)), (0, 0)), mode='constant', constant_values=-1)
        o = {'points': f.astype(np.float32), 'coords': coords_multi_bin}
        return o
