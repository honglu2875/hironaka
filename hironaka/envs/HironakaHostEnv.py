from typing import Any, Dict, Optional

import numpy as np
from gym import spaces

from hironaka.host import Host
from hironaka.envs.HironakaBase import HironakaBase

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
        super().__init__(host, **{**config_kwargs, **kwargs})
        self.observation_space = \
            spaces.Box(low=-1.0, high=np.inf, shape=(self.max_number_points, self.dimension), dtype=np.float32)
        self.action_space = spaces.MultiBinary(self.dimension)

        self.host = host

    def _post_reset_update(self):
        self.step(action = None)

    def step(self, action):
        if action in self._coords:
            self._points.shift([self._coords], [action])
            self._points.get_newton_polytope()
            self.stopped = self._points.ended
            reward = 1. if not self._points.ended else 0.
        else:
            self.stopped = True #self.stop_after_invalid_move
            reward = 10e-3 #self.invalid_move_penalty

        if self._points.ended:
            self._coords = []
        else:
            self._coords = self.host.select_coord(self._points)[0]

        coords_multi_bin = np.zeros(self.dimension)
        coords_multi_bin[self._coords] = 1

        observation = self._get_obs()
        info = self._get_info()
        self.stopped = self._points.ended

        return observation, reward, self.stopped, info

    def _get_obs(self):
        coords_multi_bin = self._get_coords_multi_bin()
        f = np.array(self._points.get_features()[0])
        f = np.pad(f, ((0, self.max_number_points - len(f)), (0, 0)), mode='constant', constant_values=-1)
        o = {'points': f.astype(np.float32), 'coords': coords_multi_bin}
        return o


# import gym
# import numpy as np
# from gym import spaces
#
# from hironaka.abs import Points
# from hironaka.util import generate_points
#
#

# class HironakaHostEnv(gym.Env):  # fix a host inside, receive agent moves from outside
#     metadata = {"render_modes": ["ansi"], "render_fps": 1}
#
#     def __init__(self, host, dim=3, max_pt=10, max_value=10, invalid_move_penalty=-1e-3, stop_after_invalid_move=False):
#         self.dim = dim
#         self.max_pt = max_pt
#         self.max_value = max_value
#         self.host = host
#         self.invalid_move_penalty = invalid_move_penalty
#         self.stop_after_invalid_move = stop_after_invalid_move
#         self.stopped = False
#
#         self.observation_space = spaces.Dict(
#             {
#                 "points": spaces.Box(low=-1.0, high=np.inf, shape=(self.max_pt, self.dim), dtype=np.float32),
#                 "coords": spaces.MultiBinary(self.dim)
#             }
#         )
#
#         self.action_space = spaces.Discrete(self.dim)
#         self._points = None
#         self._coords = None
#
#         self.window = None
#         self.clock = None
#
#     def reset(self, seed=None, return_info=False, options=None, points=None):
#         super().reset(seed=seed)
#
#         if points is None:
#             self._points = Points([generate_points(self.max_pt, dim=self.dim, max_number=self.max_value)])
#         else:
#             self._points = Points(points)
#
#         self._points.get_newton_polytope()
#
#         # I forgot to add the initial coordinate choice when reset.
#         # There should be a uniformed way that separates reset and taking the first step.
#         if self._points.ended:
#             self._coords = None
#         else:
#             self._coords = self.host.select_coord(self._points)[0]
#         coords_multi_bin = np.zeros(self.dim)
#         coords_multi_bin[self._coords] = 1
#
#         observation = self._get_obs(coords_multi_bin=coords_multi_bin)
#
#         info = self._get_info()
#         return (observation, info) if return_info else observation
#
#     def step(self, action):
#
#         # The host takes an action from the agent and resolve it
#         # If the action is valid (inside the coordnates chose by the host), then resolve the game, update the situation, award the agent, pick a subset of coordinates, and give a new observation to the agent.
#         # If the action is not valid, let the game stop and give the agent a very negative feedback.
#
#         if action in self._coords:
#             self._points.shift([self._coords], [action])
#             self._points.get_newton_polytope()
#             # Since you always take newton polytope after shift, maybe it would be better to embed points.get_newton_polytope() into points.shift()?
#             self.stopped = self._points.ended
#             reward = 1. if not self._points.ended else 0.
#         else:
#             self.stopped = self.stop_after_invalid_move
#             reward = self.invalid_move_penalty
#
#         # here we conjour host.select_coord to select a subset of coordinates by a method given by the host.
#         if self._points.ended:
#             self._coords = []
#         else:
#             self._coords = self.host.select_coord(self._points)[0]
#         coords_multi_bin = np.zeros(self.dim)
#         coords_multi_bin[self._coords] = 1
#
#         observation = self._get_obs(coords_multi_bin=coords_multi_bin)
#         info = self._get_info()
#
#         return observation, reward, self.stopped, info
#
#     def render(self, mode='ansi'):
#         print(self._points)
#         print(self._coords)
#
#     def close(self):
#         pass
#
#     def _get_obs(self, coords_multi_bin):
#         # I add coords_multi_bin parameter to pass the choice into creating the new observation.
#         f = np.array(self._points.get_features()[0])
#         f = np.pad(f, ((0, self.max_pt - len(f)), (0, 0)), mode='constant', constant_values=-1)
#         o = {'points': f.astype(np.float32), 'coords': coords_multi_bin}
#         return o
#
#     def _get_info(self):
#         return dict()
