import gym
from gym import spaces
import numpy as np

from hironaka.abs import Points
from hironaka.util import generate_points


class HironakaEnv(gym.Env):  # fix host, agent moves
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, host, dim=3, max_pt=10, bad_move_penalty=-1e-3):
        self.dim = dim
        self.max_pt = max_pt
        self.host = host
        self.bad_move_penalty = bad_move_penalty
        self.stopped = False

        self.observation_space = spaces.Dict(
            {
                "points": spaces.Box(low=-1.0, high=np.inf, shape=(self.max_pt, self.dim), dtype=np.float32),
                "coords": spaces.MultiBinary((self.dim,))
            }
        )

        self.action_space = spaces.Discrete(self.dim)
        self._points = None
        self._coords = None

        self.window = None
        self.clock = None

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._points = Points([generate_points(self.max_pt, dim=self.dim)])

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        if action in self._coords:
            self._points.shift([self._coords], [action])
            self._points.get_newton_polytope()
            reward = 1. if not self._points.ended else 0.
        else:
            reward = self.bad_move_penalty
        observation = self._get_obs()
        info = self._get_info()
        self.stopped = self._points.ended

        return observation, reward, self.stopped, info

    def render(self, mode='ansi'):
        print(self._points)
        print(self._coords)

    def close(self):
        pass

    def _get_obs(self):
        def regularize(pts):
            assert len(pts) > 0
            assert len(pts) <= self.max_pt
            assert len(pts[0]) == self.dim
            diff = self.max_pt - len(pts)
            return np.array(pts + [[-1] * self.dim for _ in range(diff)])

        f = self._points.get_features()[0]
        if self._points.ended:
            self._coords = None
        else:
            self._coords = self.host.select_coord(self._points)[0]
        coords_multi_bin = np.zeros(self.dim)
        coords_multi_bin[self._coords] = 1

        o = {'points': np.array(regularize(f)).astype(np.float32), 'coords': coords_multi_bin}
        return o

    def _get_info(self):
        return self._points.ended
