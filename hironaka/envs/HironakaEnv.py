import gym
from gym import spaces
import numpy as np

from hironaka.abs import Points
from hironaka.util import generate_points


class HironakaEnv(gym.Env):  # fix host, agent moves
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, host, dim=3, max_pt=10, batch_num=1):
        self.dim = dim
        self.max_pt = max_pt
        self.batch_num = batch_num
        self.host = host
        self.stopped = False

        self.observation_space = spaces.Dict(
            {
                "points": spaces.Box(low=-1.0, high=np.inf, shape=(batch_num, self.max_pt, self.dim), dtype=np.float32),
                "coords": spaces.MultiBinary((self.batch_num, self.dim))
            }
        )

        self.action_space = spaces.MultiDiscrete([self.batch_num, self.dim])
        self._points = None
        self._coords = None

        self.window = None
        self.clock = None

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._points = Points([generate_points(self.max_pt)])

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        action = list(action)
        assert len(action) == self.batch_num
        for i in range(self.batch_num):
            assert action[i] in self._coords[i]

        self._points.shift(self._coords, action)
        self._points.get_newton_polytope()

        observation = self._get_obs()
        info = self._get_info()
        self.stopped = self._points.ended
        reward = 1 if not self.stopped else 0

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

        f = self._points.get_features()
        if self._points.ended:
            self._coords = None
        else:
            self._coords = self.host.select_coord(self._points)
        coords_multi_bin = np.zeros((self.batch_num, self.dim))

        for i in range(self.batch_num):  # TODO: vectorize??
            coords_multi_bin[i, self._coords] = 1

        o = {'points': np.array([regularize(pts) for pts in f]).astype(np.float32), 'coords': coords_multi_bin}
        return o

    def _get_info(self):
        return self._points.ended
