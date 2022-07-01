import gym
from gym import spaces
import numpy as np

from hironaka.abs import Points
from hironaka.util import generate_points


class HironakaAgentEnv(gym.Env):  # fix an agent inside, receive host moves from outside
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, agent, dim=3, max_pt=10, max_value=10):
        self.dim = dim
        self.max_pt = max_pt
        self.max_value = max_value
        self.agent = agent
        self.stopped = False

        self.observation_space = spaces.Box(low=-1.0, high=np.inf, shape=(self.max_pt, self.dim), dtype=np.float32)

        self.action_space = spaces.MultiBinary(self.dim)
        self._points = None
        self._coords = None

        self.window = None
        self.clock = None

    def reset(self, seed=None, return_info=False, options=None, points=None):
        super().reset(seed=seed)

        if points is None:
            self._points = Points([generate_points(self.max_pt, dim=self.dim, max_number=self.max_value)])
        else:
            self._points = Points(points)
        self._points.get_newton_polytope()

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        self.agent.move(self._points, [np.where(action == 1)[0]])

        observation = self._get_obs()
        info = self._get_info()
        self.stopped = self._points.ended
        reward = 1 if self._points.ended else 0

        return observation, reward, self.stopped, info

    def render(self, mode='ansi'):
        print(self._points)
        print(self._coords)

    def close(self):
        pass

    def _get_obs(self):
        f = np.array(self._points.get_features()[0])
        f = np.pad(f, ((0, self.max_pt - len(f)), (0, 0)), mode='constant', constant_values=-1)
        o = f.astype(np.float32)
        return o

    def _get_info(self):
        return dict()
