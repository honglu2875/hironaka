import unittest

import gym
import numpy as np
from gym.envs.registration import register

from hironaka.agent import RandomAgent
from hironaka.host import Zeillinger
from hironaka.abs import Points

register(
    id='hironaka/Hironaka',
    entry_point='hironaka.envs:HironakaEnv',
    max_episode_steps=10000,
)


class TestEnv(unittest.TestCase):
    def test_run_env(self):
        env = gym.make('hironaka/Hironaka', host=Zeillinger())
        agent = RandomAgent()

        o = env.reset()
        env.render()
        while not env.stopped:
            action = agent.move(Points(np.expand_dims(o['points'], axis=0)), [np.where(o['coords'] == 1)[0]], inplace=False)
            o, r, stopped, info = env.step(action[0])
            print(f"Reward: {r}")
            env.render()
