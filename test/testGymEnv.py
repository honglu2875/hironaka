import unittest

import gym
import numpy as np
from gym.envs.registration import register

from hironaka.agent import RandomAgent
from hironaka.host import Zeillinger, RandomHost
from hironaka.abs import Points

register(
    id='hironaka/HironakaHost-v0',
    entry_point='hironaka.envs:HironakaHostEnv',
    max_episode_steps=10000,
)

register(
    id='hironaka/HironakaAgent-v0',
    entry_point='hironaka.envs:HironakaAgentEnv',
    max_episode_steps=10000,
)


class TestEnv(unittest.TestCase):
    def test_host_run_env(self):
        env = gym.make('hironaka/HironakaHost-v0', host=Zeillinger())
        agent = RandomAgent()

        o = env.reset()
        env.render()
        while not env.stopped:
            action = agent.move(Points(np.expand_dims(o['points'], axis=0)), [np.where(o['coords'] == 1)[0]],
                                inplace=False)
            o, r, stopped, info = env.step(action[0])
            print(f"Reward: {r}")
            env.render()

    def test_agent_run_env(self):
        dim = 4
        env = gym.make('hironaka/HironakaAgent-v0', agent=RandomAgent(), dimension=dim)
        host = RandomHost()

        o = env.reset()
        env.render()
        while not env.stopped:
            action = host.select_coord(Points(np.expand_dims(o, axis=0)))[0]
            action_input = np.zeros(dim)
            action_input[action] = 1
            o, r, stopped, info = env.step(action_input)
            print(f"Reward: {r}")
            env.render()
