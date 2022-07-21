import unittest

import gym
import numpy as np
from gym.envs.registration import register

from hironaka.core import ListPoints
from hironaka.agent import RandomAgent
from hironaka.host import Zeillinger, RandomHost

register(
    id='hironaka/HironakaHost-v0',
    entry_point='hironaka.gym_env:HironakaHostEnv',
    max_episode_steps=10000,
)

register(
    id='hironaka/HironakaAgent-v0',
    entry_point='hironaka.gym_env:HironakaAgentEnv',
    max_episode_steps=10000,
)


class TestEnv(unittest.TestCase):
    def test_host_run_env(self):
        print("Host test starts.")
        env = gym.make('hironaka/HironakaHost-v0', host=Zeillinger())
        agent = RandomAgent()

        o = env.reset()
        env.render()
        stopped = False
        while not stopped:
            action = agent.move(ListPoints(np.expand_dims(o.get('points'), axis=0)), [np.where(o.get('coords') == 1)[0]],
                                inplace=False)
            o, r, stopped, info = env.step(action[0])
            print(f"Reward: {r}")
            env.render()

        print("Host test ends.")

    def test_agent_run_env(self):
        print("Agent test starts.")
        dim = 4
        env = gym.make('hironaka/HironakaAgent-v0', agent=RandomAgent(), dimension=dim)
        host = RandomHost()

        o = env.reset()
        env.render()
        stopped = False
        while not stopped:
            action = host.select_coord(ListPoints(np.expand_dims(o, axis=0)))[0]
            action_input = np.zeros(dim)
            action_input[action] = 1
            o, r, stopped, info = env.step(action_input)
            print(f"Reward: {r}")
            env.render()

        print("Agent test ends.")
