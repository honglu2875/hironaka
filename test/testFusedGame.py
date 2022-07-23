import unittest

import gym
import torch
from gym import register
from stable_baselines3 import DQN

from hironaka.agent import ChooseFirstAgent
from hironaka.core import TensorPoints
from hironaka.host import Zeillinger
from hironaka.util import FusedGame

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

config = {
    "dimension": 3,
    "max_value": 20,
    "masked": True,
    "max_number_points": 20,
    "use_cuda": True,
    "normalized": False,
    "value_threshold": 1e8,
    "step_threshold": 200,
    "fixed_penalty_crossing_threshold": 0,
    "stop_at_threshold": True,
    "improve_efficiency": True,
    "scale_observation": True,
    "reward_based_on_point_reduction": False,
    "use_discrete_actions_for_host": True
}

sb3_policy_config = {
    "net_arch": [32, 32],
    "normalize_images": False,
}


class TestFusedGame(unittest.TestCase):
    def test_random_run(self):
        env_a = gym.make("hironaka/HironakaAgent-v0", agent=ChooseFirstAgent(), config_kwargs=config)
        model_h = DQN("MlpPolicy", env_a, verbose=0, policy_kwargs=sb3_policy_config, batch_size=32, learning_rate=1e-5,
                      learning_starts=100)
        env_h = gym.make("hironaka/HironakaHost-v0", host=Zeillinger(), config_kwargs=config)
        model_a = DQN("MultiInputPolicy", env_h, verbose=0, policy_kwargs=sb3_policy_config, batch_size=32,
                      learning_rate=1e-5, learning_starts=100)

        points_t = torch.randint(5, (2, 20, 3)).type(torch.float)

        p = TensorPoints(points_t, padding_value=-1e-8)
        game = FusedGame(model_h.q_net, model_a.q_net, device_key='cpu')

        r = game.step(p, 'host', scale_observation=False)
        print(r)
        r = game.step(p, 'agent', scale_observation=False)
        print(r)
