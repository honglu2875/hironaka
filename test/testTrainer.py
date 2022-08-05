import pathlib
import unittest

import gym
import torch
from gym import register
from stable_baselines3 import DQN

from hironaka.agent import ChooseFirstAgent
from hironaka.core import TensorPoints
from hironaka.host import Zeillinger
from hironaka.src import merge_experiences
from hironaka.trainer.DQNTrainer import DQNTrainer
from hironaka.trainer.FusedGame import FusedGame
from hironaka.trainer.ReplayBuffer import ReplayBuffer

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
    "max_num_points": 20,
    "device_key": 'cuda',
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


class TestTrainer(unittest.TestCase):
    def test_fused_game_random_run(self):
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
        r = game.step(p, 'agent', scale_observation=False)

    def test_replay_buffer_shape(self):
        env_a = gym.make("hironaka/HironakaAgent-v0", agent=ChooseFirstAgent(), config_kwargs=config)
        model_h = DQN("MlpPolicy", env_a, verbose=0, policy_kwargs=sb3_policy_config, batch_size=32, learning_rate=1e-5,
                      learning_starts=100)
        env_h = gym.make("hironaka/HironakaHost-v0", host=Zeillinger(), config_kwargs=config)
        model_a = DQN("MultiInputPolicy", env_h, verbose=0, policy_kwargs=sb3_policy_config, batch_size=32,
                      learning_rate=1e-5, learning_starts=100)
        points_t = torch.randint(5, (2, 20, 3)).type(torch.float)
        p = TensorPoints(points_t, padding_value=-1e-8)
        game = FusedGame(model_h.q_net, model_a.q_net, device_key='cpu')

        replay_buffer = ReplayBuffer((20, 3), 8, 1000, torch.device('cpu'))
        exp = game.step(p, 'host', scale_observation=False)
        length = exp[0].shape[0]

        old_pos = replay_buffer.pos
        replay_buffer.add(*exp)  # If adding buffer is successful -> the shapes are right.
        assert replay_buffer.pos - old_pos == length  # make sure add in the same amount of rows

        replay_buffer = ReplayBuffer({'points': (20, 3), 'coords': (3,)}, 3, 1000, torch.device('cpu'))
        exp = game.step(p, 'agent', scale_observation=False)
        replay_buffer.add(*exp)

        roll_outs = [
            game.step(p, 'agent', scale_observation=False, exploration_rate=0)
            for _ in range(5)]
        replay_buffer.add(*merge_experiences(roll_outs))

    def test_dqn_trainer(self):
        dqn_trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test.yml')
        dqn_trainer.train(1)
        print(dqn_trainer.time_log)
        print(dqn_trainer.fused_game.time_log)
