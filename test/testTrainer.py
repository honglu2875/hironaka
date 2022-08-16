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
from hironaka.trainer.player_modules import RandomAgentModule


class TestTrainer(unittest.TestCase):
    dqn_trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test.yml')

    def test_dummy_module_init(self):
        with self.assertRaises(Exception) as context:
            trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test_host_only.yml')
        trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test_host_only.yml',
                             agent_net=RandomAgentModule(3, 20, device=torch.device('cpu')))
        assert trainer.trained_roles == ['host']
        trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test_host_only.yml',
                             agent_net=self.dqn_trainer.agent_net)
        assert trainer.trained_roles == ['host']

    def test_replace_net(self):
        trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test_host_only.yml',
                             agent_net=RandomAgentModule(3, 20, device=torch.device('cpu')))
        trainer.replace_nets(agent_net=self.dqn_trainer.agent_net)
        assert 'agent' not in trainer.trained_roles

        trainer = DQNTrainer(str(pathlib.Path(__file__).parent.resolve()) + '/dqn_config_test.yml',
                             agent_net=RandomAgentModule(3, 20, device=torch.device('cpu')))
        assert 'agent' not in trainer.trained_roles
        with self.assertRaises(Exception) as context:
            trainer.set_trainable(['agent'])
        trainer.replace_nets(agent_net=self.dqn_trainer.agent_net)
        trainer.set_trainable(['agent'])
        assert 'agent' in trainer.trained_roles

    def test_fused_game_random_run(self):
        """
        env_a = gym.make("hironaka/HironakaAgent-v0", agent=ChooseFirstAgent(), config_kwargs=config)
        model_h = DQN("MlpPolicy", env_a, verbose=0, policy_kwargs=sb3_policy_config, batch_size=32, learning_rate=1e-5,
                      learning_starts=100)
        env_h = gym.make("hironaka/HironakaHost-v0", host=Zeillinger(), config_kwargs=config)
        model_a = DQN("MultiInputPolicy", env_h, verbose=0, policy_kwargs=sb3_policy_config, batch_size=32,
                      learning_rate=1e-5, learning_starts=100)
        """
        points_t = torch.randint(5, (2, 20, 3)).type(torch.float)

        p = TensorPoints(points_t, padding_value=-1e-8)
        game = self.dqn_trainer.fused_game

        r = game.step(p, 'host', scale_observation=False)
        r = game.step(p, 'agent', scale_observation=False)

    def test_replay_buffer_shape(self):
        points_t = torch.randint(5, (2, 20, 3)).type(torch.float)
        p = TensorPoints(points_t, padding_value=-1e-8)

        game = self.dqn_trainer.fused_game
        replay_buffer = ReplayBuffer((20, 3), 4, 100, torch.device('cpu'))
        exp = game.step(p, 'host', scale_observation=False)
        length = exp[0].shape[0]

        old_pos = replay_buffer.pos
        replay_buffer.add(*exp)  # If adding buffer is successful -> the shapes are right.
        assert replay_buffer.pos - old_pos == length  # make sure add in the same amount of rows

        replay_buffer = ReplayBuffer({'points': (20, 3), 'coords': (3,)}, 3, 100, torch.device('cpu'))
        exp = game.step(p, 'agent', scale_observation=False)
        replay_buffer.add(*exp)

        roll_outs = [
            game.step(p, 'agent', scale_observation=False, exploration_rate=0)
            for _ in range(5)]
        replay_buffer.add(*merge_experiences(roll_outs))

    def test_dqn_trainer(self):
        self.dqn_trainer.train(1)
        print(self.dqn_trainer.time_log)
        print(self.dqn_trainer.fused_game.time_log)
