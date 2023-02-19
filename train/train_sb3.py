import argparse
import logging
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any, Optional, Union, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from hironaka.policy import NNPolicy
from hironaka.validator import HironakaValidator
import wandb
import gym
from gym.envs.registration import register
import yaml

from stable_baselines3 import DQN

from hironaka.agent import RandomAgent, ChooseFirstAgent, PolicyAgent
from hironaka.host import Zeillinger, RandomHost, PolicyHost
from hironaka.util.sb3_util import CustomLogger, configure

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm


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

sb3_policy_config = {
    "net_arch": [256] * 4,
    "normalize_images": False}


class SB3Logger(CustomLogger):
    prefix = ""
    steps = defaultdict(int)

    def record(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().record(key, value, exclude)

    def record_mean(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().record_mean(key, value, exclude)

    def export(self):
        logs = defaultdict(dict)
        for key in self.history_value:
            for i in range(len(self.history_value[key])):
                logs[i].update({f"{self.prefix}/{key}": self.history_value[key][i],
                                f"{self.prefix}/{key}_step": self.steps[key] + i})
            self.history_value[key] = []
            self.steps[key] += len(self.history_value[key])
        for key in self.history_mean_value:
            for i in range(len(self.history_mean_value[key])):
                logs[i].update({f"{self.prefix}/{key}_mean": self.history_mean_value[key][i],
                                f"{self.prefix}/{key}_mean_step": self.steps[key] + i})
            self.history_mean_value[key] = []
            self.steps[key] += len(self.history_mean_value[key])
        return logs


class HostLogger(SB3Logger):
    prefix = "host"


def combine(log1, log2) -> dict:
    logs = defaultdict(dict)
    for i in log1:
        logs[i].update(log1[i])
    for i in log2:
        logs[i].update(log2[i])
    return logs


class AgentLogger(SB3Logger):
    prefix = "agent"


class ValidateCallback(BaseCallback):
    def __init__(
        self, nnagent, nnhost, cfg, agent_logger, host_logger, save_frequency,
        model_a, model_h, model_path, role, verbose: int = 0
    ):
        super().__init__(verbose)
        self.nnagent = nnagent
        self.nnhost = nnhost
        self.cfg = cfg
        self.agent_logger = agent_logger
        self.host_logger = host_logger
        self.save_frequency = save_frequency
        self.model_a = model_a
        self.model_h = model_h
        self.model_path = model_path
        self.last_n_updates = 0
        self.role = role

    def _on_step(self) -> bool:
        return True

    def on_rollout_start(self):
        n_update = self.model_a._n_updates if self.role == "agent" else self.model_h._n_updates

        if n_update // self.save_frequency <= self.last_n_updates // self.save_frequency:
            return

        print("agent validation:")
        _num_games = 1000
        agents = [self.nnagent, RandomAgent(), ChooseFirstAgent()]
        agent_names = ["neural_net", "random_agent", "choose_first"]
        perf_log = {"validation_step": n_update // self.save_frequency}
        for agent, name in zip(agents, agent_names):
            validator = HironakaValidator(self.nnhost, agent, config_kwargs=self.cfg)
            result = validator.playoff(_num_games)
            print(str(type(agent)).split("'")[-2].split(".")[-1])
            print(f" - number of games:{len(result)}")
            perf_log[f"neural_net-{name}"] = len(result) / _num_games
        print(f"host validation:")
        hosts = [self.nnhost, RandomHost(), Zeillinger()]
        host_names = ["random_host", "zeillinger"]
        for host, name in zip(hosts, host_names):
            validator = HironakaValidator(host, self.nnagent, config_kwargs=self.cfg)
            result = validator.playoff(_num_games)
            print(str(type(host)).split("'")[-2].split(".")[-1])
            print(f" - number of games:{len(result)}")
            perf_log[f"{name}-neural_net"] = len(result) / _num_games
        logs = combine(self.agent_logger.export(), self.host_logger.export())
        logs[0].update(perf_log)
        for i in logs:
            wandb.log(logs[i], commit=True)

        self.model_a.save(f"{self.model_path}/{self.cfg.version_string}_epoch_{n_update // self.save_frequency}_agent")
        self.model_h.save(f"{self.model_path}/{self.cfg.version_string}_epoch_{n_update // self.save_frequency}_host")


def main(config_file: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
    sb3_logger_host = configure(None, None, HostLogger)
    sb3_logger_agent = configure(None, None, AgentLogger)
    wandb.init(project="hironaka_sb3", config=config_file)

    model_path = 'models'
    if config_file is None:
        config_file = 'train/config.yml'
    if not os.path.exists(model_path):
        logger.info("Created 'models/'.")
        os.makedirs(model_path)
    else:
        logger.warning("Model folder 'models/' already exists.")

    with open(config_file, "r") as stream:
        config = yaml.safe_load(stream)  # Generate the config as a dict object

    training_config = config['global']

    epoch = config['training']['epoch']
    batch_size = config['training']['batch_size']
    save_frequency = config['training']['save_frequency']
    total_timestep = config['training']['total_timestep']
    learning_starts = config['training']['learning_starts']
    lr = config['training']['lr'] if 'lr' in config['training'] else 1e-3
    version_string = config['models']['version_string']
    log_interval = config['training']['log_interval']

    env_h = gym.make("hironaka/HironakaHost-v0",
                     host=Zeillinger(),
                     config_kwargs=training_config)
    model_a = DQN("MultiInputPolicy", env_h,
                  verbose=0, policy_kwargs=sb3_policy_config,
                  batch_size=batch_size, learning_starts=learning_starts,
                  learning_rate=lr, target_update_interval=total_timestep*5)
    model_a.set_logger(sb3_logger_agent)

    p_a = NNPolicy(model_a.q_net.q_net, mode='agent', eval_mode=True, **training_config)
    nnagent = PolicyAgent(p_a)
    env_a = gym.make("hironaka/HironakaAgent-v0", agent=nnagent, config_kwargs=training_config)

    model_h = DQN("MlpPolicy", env_a,
                  verbose=0, policy_kwargs=sb3_policy_config,
                  batch_size=batch_size, gamma=1, learning_starts=learning_starts,
                  learning_rate=lr, target_update_interval=total_timestep*5)
    model_h.set_logger(sb3_logger_host)

    p_h = NNPolicy(model_h.q_net.q_net, mode='host', eval_mode=True, **training_config)
    nnhost = PolicyHost(p_h, **training_config)
    env_h = gym.make("hironaka/HironakaHost-v0", host=nnhost, config_kwargs=training_config)

    running_lr = lr

    callback_a = ValidateCallback(nnagent, nnhost, training_config, sb3_logger_agent, sb3_logger_host, save_frequency,
                                  model_a, model_h, model_path, "agent")
    callback_h = ValidateCallback(nnagent, nnhost, training_config, sb3_logger_agent, sb3_logger_host, save_frequency,
                                  model_a, model_h, model_path, "host")
    for i in range(epoch):
        model_a.lr_schedule = lambda _: running_lr
        model_h.lr_schedule = lambda _: running_lr
        model_a.learn(total_timesteps=total_timestep,
                      log_interval=log_interval,
                      reset_num_timesteps=False,
                      callback=callback_a)
        model_h.learn(total_timesteps=total_timestep,
                      log_interval=log_interval,
                      reset_num_timesteps=False,
                      callback=callback_h)
        running_lr *= 0.95


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the host and agent.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Specify config file location.")
    args = parser.parse_args()
    config_args = vars(args)
    main(**config_args)
