import argparse
import logging
import os
import pathlib
import sys

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


def main(config_file: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))

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

    version_string = config['models']['version_string']

    env_h = gym.make("hironaka/HironakaHost-v0", host=Zeillinger(), config_kwargs=training_config)
    model_a = DQN("MultiInputPolicy", env_h, verbose=0, policy_kwargs=sb3_policy_config, batch_size=batch_size)

    p_a = NNPolicy(model_a.q_net.q_net, mode='agent', eval_mode=True, **training_config)
    nnagent = PolicyAgent(p_a)
    env_a = gym.make("hironaka/HironakaAgent-v0", agent=nnagent, config_kwargs=training_config)

    model_h = DQN("MlpPolicy", env_a, verbose=0, policy_kwargs=sb3_policy_config, batch_size=batch_size, gamma=1)

    p_h = NNPolicy(model_h.q_net.q_net, mode='host', eval_mode=True, **training_config)
    nnhost = PolicyHost(p_h, **training_config)
    env_h = gym.make("hironaka/HironakaHost-v0", host=nnhost, config_kwargs=training_config)

    for i in range(epoch):
        model_a.learn(total_timesteps=total_timestep)
        model_h.learn(total_timesteps=total_timestep)

        # Validation

        if i % save_frequency == 0:
            print(f"Epoch {i * 5}")
            print("agent validation:")
            _num_games = 1000
            agents = [nnagent, RandomAgent(), ChooseFirstAgent()]
            agent_names = ["neural_net", "random_agent", "choose_first"]
            perf_log = {}
            for agent, name in zip(agents, agent_names):
                validator = HironakaValidator(nnhost, agent, config_kwargs=config)
                result = validator.playoff(_num_games)
                print(str(type(agent)).split("'")[-2].split(".")[-1])
                print(f" - number of games:{len(result)}")
                perf_log[f"neural_net-{name}"] = len(result) / _num_games
            print(f"host validation:")
            hosts = [nnhost, RandomHost(), Zeillinger()]
            host_names = ["random_host", "zeillinger"]
            for host, name in zip(hosts, host_names):
                validator = HironakaValidator(host, nnagent, config_kwargs=config)
                result = validator.playoff(_num_games)
                print(str(type(host)).split("'")[-2].split(".")[-1])
                print(f" - number of games:{len(result)}")
                perf_log[f"{name}-neural_net"] = len(result) / _num_games
            wandb.log(perf_log, step=i)

        # Save model
        if i % save_frequency == 0:
            model_a.save(f"{model_path}/{version_string}_epoch_{i}_agent")
            model_h.save(f"{model_path}/{version_string}_epoch_{i}_host")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train the host and agent.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config_file", help="Specify config file location.")
    args = parser.parse_args()
    config_args = vars(args)
    main(**config_args)
