# /train

First of all, this is **NOT** a submodule. This is a separate area for python scripts related to training. We want to
keep a record of some important training scripts but they are not decent enough to be merged into the mainn library.
This folder exists only for documenting purposes. Please proceed with caution.

## .train_sb3.py

The most straightforward and perhaps lazy way to solve an RL problem is to plug into a general-purpose RL trainer and
let it find a policy for the players. It wraps the environment into OpenAI Gym, and there are some amazing implementations that connects to these (stable-baselines3, etc).

This script uses our gym wrapper `gym_env` to train using `stable_baselines3`.

## .jax_mcts.py
The JAX training script. It runs and is our best effort so far (with the right hyper-parameters of course).