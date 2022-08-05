# /train

First of all, this is **NOT** a submodule. This is a separate area for python scripts related to training. We want to
keep a record of some important training scripts but they are not decent enough to be merged into the mainn library.
This folder exists only for archiving purposes. Please proceed with caution.

## .train_sb3.py

The most straightforward and perhaps lazy way to solve an RL problem is to plug into a general-purpose RL trainer and
let it find a policy for the players. There has been a lot of general purpose algorithms that are agnostic of the exact
game (AlphaZero, MuZero, etc.). Without using those cannons there are already libraries wrapping games into a
well-defined framework (gym), and there are some amazing implementations that connects to these (stable-baseline,
stable-baselines3, etc).

This script uses our gym wrapper `gym_env` to train using `stable_baselines3`.