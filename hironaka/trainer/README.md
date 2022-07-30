# hironaka.trainer
This module provides an alternative method to facilitate training and is further used for large-scale distributed training and fine-tuning.

Using the gym environment through RL implementations like `stable-baseline3` is a great way to train. Codes are not published along with the repo (very dirty and messy, but may be available upon requests). But there are a few points which make it not a horrible idea to recreate a custom training facility:
- Our `gym` environment (`gym_env`) uses `ListPoints` and runs one game at a time. But the states can be recorded as a `Tensor` and operations can be vectorized, which allows us to run multiple games in GPU. This might be doable in `gym` too but may require multiple messy wrappers to reformat data.
- `stable-baseline3` is a very professionally written project. But everything comes with a trade-off. For example, when tracking and customizing a training process, one has to go through multiple subclasses in the inheritance chain scattered across the folders. Modularity is certainly a great engineering concept, but a fused and streamlined structure may have roles to play when horizontal scaling is unnecessary.
- We are running our codes on a massive GPU cluster consisting of thousands of nodes each of which has 8 Nvidia A100. Distributed training would have a lot of meanings in this context (distributed hyperparameter searching, distributed training of one single network, distributed model selection like genetic algorithm, etc.), which will eventually require a lot of customized modules.

# Content
The classes center around `Trainer`. Every trainer must be initialized with an exhaustive configuration dict **without** having hidden parameters defaulting to certain values. This way, users are always reminded about everything that goes into training. 

An example config file in the format of YAML is given for each `Trainer` subclass.
## Trainer
`Trainer` provides the facility of our training. To implement a specific way of training, please:
- create a class that inherits `Trainer`,
- read the docstring on what must be implemented and what can be overridden.
- implement the training logic in `_train()` based on what the class provides.

What the class provides include but are not limited to:
- Key objects:
  - `Trainer.host_net, Trainer.agent_net`: host and agent networks.
  - `Trainer.host_optimizer, Trainer.agent_optimizer`: host and agent optimizers.
  - `Trainer.host_lr_scheduler, Trainer.agent_lr_scheduler`: host and agent learning-rate schedulers.
  - `Trainer.host_er_scheduler, Trainer.agent_er_scheduler`: host and agent exploration-rate schedulers.
  - `Trainer.host_replay_buffer, Trainer.agent_replay_buffer`: host and agent replay buffers.
  - `Trainer.fused_game`: a FusedGame object for roll-out. Based on the host and agent network.
- Key methods:
  - `Trainer.set_training()`: set training mode to `True` or `False`. It impacts layers in networks like `BatchNorm1d`. 
  - `Trainer._roll_out()`: create roll-outs in the form of experiences (obs, actions, rewards, dones, next_obs).

### DQNTrainer
  Performs a classic DDQN.
## FusedGame
`FusedGame` is a class that fuses together everything about the gameplay. It avoids `gym` and our `Agent, Host, Game`, etc. All the data is process as `torch.Tensor`.
## ReplayBuffer
`ReplayBuffer` is a very simple implementation of replay buffer.
## Scheduler
`Scheduler` is basically a function that takes the number of steps as an input. The only extras are that it persists custom data.