# Hironaka

A utility package for a reinforcement learning study of Hironaka's game of local resolution of singularities and its
variation problems.

# Quick start
[This quick tutorial](https://cocalc.com/share/public_paths/5db3252a0bcb8d068aad2ee53bf5a1ce85753ebf) provides a brief
demonstration of key classes in this repo. It is highly recommended to take a look first if you are an example-oriented
learner.

There are 2 ways to start a proper Reinforcement learning training:
- (TL;DR, clone this [Google Colab file](https://colab.research.google.com/drive/1nVnVA6cyg0GT5qTadJTJH7aU6smgopLm?usp=sharing), forget what I say below and start your adventure)

    If you trust that I am not a Python idiot, feel free to subclass my own interface `Trainer` and write a double-player training routine. `DQNTrainer` is a quick implementation combining `Trainer` with `stable-baseline3`'s DQN codes. It runs in 3 lines:
    ```
    from hironaka.trainer.DQNTrainer import DQNTrainer
    trainer = DQNTrainer('dqn_config_test.yml')
    trainer.train(100)
    ```
  Of course, for this to work you need to 
  - set up the system path, so that Python can import those stuff;
  - copy the config file `dqn_config_test.yml` from `.test/` to your running folder.
- Assuming you are here in the project folder, and `requirements.txt` are satisfied (or create a venv and run `pip install -r requirements.txt`), run the following:
    ```
    python train/train_sb3.py
    ```
  It starts from our base classes `Host, Agent`, goes through the gym wrappers `.gym_env.HironakaHostEnv, .gym_env.HironakaAgentEnv`, and ends up using `stable_baseline3`'s implementations. In this particular script, it uses their `DQN` class. But you can totally try other stuff like `PPO` with corresponding adjustments.

# Contents

For ML and RL specialists, hopefully the [Quick Start](#quick-start) already gives you a good idea about where to start. In addition, check out

- [Rule of the game](#rule-of-the-game)
- [The structure of the repo](#the-structure-of-the-repo)

For math-oriented viewers or ML experts who are intrigued about the background story, please feel free to continue with:

- [What is a resolution of singularity](#what-is-a-resolution-of-singularity)
- [What is Hironaka's polyhedral game](#what-is-hironakas-polyhedral-game)
- [Variations of Hironaka's game](#variations-of-hironakas-game)
- [Further topics](#further-topics)

# Rule of the game

## The definition of the game

The original Hironaka game is a game consisting of 2 players. They operate in a non-symmetric fashion. To emphasize the
difference, let us call player 1 the "host", player 2 the "agent". For every turn the game has a `state`, the host makes
a move, and the agents makes a move. Their moves change the `state` and the game goes into the next turn.

A `state` is represented by a set of points $S\in\mathbb Z^n$ who are the vertices of the Newton polytope $S$ itself.

Every turn,

- The host chooses a subset $I\subset \{1,2,\cdots, n\}$ such that $|I|\geq 2$.
- The agent chooses a number $i\in I$.

$i, I$ together changes the `state` $S$ to the next according to the following linear change of variables:

$$x_j \mapsto \begin{cases}x_j, &\qquad\text{if } i\neq j \newline \sum\limits_{k\in I} x_k, &\qquad\text{if }i=j
\end{cases},$$

for points $(x_1,\cdots,x_n)\in \mathbb Z^n$. We subsequently apply Newton polytope to the transformed points and only
keep the vertices.

A `state` is terminal if it consists of one single point. In this case, the game will not continue and the host wins. As
a result, the host wants to reduce the number of $S$ as quickly as possible, but the agent wants to keep the number of
$S$ for as long as possible.

# The structure of the repo
For the detailed structure of the repo, please check out the README starting from the [hironaka](hironaka) package. But we would like to draw ML and RL researcher's attention to this submodule first:
 - [hironaka.trainer](hironaka/trainer)

This is directly related to the current progress of the model training. I hope the codes and comments are self-explanatory.

---
Now the big question is, how is this game related to some fundamental questions in pure math, or specifically, algebraic
geometry.

# What is a resolution of singularity

[intro]

## Smoothness

[definition] & [examples]

## Singularities

[examples]

## Blow-up: turning singularities into smooth points

[examples]

## Hironaka's theorem

[statement]

# What is Hironaka's polyhedral game

## Monomial ideals

[definition] & [examples]

## Rephrasing the local resolution of singularity problem

[definitions]

# Variations of Hironaka's game

## Thom game

## Singularity theory and Thom polynomial

# Further topics

...
