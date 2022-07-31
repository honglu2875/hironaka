# Hironaka

A utility package for a reinforcement learning study of Hironaka's game of local resolution of singularities and its
variation problems.

# Quick start
[This quick tutorial](https://cocalc.com/share/public_paths/5db3252a0bcb8d068aad2ee53bf5a1ce85753ebf) provides a brief
demonstration of some key classes in this repo.

There are 2 ways to start a proper Reinforcement Learning training:
- (TL;DR, clone this [Google Colab file](https://colab.research.google.com/drive/1nVnVA6cyg0GT5qTadJTJH7aU6smgopLm?usp=sharing), forget what I say below and start your adventure)

    `DQNTrainer` is a quick implementation combining my interface `Trainer` with `stable-baseline3`'s DQN codes. It runs in 3 lines:
    ```
    from hironaka.trainer.DQNTrainer import DQNTrainer
    trainer = DQNTrainer('dqn_config_test.yml')
    trainer.train(100)
    ```
  Of course, for this to work you need to 
  - set up the system path so that Python can import those stuff;
  - copy the config file `dqn_config_test.yml` from `.test/` to your running folder.
- When you are here in the project folder and `requirements.txt` are met (or create a venv and run `pip install -r requirements.txt`), try the following:
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

## Rewards and metrics

Defining the reward function is an open-ended question. But our goal is:
- The host needs to minimize the game length (to stop in finite steps is already a challenge).
- The agent needs to maximize the game length.

Therefore, the most straightforward reward function for a host is `1 if game.ended else 0`. For agent, we switch around to `1 if not game.ended else 0`. There are more (e.g., step reward:=number of points killed for host). But experiments have been telling us to focus on the original one.

### An interesting metric: $\rho$
Fix a pair of host and agent. Given an integer $n$, we can run a random game for $n$ steps (restart another random game if ended). Let $g(n)$ be the number of games that happen during the $n$ steps. An important metric that measures the pair of host and agent is the ratio

$$\rho(n) = \dfrac{g(n)}{n}.$$

This ratio directly relates to the cumulative rewards for host and agent ($n\rho(n)$ for host and $n(1-\rho(n))$ for agent). If a limit

$$\DeclareMathOperator*{\lim}{\text{lim}}\rho = \displaystyle\lim_\limits{n\rightarrow\infty}\rho(n)$$

exists, it would be an important measure for the pair of host and agent: 
- if $\rho$ is high, the host is strong; 
- if $\rho$ is low, the agent is strong. 

The existence of the limit is not proven, but empirically $\rho(n)$ seems to converge to some values when $n$ grows larger.

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
