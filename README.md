# Hironaka

A utility package for a reinforcement learning study of Hironaka's game of local resolution of singularities and its
variation problems.

# Quick start

[This quick tutorial](https://cocalc.com/share/public_paths/5db3252a0bcb8d068aad2ee53bf5a1ce85753ebf) provides a brief
demonstration of key classes in this repo. It is highly recommended to take a look first if you are an example-oriented
learner.

# Contents

For ML and RL specialists, the following are what you need for a quick start.

- [Rule of the game](#rule-of-the-game)
- [The quick tutorial](https://cocalc.com/share/public_paths/5db3252a0bcb8d068aad2ee53bf5a1ce85753ebf)
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
