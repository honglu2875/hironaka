![status](https://github.com/honglu2875/hironaka/actions/workflows/main.yml/badge.svg?branch=main)

# Hironaka

A utility package for a reinforcement learning study of Hironaka's game of local resolution of singularities and its
variation problems.

# Quick start

## Basic usage

[This quick tutorial](https://cocalc.com/share/public_paths/5db3252a0bcb8d068aad2ee53bf5a1ce85753ebf) provides a short
demonstration of some key classes in this repo.

## Reinforcement Learning

There are 2 ways to start a proper Reinforcement Learning training:

- (TL;DR, clone
  this [Google Colab file](https://colab.research.google.com/drive/1nVnVA6cyg0GT5qTadJTJH7aU6smgopLm?usp=sharing),
  forget what I say below and start your adventure)

  `DQNTrainer` is a quick implementation combining my interface `Trainer` with `stable-baseline3`'s DQN codes. It runs
  in 3 lines:
    ```python
    from hironaka.trainer.DQNTrainer import DQNTrainer
    trainer = DQNTrainer('dqn_config_test.yml')
    trainer.train(100)
    ```
  Of course, for this to work you need to
    - set up the system path so that Python can import those stuff;
    - copy the config file `dqn_config_test.yml` from `.test/` to your running folder.
- When you are here in the project folder and `requirements.txt` are met (or create a venv and
  run `pip install -r requirements.txt`), try the following:
    ```bash
    python train/train_sb3.py
    ```
  It starts from our base classes `Host, Agent`, goes through the gym
  wrappers `.gym_env.HironakaHostEnv, .gym_env.HironakaAgentEnv`, and ends up using `stable_baseline3`'s
  implementations. In this particular script, it uses their `DQN` class. But you can totally try other stuff like `PPO`
  with corresponding adjustments.

## Experiments

Some of our experimental results are documented here: https://github.com/honglu2875/hironaka-experiments

# Contents

For ML and RL specialists, hopefully the [Quick Start](#quick-start) already gives you a good idea about where to start.
In addition, please check out

- [Rule of the game](#rule-of-the-game)
- [The structure of the repo](#the-structure-of-the-repo)

For math-oriented viewers or ML experts who are intrigued about the background story, please feel free to continue with:

- [What is a resolution of singularity](#what-is-a-resolution-of-singularity)
- [What is Hironaka's polyhedral game](#what-is-hironakas-polyhedral-game)
- [Variations of Hironaka's game](#variations-of-hironakas-game)
- [Further topics](#applications-and-open-questions)

# Rule of the game

## The definition of the game

All versions of the game consist of 2 players. They operate in a non-symmetric fashion. To emphasize the
difference, let us call Player A the "host", player B the "agent". For every turn the game has a `state`, the host makes
a move, and the agents makes a move. Their moves change the `state` and the game goes into the next turn.

### States

A `state` is represented by a set of points $S\in\mathbb Z^n$ satisfying certain rules depending of the different versions
of the game. At each turn,

- The host chooses a subset $I\subset \{1,2,\cdots, n\}$ such that $|I|\geq 2$.
- The agent chooses a number $i\in I$.

### State change

The pair $(I, i)$ defines a state change, which is a simple linear transformation from $S$ to $S'$ according to a 
certain rule. Player A wins if 
the `state` becomes `terminal state`, where the set of `terminal states` are defined in each version 
slightly different ways. 

## Rewards and metrics

Defining the reward function is an open-ended question. But our goal is:

- The host needs to minimize the game length (to stop in finite steps is already a challenge).
- The agent needs to maximize the game length.

Therefore, the most straightforward reward function for a host is `1 if game.ended else 0`. For agent, we switch around
to `1 if not game.ended else 0`. There are more (e.g., step reward:=number of points killed for host). But experiments
have been telling us to focus on the original one.

### An interesting metric: $\rho$

With certain amount of abuse of notations (please forgive me for omitting the dependence on random initial states and sequences of actions), the metric is heuristically described as follows:

Fix a pair of host and agent. Given an integer $n$, we can run a random game for $n$ steps (restart another random game
if ended). Let $g(n)$ be the number of games that happen during the $n$ steps. An important metric that measures the
pair of host and agent is the ratio

$$\rho(n) = \dfrac{g(n)}{n}.$$

This ratio directly relates to the cumulative rewards for host and agent ($n\rho(n)$ for host and $n(1-\rho(n))$ for
agent). If a limit

$$\DeclareMathOperator*{\lim}{\text{lim}}\rho = \displaystyle\lim_\limits{n\rightarrow\infty}\rho(n)$$

exists (w.r.t. random initializations over a compact subspace of states), it would be an important measure for the pair of host and agent:

- if $\rho$ is high, the host is strong;
- if $\rho$ is low, the agent is strong.

The existence of the limit is not proven, but empirically $\rho(n)$ seems to converge to some values when $n$ grows
larger.

# The structure of the repo

For the detailed structure of the repo, please check out the README starting from the [hironaka](hironaka) package. But
we would like to draw ML and RL researcher's attention to this submodule first:

- [hironaka.trainer](hironaka/trainer)

This is directly related to the current progress of the model training. I hope the codes and comments are
self-explanatory.

---
Now the big question is, how this game is related to some fundamental questions in pure math, or specifically, algebraic
geometry.

# What is a resolution of singularity

An affine algebraic variety

$$X =\{(x_1,\ldots, x_n): f_1(x_1,\ldots, x_n)=\ldots =f_k(x_1,\ldots, x_n)=0\} \subset A^n$$

is the common zero locus of polynomial equations. Affine varieties play central role in mathematics, physics and biology.   

Affine varieties cut out by one polynomial equation are called affine hypersurfaces. E.g

$$X=\{(x_1,\ldots, x_n):f(x_1,\ldots, x_n)=0\}$$

## Singularities

We can think of varieties as "shapes in affine spaces", and at a generic point 
$x \in X$ the variety locally is $A^r$ for some $r$, which we call the dimension of $X$.
However, there are special, ill-behaved points, where the local geometry of $X$ is less patent.

The set $X$ is singular at a point $a \in X$ if the Jacobian matrix

$$Jac(X,a)=\left(\frac{\partial f_i}{\partial x_j}\right)(a)$$

at a is of rank smaller than $n-dim(X)$. The set of singular points of $X$ is called the singular locus of $X$.

## Blow-up: turning singularities into smooth points

Resolution of singularities is a classical central problem in geometry. By resolution we mean that we substitute the original,
possibly singular $X$ with a nonsingular $Y$ with a proper 
birational map $f:Y \to X$ such that $f$ is an isomorphism over some open dense subset of X.

The celebrated Hironaka theorem asserts that such resolution exists for all $X$, and it can be constructed from an 
elementary type of operation called blowing up. Blowing up or blowup is a type of geometric transformation which 
replaces a subspace of a given space  with all the directions pointing out of that subspace. 

For example, the blowup of a point in a plane replaces the point 
with the projectivized tangent space at that point. The geometric picture looks like the following: 

![blow-up](img/blow-up.png)

([The lecture note where I took the screenshot from](https://www.maths.tcd.ie/~btyrrel/flatness.pdf). 
This picture ultimately came from our beloved textbook: Algebraic geometry *by Robin Hartshorne*)

## Hironaka's theorem

The most famous and general result concerning resolution of singularities was given by Heisuke Hironaka in 1964 
He proved that the resolution of singularities can be achieved by a sequence of blowups 

$$Y=X_n \to X_{n-1} \to \ldots \to X_0=X$$

if the characteristic of the base field is zero.

This beautiful and fundamental work was recognized with a Fields medal in 1970. Recently, Villamayor and, independently, 
Bierstone and Milman have clarified the process of resolution of singularities in characteristic zero, 
explicitly describing the algorithmic nature of the resolution process. 

Using de Jong's deep ideas, simple proofs of Hironaka's theorem have been discovered by de Jong and Abramovich and by 
Bogomolov and Pantev.

The latest big progress in approaching a simple resolution algorithm was achieved by 
Abramovich-Tempkin-Vlodarczyk and independenty by McQuillen, who 
came up with a simple stacky presentation using weighted blow-ups. 

# What is Hironaka's polyhedral game

In the literature, there appear at least 11 proofs for Hironaka's celebrated theorem on the resolution of singularities
of varieties of arbitrary dimension defined over fields of characteristic zero. These proofs associate invariants to 
singularities, and show that certain type of blow-ups improve the invariant. 

We can interpret resolution as a game between two players. Player A attempts to improve the singularities. 
Player B is some malevolent adversary who tries to keep the singularities alive as long as possible. 
The first player chooses the centres of the blowups, the second provides new order functions after each blowup. 

## Monomial ideals

[definition] & [examples]

## Rephrasing the local resolution of singularity problem

[definitions]

# Variations of Hironaka's game

## Short summary 

The formulation of the resolution as a game goes back to Hironaka himself. He introduced the polyhedra game where 
Player A has a winning strategy, which provide resolution of hypersurface singularities. He formulated a "hard" 
polyhedra game, where a winning strategy for Player A would imply the resolution theorem in full generality, but such 
winning strategy does not necessarily exist. 
Later Hauser defined a game which provided a new proof of the Hironaka theorem. 

Certain modified version of the game, due to Bloch and Levine provide solutions to the moving cylcles problem. 
Finally, recent work of Berczi shows that a restricted weighted version of the Hironaka game provides closed integration
formulas over Hilbert scheme of points on manifolds.  

## The basic version of the game

This project started out with [this basic version of the game](#the-definition-of-the-game) (see the link for the rules). 
It is a simplified version of [Hironaka's original polyhedra game](#hironakas-polyhedra-game).

## Hauser game 

This version of the Hironaka game was suggested by Hauser. A simple winning strategy was given by Zeillinger. The existence of winning 
strategy proves the resolution theorem for hypersurfaces.

**The rules:**
- `states`: A finite set of points $S\subset\mathbf{N}^n$, such that $S$ is the set of vertices of the positive 
convex hull $\Delta=\{S+\mathbf{R}^n_+\}$. 
- `state change`: Given the pair $(I,i)$ chosen by the host, for $x=(x_1,\cdots,x_n)\in\mathbb Z^n$ we define
$T_{I,i}(x)=(x_1',\ldots, x_n')$ where

$$x_j' = \begin{cases}x_j, &\qquad\text{if } i\neq j \newline \sum\limits_{k\in I} x_k, &\qquad\text{if }i=j
\end{cases},$$

The new `state` $S'$ is formed by the vertices of the Newton polyhedron of $\Delta'=\{T_{I,i}(x):x\in S\}$.
- `terminal states`: a state $S$ is terminal if it consists of one single point. 
\end{enumerate}
In short, the host wants to reduce the size of $S$ as quickly as possible, but the agent wants to keep the size of
$S$ large.

## Hironaka's polyhedra game 

This is the original Hironaka game from 1970. A winning strategy for the host
was given by Mark Spivakovsky in 1980, which proved the resolution theorem for hypersurfaces.  

**The rules:**
- `states`: A finite set of rational points $S \subset \mathbf{Q}^n$, such that $\sum\limits_{i=1}^n x_i>1$ for all 
$(x_1,\ldots, x_n)\in S$, and $S$ is the set of vertices of the positive 
convex hull $\Delta=\{S+\mathbf{R}^n_+\}$. 
- `move`: The host chooses a subset $I\subset \{1,2,\cdots, n\}$ such that $|I|\geq 2$ and 
\sum\limits_{i\in I}x_i\ge 1$ for all $(x_1,\ldots, x_n)\in S$. The agent chooses a number $i\in I$.
- `state change`: Given the pair $(I,i)$ chosen by the host, for $x=(x_1,\cdots,x_n)\in \mathbb Z^n$ we define
$T_{I,i}(x)=(x_1',\ldots, x_n')$ where

$$x_j' = \begin{cases}x_j, &\qquad\text{if } i\neq j \newline \sum\limits_{k\in I} x_k -1, &\qquad\text{if }i=j
\end{cases},$$

The new `state` $S'$ is formed by the vertices of the Newton polyhedron of $\Delta'=\{T_{I,j}(x):x\in S\}$.
- `terminal states`: a state $S$ is terminal if it consists a point $(x_1,\ldots, x_n)$ such that 
$\sum\limits_{i=1}^n x_i \le 1$. 

## Hard polyhedra game 

The hard polyhedra game was proposed by Hironaka in 1978. 
Hironaka has proved that an affirmative solution of this game would imply the local uniformization theorem for an
algebraic variety over an algebraically closed field of any characteristic.
Mark Spivakovsky showed that Player A does not always have a winning strategy

**The rules:**
- `states`: A finite set of rational points $S \subset \mathbf{Q}^n$, such that $\sum\limits_{i=1}^n x_i>1$ for all 
$(x_1,\ldots, x_n)\in S$, the denominators are bounded by some fix $N$, and $S$ is the set of vertices of the positive 
convex hull $\Delta=\{S+\mathbf{R}^n_+\}$. 
- `move`: The host chooses a subset $I\subset \{1,2,\cdots, n\}$ such that $|I|\geq 2$ and 
\sum\limits_{i\in I}x_i\ge 1$ for all $(x_1,\ldots, x_n)\in S$. 
The agent chooses some element $i\in S$ and modifies the Newton polygon $\Delta$ to a set $\Delta^*$ by
the following procedure: first, the agent selects a finite number of points $y=(y_1,\ldots, y_n)$, all of whose 
coordinates are rational numbers with denominators bounded by $N$ as above, and for each of which there exists
an $x = (x_1, \ldots, x_n)\in \Delta$ which satisfy some basic relations. $\Delta^*$ is then taken to be the positive 
convex hull of $\Delta \cup \{selected points\}$.

- `state change`: Given the pair $(I,i)$ chosen by the host, for $x=(x_1,\cdots,x_n)\in \mathbb Z^n$ we define
$T_{I,i}(x)=(x_1',\ldots, x_n')$ where

$$x_j' = \begin{cases}x_j, &\qquad\text{if } i\neq j \newline \sum\limits_{k\in I} x_k -1, &\qquad\text{if }i=j
\end{cases},$$

The new `state` $S'$ is formed by the vertices of the Newton polyhedron of $\Delta'=\{T_{I,j}(x):x\in S\}$.
- `terminal states`: a state $S$ is terminal if it consists a point $(x_1,\ldots, x_n)$ such that 
$\sum\limits_{i=1}^n x_i \le 1$. 

## The Stratify game 

In 2012 Hauser and Schicho introduced a combinatorial game, called Stratify. It exhibits the axiomatic and logical 
structure of the existing proofs for the resolution of singularities of algebraic varieties in characteristic zero. 
The resolution is typically built on a sequence of blowups in smooth centres which are chosen as the smallest stratum 
of a suitable stratification of the variety. The choice of the stratification and the proof of termination of the 
resolution procedure are both established by induction on the ambient dimension. 

## Thom game

In 2021 Berczi introduced the Thom game, which is a weighted version of the Hironaka game. It has a winning strategy, and
every run of the game provides a blow-up tree, which encodes a formula for Thom polynomials of singularities, answering 
long-standing question in enumerative geometry.

**The rules:**
- `states`: A pair (S,w), where: S is a finite set of points $S \subset \mathbf{N}^n$, such that $S$ is the set of 
vertices of the positive convex hull $\Delta=\{S+\mathbf{R}^n_+\}$; $w=(w_1,\ldots, w_n)\in \mathbf{N}^n$ is a weight 
vector associating a nonnegative integer weight to all coordinates.
- `move`: The host chooses a subset $I\subset \{1,2,\cdots, n\}$ such that $|I|\geq 2$ and 
\sum\limits_{i\in I}x_i\ge 1$ for all $(x_1,\ldots, x_n)\in S$.
The agent chooses an $i\in I$ such that $w_i$ is minimal in $\{w_j: j\in I\}$.
- `state change`: Given the pair (I,i) chosen by the host, for $x=(x_1,\cdots,x_n)\in \mathbb Z^n$ we define
$T_{I,i}(x)=(x_1',\ldots, x_n')$ where

$$x_j' = \begin{cases}x_j, &\qquad\text{if } i\neq j \newline \sum\limits_{k\in I} x_k, &\qquad\text{if }i=j
\end{cases},$$

The new `state` $S'$ is formed by the vertices of the Newton polyhedron of $\Delta'=\{T_{I,i}(x):x\in S\}$, shifted by
a positive integer multiple of $(-1,\ldots, -1)$ such that $S'$ still sits in the positive quadrant, but any
further shift will move it out. 
The new weight vector is

$$w'_j=\begin{cases}w_j, &\qquad\text{if } j=i \quad\text{or}\quad j\notin I \newline w_j-w_i &\qquad\text{if } j \in I\setminus \{i\}
\end{cases},$$

- `terminal states`: a state $S$ is terminal if it consists of one single point, and  

## The Abramovich-Tempkin-Vlodarczyk game

In 2020 Abramovich-Tempkin-Vlodarczyk introduced a new resolution algorithm, based on weighted blow-ups. It significantly 
simplifies the resolution process and uses intrinstic invariants of singularities which improve
after each blow-up. 

# Applications and open questions

## Hilbert scheme of points on manifolds

## Singularity theory and Thom polynomials

## Comparing performance of different winning strategies


## Other problems



## References

[]
