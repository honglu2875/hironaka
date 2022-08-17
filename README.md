# Hironaka

A utility package for reinforcement learning study of Hironaka's game of local resolution of singularities and its
variation problems.

# Quick start

[This notebook](https://cocalc.com/share/public_paths/5db3252a0bcb8d068aad2ee53bf5a1ce85753ebf) provides a brief
demonstration of key classes in this repo. It is highly recommended to take a look first if you are an example-oriented
learner.

# Contents

For ML and RL specialists, the following two sections should give you an overview:

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

......

---
Now the big question is, how is this game related to some fundamental questions in pure math, or specifically, algebraic
geometry.

# What is a resolution of singularity

An affine algebraic variety  
$$X = \{(x_1,\ldots, x_n): f_1(x_1,\ldots, x_n)=\ldots =f_k(x_1,\ldots, x_n)=0\} \subset A^n$$
is the common zero locus of polynomial equations. Affine varieties play central role in mathematics, physics and biology.   

Affine varieties cut out by one polynomial equation are called affine hypersurfaces. E.g
$$X=\{\}$$


## Singularities

[definition] & [examples]

We can think of varieties as "shapes in affine spaces", and at a generic point 
$x \in X$ the variety locally is $A^r$ for some $r$, which we call the dimension of $X$.
However, there are special, ill-behaved points, where the local geometry of $X$ is less patent.

The set $X$ is singular at a point $a \in X$ if the Jacobian matrix
$$Jac(X,a)=\left(\partial f_i \partial x_j\right)(a)$$
at a is of rank smaller than $n-dim(X)$. The set of singular points of $X$ is called the singular locus of $X$.




## Blow-up: turning singularities into smooth points

Resolution of singularities is a classical central problem in geometry. By resolution we mean that we substitute the original,
possibly singular $X$ with a nonsingular $Y$ with a proper 
birational map $f:Y \to X$ such that $f$ is an isomorphism over some open dense subset of X.

The celebrated Hironaka theorem asserts that such resolution exists for all $X$, and it can be constructed from an 
elementary type of operation called blowing up. Blowing up or blowup is a type of geometric transformation which 
replaces a subspace of a given space  with all the directions pointing out of that subspace. 

For example, the blowup of a point in a plane replaces the point 
with the projectivized tangent space at that point...
[examples]

## Hironaka's theorem

The most famous and general result concerning resolution of singularities was given by Heisuke Hironaka in 1964 
He proved that the resolution of singularities can be achieved by a sequence of blowups 
\[Y=X_n \to X_{n-1} \to \ldots \to X_0=X\]
if the characteristic of the base field is zero.

This beautiful and fundamental work was recognized with a Fields medal in 1970. Recently, Villamayor and, independently, 
Bierstone and Milman have clarified the process of resolution of singularities in characteristic zero, 
explicitly describing the algorithmic nature of the resolution process. 

Using de Jong's deep ideas, simple proofs of Hironaka's theorem have been discovered by de Jong and Abramovich and by 
Bogomolov and Pantev.




# What is Hironaka's polyhedral game

## Monomial ideals

[definition] & [examples]

## Rephrasing the local resolution of singularity problem

[definitions]

# Variations of Hironaka's game

## Spivakovsky game

## Hauser game 

## Bloch-Levine game for moving cycles

## Thom game

# Applications and open questions

## Singularity theory and Thom polynomials

Geometric invariant theory is the algebraic set-up to construct quotients of algebraic varieties by algebraic group 
actions. A key technical conditions for GIT is called the semistability=stability condition. When this fails, there is 
a blow-up algorithm to prepare our space for GIT. When the acting group is a non-reductive reparametrisation group, the 
GIT blow-up procedure can be interpreted using the Hironaka game.

## Comparing performance of different winnig strategies

## Other problems

From "FORTY QUESTIONS ON SINGULARITIES OF ALGEBRAIC VARIETIES" by Hauser and Schicho

...
