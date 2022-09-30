import os

from setuptools import find_packages, setup

with open(os.path.join("hironaka", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

long_description = """

# Hironaka

A utility package for a reinforcement learning study of singularities in algebraic geometry. Algebraic geometry and machine learning could meet the following way:

From the mathematics side:

- A range of Algebraic Geometry problems => study singularities => understand the resolutions of singularities => reduce to a game similar to Hironaka game.

From the machine learning side:

- Reinforcement Learning => solve 2-player games (Markov Decision Processes) => solve Hironaka game and its variants.

We study this connection and refactor our codes into this repo.


## Quick example

- (TL;DR, clone
  this [Google Colab file](https://colab.research.google.com/drive/1nVnVA6cyg0GT5qTadJTJH7aU6smgopLm?usp=sharing),
  forget what I say below and start your adventure)

  `dqn_trainer` is a quick implementation combining my interface `Trainer` with `stable-baseline3`'s DQN codes. It runs
  in 3 lines:
    ```python
    from hironaka.trainer.dqn_trainer import dqn_trainer
    trainer = dqn_trainer('dqn_config_test.yml')
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


"""  # noqa:E501

setup(
    name="hironaka",
    packages=[package for package in find_packages() if package.startswith("hironaka")],
    package_data={"hironaka": ["version.txt"]},
    install_requires=[
        "absl-py",
        "numpy",
        "treelib",
        "gym==0.21",
        "torch>=1.12.0",
        "scipy",
        "PyYAML",
        "sympy",
        "stable-baselines3",
        "jax",
        "jaxlib",
        "flax>=0.6.0",
        "mctx>=0.0.2",
        "chex>=0.1.4",
        "optax>=0.1.3",
        "tensorboard",
        "tensorflow-cpu",
    ],
    extras_require={
        "tests": [
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
        "extra": [
            "tensorflow",
        ],
    },
    description="A utility package for a reinforcement learning study of singularities in algebraic geometry",
    author="Gergely Berczi, Honglu Fan, Mingcong Zeng",
    url="https://github.com/honglu2875/hironaka",
    author_email="honglu.math@gmail.com",
    keywords="reinforcement-learning machine-learning pure-mathematics algebraic geometry"
    "mcts deep-q-learning alphazero python",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
