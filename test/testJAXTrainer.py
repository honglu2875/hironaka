import os
import pathlib
import unittest
from functools import partial

import flax
import numpy as np
from flax.training.train_state import TrainState

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import jax
import jax.numpy as jnp
from hironaka.jax import JAXTrainer
from hironaka.jax.util import select_sample_after_sim, action_wrapper


def same_dict(d1, d2) -> bool:
    try:
        if isinstance(d1, (dict, flax.core.FrozenDict)):
            for key in d1:
                if not same_dict(d1[key], d2[key]):
                    return False
        elif isinstance(d1, (jnp.ndarray, np.ndarray)):
            return jnp.all(d1 == d2)
        else:
            return d1 == d2
    except:
        return False
    return True


class TestJAXTrainer(unittest.TestCase):
    trainer = JAXTrainer(jax.random.PRNGKey(42), str(pathlib.Path(__file__).parent.resolve()) + "/jax_config.yml")

    def test_training_and_save_load(self):
        # Why can one not be able to force mock devices on GitHub's testing machine?
        # assert len(jax.devices()) == 2
        key = jax.random.PRNGKey(42)
        for role in ["host", "agent"]:
            keys = jax.random.split(key, num=len(jax.devices()) + 2)
            key, subkey = keys[0], keys[1]
            device_keys = keys[2:]

            # Test both simplified simulation and mcts simulation
            exp = self.trainer.simulate(subkey, role)
            exp = self.trainer.simulate(subkey, role, use_mcts_policy=True)

            assert exp[0].shape[1] == self.trainer.eval_batch_size * self.trainer.max_length_game
            # Test the post-selection of rollouts (prevent the dataset from being dominated by terminal states)
            if role == "agent":
                # Assert all invalid agent actions have probability 0.0
                assert jnp.all(exp[0][0, :, -3:] - exp[1][0, :, :] >= 0)
            mask = jax.pmap(select_sample_after_sim, static_broadcasted_argnums=(0, 2, 3))(role, exp, 3, True, device_keys)
            self.trainer.train(subkey, role, 10, exp, random_sampling=True, mask=mask)

        original_state = {"host": self.trainer.host_state, "agent": self.trainer.agent_state}
        # Make sure they are immutable -> unchanged after trainings
        assert isinstance(original_state["host"], TrainState)
        assert isinstance(original_state["agent"], TrainState)

        # Save checkpoints.
        for path in ["runs/test/host_10", "runs/test/agent_10"]:
            if os.path.exists(path):
                os.remove(path)
        self.trainer.save_checkpoint("runs/test")
        for path in ["runs/test/host_10", "runs/test/agent_10"]:
            assert os.path.exists(path)

        for role in ["host", "agent"]:
            key, subkey = jax.random.split(key)
            exp = self.trainer.simulate(subkey, role)
            self.trainer.train(subkey, role, 10, exp, random_sampling=True)

        self.trainer.load_checkpoint("runs/test")
        for role in ["host", "agent"]:
            assert same_dict(original_state[role].params, self.trainer.get_state(role).params)
            # Optimizer needs to be matched too, as it is supposed to be a feature of `restore_checkpoint` with `target`
            assert same_dict(original_state[role].tx, self.trainer.get_state(role).tx)

    def test_rollout_postprocess(self):
        rollout = (
            jnp.array(
                [
                    [
                        [
                            0.7777778,
                            0.05555556,
                            1.0,
                            0.16666667,
                            0.22222222,
                            0.22222222,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                        [
                            0.9333333,
                            0.06666666,
                            1.0,
                            0.19999999,
                            0.26666665,
                            0.46666664,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                        [
                            0.9333334,
                            0.06666667,
                            1.0,
                            0.20000002,
                            0.26666668,
                            0.4666667,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                        [
                            0.9333333,
                            0.06666666,
                            1.0,
                            0.19999999,
                            0.26666665,
                            0.46666664,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                        [
                            0.9333334,
                            0.06666667,
                            1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                    ],
                    [
                        [
                            0.16666667,
                            0.16666667,
                            0.8333333,
                            0.8888889,
                            0.7777778,
                            0.05555556,
                            0.16666667,
                            1.0,
                            0.0,
                            0.44444445,
                            0.05555556,
                            0.44444445,
                            0.05555556,
                            0.22222222,
                            0.6666667,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        [
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            0.14285716,
                            1.0,
                            0.0,
                            0.38095242,
                            0.8095239,
                            0.38095242,
                            0.04761905,
                            0.8095239,
                            0.57142866,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        [
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            0.7272727,
                            0.63636357,
                            0.0,
                            1.0,
                            0.5151515,
                            0.24242423,
                            0.9090909,
                            0.5151515,
                            0.36363637,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        [
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            0.775862,
                            0.36206892,
                            0.0,
                            0.99999994,
                            0.29310343,
                            0.13793102,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                        [
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            -1.0,
                            0.88,
                            0.28,
                            0.0,
                            1.0,
                            0.22666667,
                            0.10666667,
                            -1.0,
                            -1.0,
                            -1.0,
                            1.0,
                            1.0,
                            0.0,
                        ],
                    ],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [
                        [9.3919918e-02, 2.0432323e-03, 9.0403688e-01],
                        [2.7057916e-01, 1.4561049e-03, 7.2796470e-01],
                        [2.7057916e-01, 1.4561049e-03, 7.2796470e-01],
                        [2.7057916e-01, 1.4561049e-03, 7.2796470e-01],
                        [2.7057916e-01, 1.4561049e-03, 7.2796470e-01],
                    ],
                    [
                        [9.8180562e-01, 1.7476840e-02, 7.1753026e-04],
                        [5.1261973e-01, 4.8487249e-01, 2.5077474e-03],
                        [9.1848236e-01, 9.6930191e-04, 8.0548234e-02],
                        [2.4517733e-01, 2.4674647e-03, 7.5235522e-01],
                        [3.0273062e-01, 2.4294229e-03, 6.9483995e-01],
                    ],
                ],
                dtype=jnp.float32,
            ),
            jnp.array(
                [
                    [0.0622959, 0.1566598, 0.15665981, 0.1566598, 0.15665981],
                    [0.14874144, -0.00777596, -0.0012398, 0.00597944, -0.04718201],
                ],
                dtype=jnp.float32,
            ),
        )

        v = jnp.array(
            [-0.960596, -0.970299, -0.9801, -0.98999995, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32
        )  # The first game ended, resulting in discounted reward (penalty)
        assert jnp.all(self.trainer.rollout_postprocess(rollout, "agent")[2] - v < 1e-6)

    def test_mcts_policy_fns(self):
        orig_size = self.trainer.eval_batch_size
        # Drop the burden for the test
        self.trainer.eval_batch_size = 10
        self.trainer.update_fns('host')
        self.trainer.update_fns('agent')

        host = action_wrapper(partial(jax.pmap(self.trainer.host_mcts_policy_fn),
                                      params=self.trainer.host_state.params, opp_params=self.trainer.agent_state.params))
        agent = action_wrapper(partial(jax.pmap(self.trainer.agent_mcts_policy_fn),
                                       params=self.trainer.agent_state.params, opp_params=self.trainer.host_state.params))

        rho, details = self.trainer.compute_rho(host, agent)
        print(rho, details)

        self.trainer.eval_batch_size = orig_size
        self.trainer.update_fns('host')
        self.trainer.update_fns('agent')