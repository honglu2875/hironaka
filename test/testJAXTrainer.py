import pathlib
import unittest

import jax

from hironaka.jax import JAXTrainer


class TestJAXTrainer(unittest.TestCase):
    def test_trainer(self):
        key = jax.random.PRNGKey(42)
        trainer = JAXTrainer(key, str(pathlib.Path(__file__).parent.resolve()) + "/jax_config.yml")
        for role in ['host', 'agent']:
            key, subkey = jax.random.split(key)
            exp = trainer.simulate(subkey, role)
            trainer.train(subkey, role, 100, exp, random_sampling=True)
