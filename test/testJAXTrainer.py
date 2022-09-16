import pathlib
import unittest
from functools import partial

import mctx
import torch
import jax
import jax.numpy as jnp

from hironaka.core import TensorPoints, JAXPoints
from hironaka.jax import JAXTrainer
from hironaka.jax.net import DResNet18, PolicyWrapper, DResNetMini
from hironaka.jax.players import all_coord_host_fn, random_host_fn, \
    char_vector_of_pts, zeillinger_fn_slice, zeillinger_fn, random_agent_fn, choose_first_agent_fn, choose_last_agent_fn
from hironaka.jax.recurrent_fn import get_recurrent_fn_for_role
from hironaka.jax.simulation_fn import get_evaluation_loop, get_single_thread_simulation
from hironaka.jax.util import flatten, make_agent_obs, get_take_actions, get_reward_fn, decode_table, \
    decode_from_one_hot, \
    batch_encode, batch_encode_one_hot, get_batch_decode_from_one_hot, \
    get_batch_decode, get_preprocess_fns, apply_agent_action_mask, get_feature_fn

from hironaka.src import get_newton_polytope_torch, shift_torch, reposition_torch, remove_repeated
from hironaka.src import get_newton_polytope_jax, shift_jax, rescale_jax, reposition_jax


class TestJAXTrainer(unittest.TestCase):
    def test_trainer(self):
        key = jax.random.PRNGKey(42)
        trainer = JAXTrainer(key, str(pathlib.Path(__file__).parent.resolve()) + "/jax_config.yml")
        for role in ['host', 'agent']:
            key, subkey = jax.random.split(key)
            exp = trainer.simulate(subkey, role)
            trainer.train(subkey, role, 10, exp, random_sampling=True)
