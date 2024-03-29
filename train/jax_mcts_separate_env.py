"""
JAX Training script without unified MC search tree. Ran on Google Cloud TPU VM v3-8.
"""

from absl import flags
from absl import app
import logging
import sys
import time
import wandb
from functools import partial

import jax
import jax.numpy as jnp

from hironaka.jax import JAXTrainer
from hironaka.jax.loss import compute_loss
from hironaka.jax.util import select_sample_after_sim


FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'jax_mcts.yml', 'The config file.')
flags.DEFINE_string('model_path', 'models', 'The model path.')
flags.DEFINE_integer('key', 42, 'The random seed.')
flags.DEFINE_bool('early_stop', False, 'Whether to stop training early when signs of overfitting are observed.')
flags.DEFINE_bool('use_mask', False, 'Whether to mask the rollout and only look at non-terminal states.')


@partial(jax.pmap, static_broadcasted_argnums=(1, 3), axis_name='d')
def p_compute_loss(params, apply_fn, sample, loss_fn, weight):
    return jax.lax.pmean(compute_loss(params, apply_fn, sample, loss_fn, weight), 'd')


def main(argv):
    key = jax.random.PRNGKey(time.time_ns() if FLAGS.key is None else FLAGS.key)
    key, subkey = jax.random.split(key)

    trainer = JAXTrainer(subkey, FLAGS.config)

    wandb.log({'learning_rate': trainer.config['host']['optim']['args']['learning_rate'],
               'batch_size': trainer.config['host']['batch_size']})
    trainer.load_checkpoint(FLAGS.model_path)

    logger = trainer.logger
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(trainer.version_string + '.log'))

    logger.info(f"learning rate: {trainer.config['host']['optim']['args']['learning_rate']}")
    logger.info(f"training batch size: {trainer.config['host']['batch_size']}")

    max_num_points, dimension = trainer.max_num_points, trainer.dimension
    dim_difference = 2 ** dimension - 2 * dimension - 1

    for i in range(10000):

        for role in ['host', 'agent']:
            keys = jax.random.split(key, num=len(jax.devices()) + 3)
            key, subkey, test_key = keys[0], keys[1], keys[2]
            device_keys = keys[3:]

            rollout = trainer.simulate(subkey, role, use_unified_tree=False)

            logger.info(f"{role} rollout finished.")
            #logger.info(f"Non-terminal states/number of all samples: {jnp.sum(mask)}/{rollout[0].shape[0] * rollout[0].shape[1]}")
            logger.info(f"Value dist: {jnp.histogram(rollout[2])}")

            key, subkey = jax.random.split(key)
            num_steps = 2000 if role == 'host' else 500
            trainer.train(subkey, role, num_steps, rollout, random_sampling=True, save_best=True)

        if i % 10 == 0:
            trainer.save_checkpoint(FLAGS.model_path)
            logger.info('--------------------')
            logger.info(f'Checkpoint saved at loop {i}.')
            logger.info(f'Best against choose first: {trainer.best_against_choose_first}.')
            logger.info(f'Best against choose last: {trainer.best_against_choose_last}.')
            logger.info(f'Best against zeillinger: {trainer.best_against_zeillinger}.')
            logger.info('--------------------')


if __name__ == '__main__':
    app.run(main)
