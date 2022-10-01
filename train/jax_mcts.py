from absl import flags
from absl import app
import logging
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp

from hironaka.jax import JAXTrainer
from hironaka.jax.loss import compute_loss
from hironaka.jax.util import select_sample_after_sim

model_path = 'models'

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'jax_mcts.yml', 'The config file.')
flags.DEFINE_int('key', 42, 'The random seed.')
flags.DEFINE_bool('early_stop', False, 'Whether to stop training early when signs of overfitting are observed.')


# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('name', 'Jane Random', 'Your name.')


@partial(jax.pmap, static_broadcasted_argnums=(1, 3), axis_name='d')
def p_compute_loss(params, apply_fn, sample, loss_fn, weight):
    return jax.lax.pmean(compute_loss(params, apply_fn, sample, loss_fn, weight), 'd')


def main(argv):
    key = jax.random.PRNGKey(time.time_ns()) if FLAGS.key is None else FLAGS.key
    key, subkey = jax.random.split(key)

    trainer = JAXTrainer(subkey, FLAGS.config)

    trainer.summary_writer.add_scalar('learning_rate', trainer.config['host']['optim']['args']['learning_rate'], 0)
    trainer.summary_writer.add_scalar('batch_size', trainer.config['host']['batch_size'], 0)
    trainer.load_checkpoint(model_path)

    logger = trainer.logger
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(trainer.config['version_string']))

    logger.info(f"learning rate: {trainer.config['host']['optim']['args']['learning_rate']}")
    logger.info(f"training batch size: {trainer.config['host']['batch_size']}")

    max_num_points, dimension = trainer.max_num_points, trainer.dimension
    dim_difference = 2 ** dimension - 2 * dimension - 1

    for i in range(10000):

        for role in ['host', 'agent']:
            keys = jax.random.split(key, num=len(jax.devices()) + 3)
            key, subkey, test_key = keys[0], keys[1], keys[2]
            device_keys = keys[3:]

            rollout = trainer.simulate(subkey, role, use_unified_tree=True)
            test_set = trainer.simulate(test_key, role, use_unified_tree=True)

            # Size offsets of input shapes for host and agent, respectively.
            offset = -dimension if role == 'host' else None
            p_offset = None if role == 'host' else -dim_difference

            rollout = rollout[0][:, ::2, :], rollout[1][:, ::2], rollout[2][:, ::2]
            test_set = test_set[0][:, ::2, :], test_set[1][:, ::2], test_set[2][:, ::2]

            # Put mask on non-terminal states. `device_keys` are not used since we turn random selection off.
            mask = jax.pmap(select_sample_after_sim, static_broadcasted_argnums=(0, 2, 3))(
                role, rollout, dimension, False, device_keys)
            mask_for_test = jax.pmap(select_sample_after_sim, static_broadcasted_argnums=(0, 2, 3))(
                role, test_set, dimension, False, device_keys)

            # Cutting the observation sizes. (Note: doing it earlier will affect the correctness of masks)
            rollout = rollout[0][..., :offset], rollout[1][..., :p_offset], rollout[2]
            test_set = test_set[0][..., :offset], test_set[1][..., :p_offset], test_set[2]

            apply_fn = trainer.get_apply_fn(role)

            if FLAGS.early_stop:
                # Validation against test set (very slow!!).
                # Agent tends to learn fast and steadily without overfitting.
                # Host is having a lot of troubles and would need a bit of higher max gradient steps.
                num_epoch = 50 if role == 'host' else 10
                prev_test_loss = jnp.inf
                counter = 0  # A counter for testing potential overfitting (when loss is not improving on hold-out set).
                for epoch in range(num_epoch):
                    key, subkey = jax.random.split(key)
                    # Trainer will only sample from masked states (non-terminal game states).
                    # The reason we do not slice arrays directly is that for each device, the numbers will be different.
                    trainer.train(subkey, role, 100, rollout, random_sampling=True, mask=mask, save_best=True)

                    # Validate using test data. It syncs as we aggregate all losses from all devices.
                    # One can choose not to do validation->early stop.
                    test_loss = p_compute_loss(
                        trainer.get_state(role).params, apply_fn, test_set, trainer.loss_fn, mask_for_test)
                    if test_loss[0] >= prev_test_loss[0]:
                        counter += 1
                        if counter == 3:
                            # Early stop after three strikes. Well, 3 is a bit arbitrary.
                            break
                    else:
                        counter = 0

                    prev_test_loss = test_loss
            else:
                key, subkey = jax.random.split(key)
                num_steps = 2000 if role == 'host' else 500
                trainer.train(subkey, role, num_steps, rollout, random_sampling=True, mask=mask, save_best=True)

        if i % 40 == 0:
            trainer.save_checkpoint(model_path)
            logger.info('--------------------')
            logger.info(f'Checkpoint saved at loop {i}.')
            logger.info(f'Best against choose first: {trainer.best_against_choose_first}.')
            logger.info(f'Best against choose last: {trainer.best_against_choose_last}.')
            logger.info(f'Best against zeillinger: {trainer.best_against_zeillinger}.')
            logger.info('--------------------')


if __name__ == '__main__':
    app.run(main)
