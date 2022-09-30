import time
import sys
import jax
import yaml
import logging

from hironaka.jax import JAXTrainer
import logging
import sys
import time

import jax
import jax.numpy as jnp
import yaml

from hironaka.jax import JAXTrainer
from hironaka.jax.loss import compute_loss
from hironaka.jax.util import select_sample_after_sim

model_path = 'models'


def main(key=None):
    key = jax.random.PRNGKey(time.time_ns()) if key is None else key
    key, subkey = jax.random.split(key)

    trainer = JAXTrainer(subkey, "jax_mcts.yml")

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

    for i in range(10000):

        for role in ['host', 'agent']:
            keys = jax.random.split(key, num=len(jax.devices()) + 3)
            key, subkey, test_key = keys[0], keys[1], keys[2]
            device_keys = keys[3:]

            rollout = trainer.simulate(subkey, role, use_unified_tree=True)
            test_set = trainer.simulate(test_key, role, use_unified_tree=True)

            offset = dimension if role == 'host' else 0
            num_epoch = 50 if role == 'host' else 10

            rollout = rollout[0][:, ::2, :], rollout[1][:, ::2], rollout[2][:, ::2]
            test_set = test_set[0][:, ::2, :], test_set[1][:, ::2], test_set[2][:, ::2]

            # put mask on non-terminal states. `device_keys` are not used since we turn random selection off.
            mask = jax.pmap(select_sample_after_sim, static_broadcasted_argnums=(0, 2, 3))(
                role, rollout, dimension, False, device_keys)
            mask_for_test = jax.pmap(select_sample_after_sim, static_broadcasted_argnums=(0, 2, 3))(
                role, test_set, dimension, False, device_keys)

            # Cutting the observation. (Note: doing it earlier will affect the correctness of masks)
            rollout = rollout[0][...:-offset], rollout[1], rollout[2]
            test_set = test_set[0][...:-offset], test_set[1], test_set[2]

            apply_fn = trainer.get_apply_fn(role)

            prev_test_loss = jnp.inf
            counter = 0  # Counter for potential overfitting (when loss is not improving on hold-out set).
            for epoch in range(num_epoch):
                key, subkey = jax.random.split(key)
                # Trainer will only sample from masked states (non-terminal game states).
                # The reason we do not slice arrays directly is that for each device, the numbers will be different.
                trainer.train(subkey, role, 100, rollout, random_sampling=True, mask=mask, save_best=True)

                # Validate using test data. It syncs as we aggregate all losses from all devices.
                # One can choose not to do validation->early stop.
                test_loss = jnp.mean(jax.pmap(compute_loss, static_broadcasted_argnums=(1, 3))(
                    trainer.get_state(role).params, apply_fn, test_set, trainer.loss_fn, mask_for_test))
                if test_loss >= prev_test_loss:
                    counter += 1
                    if counter == 3:
                        # Early stop after three strikes. Well, 3 is a bit arbitrary.
                        break
                else:
                    counter = 0

                prev_test_loss = test_loss

        if i % 40 == 0:
            trainer.save_checkpoint(model_path)


if __name__ == '__main__':
    main()
