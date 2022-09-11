import warnings
from functools import partial
from typing import Callable, Tuple

import jax
import mctx
from jax import jit, lax
import jax.numpy as jnp
from mctx import PolicyOutput

from hironaka.src import get_newton_polytope_jax, rescale_jax, shift_jax
from .recurrent_fn import get_recurrent_fn_for_role
from .util import flatten


def get_evaluation_loop(role: str, policy_fn: Callable, opponent_fn: Callable, reward_fn: Callable,
                        spec: Tuple, num_simulations: int, max_depth: int,
                        max_num_considered_actions: int, dtype=jnp.float32) -> Callable:
    """
    The factory function of `evaluation_loop` which creates a new node and do one single (batched) MCTS search using
        Gumbel MuZero policy.
    Parameters:
        role: 'host' or 'agent'
        policy_fn: the policy function.
         - parameters: observations: jnp.ndarray
         - returns: Tuple of policy_prior, value_prior
        opponent_fn: the opponent decision function.
         - parameters: observations: jnp.ndarray
         - returns: a batch of one-hot vectors of host/agent actions
        reward_fn: the reward function.
         - parameters:
            dones, jnp.ndarray
            prev_dones, jnp.ndarray
         - returns: a batch of rewards
        spec: (max_num_points, dimension)
        num_simulations: number of simulations to run in a tree search.
        max_depth: maximal depth of the MCTS tree.
        max_num_considered_actions: The maximum number of actions expanded at the root.
        dtype: (Optional) the data type.
    Returns:
        The `evaluation_loop` function who is in charge of evaluation loops (expanding nodes in an MCTS tree and improve
            policy).
        Note: Output is a FrozenDict named "PolicyOutput". `PolicyOutput.action` is the final Gumbel MuZero policy after
            the tree search.
    """
    # Create the `recurrent_fn`
    recurrent_fn = get_recurrent_fn_for_role(role, policy_fn, opponent_fn, reward_fn, spec, dtype=dtype)
    # Compile the Gumbel MuZero policy function with parameters
    muzero = jax.jit(partial(mctx.gumbel_muzero_policy, params=(), recurrent_fn=recurrent_fn,
                             num_simulations=num_simulations, max_depth=max_depth,
                             max_num_considered_actions=max_num_considered_actions))

    @jit
    def evaluation_loop(key: jnp.ndarray, root_states: jnp.ndarray) -> PolicyOutput:
        """
        Parameters:
            key: The PRNG key.
            root_states: (batch_size, *) a batch of observations as root states.
        Returns:
            `PolicyOutput` representing the search outcome.
        """
        policy_prior, value_prior = policy_fn(root_states)
        root = mctx.RootFnOutput(prior_logits=policy_prior, value=value_prior, embedding=root_states)
        policy_output = muzero(
            rng_key=key,
            root=root,
        )

        return policy_output

    return evaluation_loop


def get_single_thread_simulation(starting_key: jnp.ndarray, evaluation_loop: Callable, rollout_size: int, config: dict,
                                 dtype=jnp.float32):
    """
    A simulation process goes roughly as follows:
        0. Set up initial specs, generate a batch of random points.
        1. Run `evaluation_loop` (get policy priors -> MCTS -> improve policy)
        2. Make one step according to the improved policy and add the (obs, policy_prior, value_prior) into the
            collection of roll-outs.
        3. Repeat 0 until enough roll-outs are collected.
    """
    eval_batch_size, max_num_points, dimension = \
        config['eval_batch_size'], config['max_num_points'], config['dimension']
    max_value = config['max_value']
    rescale_fn = (lambda x: x) if config['scale_observation'] else rescale_jax
    if rollout_size % eval_batch_size != 0:
        warnings.warn(f"rollout_size cannot be divided by eval_batch_size. "
                      f"Output batch size may be different than rollout_size.")

    # Warm up and get the action numbers
    out = evaluation_loop(jax.random.PRNGKey(0), jnp.zeros((eval_batch_size, max_num_points * dimension)))
    action_num = out.action_weights.shape[1]

    @jit
    def single_thread_simulation() -> Tuple:
        """
        Returns a tuple (obs, policy_prior, value_prior). Batch sizes of all of them are `rollout_size`.
        """
        starting_keys = jax.random.split(starting_key, num=3)

        root_state = flatten(rescale_fn(get_newton_polytope_jax(
            jax.random.randint(starting_keys[1], (eval_batch_size, max_num_points, dimension),
                               0, max_value).astype(dtype))))

        def body_fn(i, keys_and_state):
            (key, subkey, loop_key), experiences, state = keys_and_state
            policy_output = evaluation_loop(loop_key, state)

            obs, policy, value = experiences
            obs = lax.dynamic_update_slice(obs, root_state, (i * eval_batch_size, 0))
            policy = lax.dynamic_update_slice(policy, policy_output.action_weights, (i * eval_batch_size, 0))
            value = lax.dynamic_update_slice(value, policy_output.search_tree.node_values[:, 0], (i * eval_batch_size,))

            action_idx = jnp.take_along_axis(policy_output.search_tree.children_index[:, 0, :],
                                             policy_output.action[:, None], axis=1)
            state = jnp.take_along_axis(policy_output.search_tree.embeddings[:, :, :],
                                        action_idx[:, None], axis=1).squeeze(1)

            return jax.random.split(key, num=3), (obs, policy, value), state

        num_loops = ((rollout_size + eval_batch_size - 1) // eval_batch_size)

        rollout_obs_init = jnp.zeros((eval_batch_size * num_loops, max_num_points * dimension))
        rollout_policy_init = jnp.zeros((eval_batch_size * num_loops, action_num))
        rollout_value_init = jnp.zeros((eval_batch_size * num_loops,))

        # `fori_loop` must return tracers of exactly the same shape
        _, experiences, _ = lax.fori_loop(0, num_loops, body_fn,
                                          (starting_keys, (rollout_obs_init,
                                                           rollout_policy_init,
                                                           rollout_value_init), root_state))

        rollout_obs, rollout_policy_prior, rollout_value_prior = experiences
        return jnp.array(rollout_obs), jnp.array(rollout_policy_prior), jnp.array(rollout_value_prior)

    return single_thread_simulation


