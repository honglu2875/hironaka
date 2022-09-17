from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import mctx
from jax import lax
from mctx import PolicyOutput

from .recurrent_fn import get_recurrent_fn_for_role


def get_evaluation_loop(
        role: str,
        policy_fn: Callable,
        opponent_fn: Callable,
        reward_fn: Callable,
        spec: Tuple,
        num_evaluations: int,
        max_depth: int,
        max_num_considered_actions: int,
        discount: float,
        rescale_points: bool,
        dtype=jnp.float32,
) -> Callable:
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
        num_evaluations: number of evaluations to run in a tree search.
        discount: the discount factor.
        max_depth: maximal depth of the MCTS tree.
        max_num_considered_actions: The maximum number of actions expanded at the root.
        rescale_points: Whether to rescale the point.
        dtype: (Optional) the data type.
    Returns:
        The `evaluation_loop` function who is in charge of evaluation loops (expanding nodes in an MCTS tree and improve
            policy).
        Note: Output is a FrozenDict named "PolicyOutput". `PolicyOutput.action` is the final Gumbel MuZero policy after
            the tree search.
    """
    # Create the `recurrent_fn`
    recurrent_fn = get_recurrent_fn_for_role(
        role, policy_fn, opponent_fn, reward_fn, spec, discount=discount, rescale_points=rescale_points, dtype=dtype
    )
    # Compile the Gumbel MuZero policy function with parameters
    muzero = jax.jit(
        partial(
            mctx.gumbel_muzero_policy,
            recurrent_fn=recurrent_fn,
            num_simulations=num_evaluations,
            max_depth=max_depth,
            max_num_considered_actions=max_num_considered_actions,
        )
    )

    def evaluation_loop(key: jnp.ndarray, root_states: jnp.ndarray, role_fn_args=(),
                        opponent_fn_args=()) -> PolicyOutput:
        """
        Parameters:
            key: The PRNG key.
            root_states: (batch_size, *) a batch of observations as root states.
            role_fn_args: (Optional) parameter for role function
            opponent_fn_args: (Optional) parameter for opponent function
        Returns:
            `PolicyOutput` representing the search outcome.
        """
        policy_prior, value_prior = policy_fn(root_states, *role_fn_args)
        root = mctx.RootFnOutput(prior_logits=policy_prior, value=value_prior, embedding=root_states)
        policy_output = muzero(
            params=(role_fn_args, opponent_fn_args),
            rng_key=key,
            root=root,
        )

        return policy_output

    return evaluation_loop


def get_single_thread_simulation(role: str, evaluation_loop: Callable, config: dict, dtype=jnp.float32):
    """
    A simulation process goes roughly as follows:
        0. Set up initial specs, generate a batch of random points.
        1. Run `evaluation_loop` (get policy priors -> MCTS -> improve policy)
        2. Make one step according to the improved policy and add the (obs, policy_prior, value_prior) into the
            collection of roll-outs.
        3. Repeat 0 until enough roll-outs are collected.
    """
    eval_batch_size, max_num_points, dimension, max_length_game = (
        config["eval_batch_size"],
        config["max_num_points"],
        config["dimension"],
        config["max_length_game"],
    )

    input_dim = max_num_points * dimension if role == "host" else (max_num_points + 1) * dimension
    action_num = 2 ** dimension - dimension - 1 if role == "host" else dimension

    def single_thread_simulation(key: jnp.ndarray, root_state, role_fn_args=(), opponent_fn_args=()) -> Tuple:
        """
        Returns a tuple (obs, policy_prior, value_prior).
        The returning sample size is `eval_batch_size * max_length_game`.
        """
        starting_keys = jax.random.split(key, num=3)

        def body_fn(i, keys_and_state):
            (key, subkey, loop_key), rollouts, state = keys_and_state
            policy_output = evaluation_loop(loop_key, state, role_fn_args=role_fn_args,
                                            opponent_fn_args=opponent_fn_args)

            obs, policy, value = rollouts
            current_obs, current_policy, current_value = (
                state.reshape(eval_batch_size, 1, input_dim),
                policy_output.action_weights.reshape(eval_batch_size, 1, action_num),
                policy_output.search_tree.node_values[:, 0].reshape(eval_batch_size, 1),
            )

            obs = lax.dynamic_update_slice(obs, current_obs, (0, i, 0))
            policy = lax.dynamic_update_slice(policy, current_policy, (0, i, 0))
            value = lax.dynamic_update_slice(value, current_value, (0, i))

            action_idx = jnp.take_along_axis(
                policy_output.search_tree.children_index[:, 0, :], policy_output.action[:, None], axis=1
            )
            state = jnp.take_along_axis(policy_output.search_tree.embeddings[:, :, :], action_idx[:, None],
                                        axis=1).squeeze(1)

            return jax.random.split(key, num=3), (obs, policy, value), state

        num_loops = max_length_game

        # `fori_loop` must return tracers of exactly the same shape
        rollout_obs_init = jnp.zeros((eval_batch_size, num_loops, input_dim))
        rollout_policy_init = jnp.zeros((eval_batch_size, num_loops, action_num))
        rollout_value_init = jnp.zeros(
            (
                eval_batch_size,
                num_loops,
            )
        )

        _, rollouts, _ = lax.fori_loop(
            0, num_loops, body_fn,
            (starting_keys, (rollout_obs_init, rollout_policy_init, rollout_value_init), root_state)
        )

        return rollouts.astype(dtype)

    return single_thread_simulation
