import unittest
from functools import partial

import flax
import mctx

import jax
import jax.numpy as jnp
from flax.jax_utils import replicate, unreplicate
from hironaka.core import JAXPoints
from hironaka.jax import JAXTrainer
from hironaka.jax.loss import policy_value_loss
from hironaka.jax.net import DResNetMini, get_apply_fn
from hironaka.jax.players import (
    all_coord_host_fn,
    choose_first_agent_fn,
    choose_last_agent_fn,
    random_agent_fn,
    random_host_fn,
    zeillinger_fn,
    zeillinger_fn_slice, get_host_with_flattened_obs,
)
from hironaka.jax.recurrent_fn import get_recurrent_fn_for_role, get_unified_recurrent_fn
from hironaka.jax.simulation_fn import get_evaluation_loop, get_simulation
from hironaka.jax.util import (
    apply_agent_action_mask,
    flatten,
    generate_pts,
    get_feature_fn,
    get_preprocess_fns,
    get_reward_fn,
    get_take_actions,
    make_agent_obs,
    rollout_sanity_tests
)
from hironaka.jax.host_action_preprocess import decode_table, get_batch_decode, get_batch_decode_from_one_hot, \
    batch_encode, batch_encode_one_hot
from hironaka.src import get_newton_polytope_jax, reposition_jax, rescale_jax, shift_jax


class TestJAX(unittest.TestCase):
    r = jnp.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [-1.0, -1.0, -1.0, -1.0], [4.0, 1.0, 2.0, 3.0], [1.0, 6.0, 7.0, 3.0]],
            [[0.0, 1.0, 3.0, 5.0], [1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
        ]
    )

    r2 = jnp.array(
        [
            [[1.0, 5.0, 3.0, 4.0], [-1.0, -1.0, -1.0, -1.0], [4.0, 3.0, 2.0, 3.0], [1.0, 13.0, 7.0, 3.0]],
            [[0.0, 1.0, 3.0, 8.0], [1.0, 1.0, 1.0, 3.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
        ]
    )

    r3 = jnp.array(
        [
            [[0.0, 2.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0], [3.0, 0.0, 0.0, 0.0], [0.0, 10.0, 5.0, 0.0]],
            [[0.0, 0.0, 2.0, 5.0], [1.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
        ]
    )

    rs = jnp.array(
        [
            [
                [0.0000, 0.2000, 0.1000, 0.1000],
                [-1.0000, -1.0000, -1.0000, -1.0000],
                [0.3000, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.5000, 0.0000],
            ],
            [
                [0.0000, 0.0000, 0.4000, 1.0000],
                [0.2000, 0.0000, 0.0000, 0.0000],
                [-1.0000, -1.0000, -1.0000, -1.0000],
                [-1.0000, -1.0000, -1.0000, -1.0000],
            ],
        ]
    )

    def test_functions(self):
        p = jnp.array(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]],
            ]
        ).astype(jnp.float32)
        extreme = jnp.array([[[1, 1, 1], [1, 1, 1]]]).astype(jnp.float32)
        extreme_a = jnp.array([[[1, 1, 1], [-1, -1, -1]]]).astype(jnp.float32)
        s = jnp.array([[[1, 2, 3], [2, 3, 4], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [-1, -1, -1]]]).astype(float)
        sr = jnp.array(
            [
                [[0.25, 0.5, 0.75], [0.5, 0.75, 1.0], [-1.0, -1.0, -1.0]],
                [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            ]
        )
        assert jnp.all(get_newton_polytope_jax(p) == self.r)
        assert jnp.all(get_newton_polytope_jax(extreme) == extreme_a)
        p = shift_jax(p, jnp.array([[0, 1, 1, 0], [1, 0, 1, 1]]), jnp.array([1, 3]))
        assert jnp.all(get_newton_polytope_jax(p) == self.r2)
        assert jnp.all(rescale_jax(s) == sr)
        assert jnp.all(reposition_jax(get_newton_polytope_jax(p)) == self.r3)
        s = jnp.array(
            [
                [
                    [0.10526316, 0.2631579, 0.0],
                    [0.0, 0.8947368, 0.15789473],
                    [-1.0, -1.0, -1.0],
                    [0.10526316, 0.21052632, 1.0],
                    [-1.0, -1.0, -1.0],
                    [0.84210527, 0.05263158, 0.47368422],
                ]
            ]
        )
        s_sorted = jnp.array(
            [
                [
                    0.10526316, 0.21052632, 1.0,
                    0.84210527, 0.05263158, 0.47368422,
                    0.0, 0.8947368, 0.15789473,
                    0.10526316, 0.2631579, 0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            ]
        )
        feature_fn = get_feature_fn("host", (6, 3))
        assert jnp.all(feature_fn(s) == s_sorted)
        sr = jnp.array(
            [
                [
                    [0.10526316, 0.2631579, 0.0],
                    [0.0, 1.0526316, 0.15789473],
                    [-1.0, -1.0, -1.0],
                    [0.10526316, 1.2105263, 1.0],
                    [-1.0, -1.0, -1.0],
                    [0.84210527, 0.5263158, 0.47368422],
                ]
            ]
        )
        assert jnp.all(shift_jax(s, jnp.array([[0, 1, 1]]), jnp.array([1])) == sr)

        # An additional test for agent features
        p = jnp.array([[1, 2, 3, -1, -1, -1, 2, 3, 4, 0, 1, 1], [-1, -1, -1, 0, 0, 1, 0, 1, 1, 1, 0, 1]], dtype=jnp.float32)
        sp = jnp.array([[2, 3, 4, 1, 2, 3, -1, -1, -1, 0, 1, 1], [0, 1, 1, 0, 0, 1, -1, -1, -1, 1, 0, 1]], dtype=jnp.float32)
        feature_fn = get_feature_fn("agent", (3, 3), scale_observation=False)
        assert jnp.all(feature_fn(p) == sp)
        feature_fn = get_feature_fn("agent", (3, 3), scale_observation=True)
        sp = jnp.array([[0.5, 0.75, 1., 0.25, 0.5, 0.75, -1, -1, -1, 0, 1, 1], [0, 1, 1, 0, 0, 1, -1, -1, -1, 1, 0, 1]], dtype=jnp.float32)
        assert jnp.all(feature_fn(p) == sp)

    def test_points_jax(self):
        p = jnp.array(
            [
                [[1, 2, 3, 4], [2, 3, 4, 5], [4, 1, 2, 3], [1, 6, 7, 3]],
                [[0, 1, 3, 5], [1, 1, 1, 1], [9, 8, 2, 1], [-1, -1, -1, -1]],
            ]
        )

        pts = JAXPoints(p)

        pts.get_newton_polytope()
        assert str(pts) == str(self.r)
        pts.shift(jnp.array([[0, 1, 1, 0], [1, 0, 1, 1]]), jnp.array([1, 3]))
        assert str(pts) == str(self.r2)
        pts.reposition()
        assert str(pts) == str(self.r3)
        pts.rescale()
        assert jnp.all(jnp.isclose(pts.points, self.rs))

    # JAXObs class was deprecated. It was used to hold both host and agent observations.
    """
    def test_jax_obs(self):
        host_obs = jnp.array([
            [[1, 2, 3], [2, 3, 4], [0, 1, 0], [-1, -1, -1]],
            [[4, 2, 2], [-1, -1, -1], [0, 0, 1], [-1, -1, -1]]
        ]).astype(jnp.float32)
        agent_obs = {'points': jnp.copy(host_obs),
                     'coords': jnp.array([[0, 1, 1], [1, 1, 1]])}
        combined = jnp.concatenate([agent_obs['points'].reshape(2, -1), agent_obs['coords']],
                                   axis=1)

        h_o = JAXObs('host', host_obs)
        assert jnp.all(h_o.get_features() == host_obs.reshape(2, -1))
        assert jnp.all(h_o.get_points() == host_obs)
        assert h_o.get_coords() is None
        a_o = JAXObs('agent', agent_obs)
        assert jnp.all(a_o.get_features() == combined)
        assert jnp.all(a_o.get_points() == host_obs)
        assert jnp.all(a_o.get_coords() == agent_obs['coords'])
        a_o2 = JAXObs('agent', combined, dimension=3)
        assert jnp.all(a_o2.get_features() == combined)
        assert jnp.all(a_o2.get_points() == host_obs)
        assert jnp.all(a_o2.get_coords() == agent_obs['coords'])

        with self.assertRaises(Exception) as context:
            a_o = JAXObs('agent', host_obs)
        with self.assertRaises(Exception) as context:
            a_o = JAXObs('agent', host_obs, dimension=7)
    """

    def test_encode_decode(self):
        decode_3 = jnp.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        assert jnp.all(decode_table(3) == decode_3)
        encode_in = jnp.array([[1, 0, 1], [1, 1, 1], [0, 1, 1]])
        encode_out = jnp.array([1, 3, 2])
        encode_one_hot_out = jnp.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        # Construct dimension-3 batch decoders
        batch_decode_from_one_hot = get_batch_decode_from_one_hot(3)
        batch_decode = get_batch_decode(3)

        assert jnp.all(batch_encode(encode_in) == jnp.array(encode_out))
        assert jnp.all(batch_encode_one_hot(encode_in) == jnp.array(encode_one_hot_out))
        assert jnp.all(batch_decode_from_one_hot(encode_one_hot_out) == encode_in)
        assert jnp.all(batch_decode(encode_out) == encode_in)

    def test_loss(self):
        pi = jnp.array([[jnp.exp(1), jnp.exp(2)], [jnp.exp(3), jnp.exp(4)]])
        v = jnp.array([5.0, 6.0])
        t_pi = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        t_v = jnp.array([1.0, -1.0])
        assert jnp.isclose(policy_value_loss(pi, v, t_pi, t_v), 8.676451)
        # Masked. They add up the the sum above.
        assert jnp.isclose(policy_value_loss(pi, v, t_pi, t_v, jnp.array([1.0, 0.0])), 2.8559828)
        assert jnp.isclose(policy_value_loss(pi, v, t_pi, t_v, jnp.array([0.0, 1.0])), 5.820468)


    def test_hosts(self):
        obs = jnp.array([[[1, 2, 3], [2, 3, 4]], [[0, 1, 2], [-1, -1, -1]]])
        all_coord_out = jnp.array([[[0, 0, 0, 1], [0, 0, 0, 1]]])
        random_host_fn(obs)
        assert jnp.all(all_coord_host_fn(obs) == all_coord_out)
        pts = jnp.array([[[0, 0, 4], [5, 0, 1], [1, 5, 1], [0, 25, 0]]])
        r = jnp.array([0, 1, 0, 0])
        assert jnp.all(zeillinger_fn_slice(pts[0]) == r)
        obs2 = jnp.array(
            [
                [[19, 15, 0, 10], [12, 0, 14, 9], [8, 14, 8, 18], [3, 18, 17, 12], [19, 6, 1, 13]],
                [[17, 3, 6, 9], [19, 1, 13, 12], [14, 0, 6, 7], [2, 15, 3, 16], [0, 16, 1, 5]],
                [[19, 0, 8, 6], [8, 9, 17, 1], [2, 3, 7, 14], [6, 19, 9, 12], [0, 19, 19, 14]],
            ],
            dtype=jnp.float32,
        )
        r = batch_encode_one_hot(jnp.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]]))
        assert jnp.all(zeillinger_fn(obs2) == r)
        pts = jnp.array(
            [
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [259.0, 5.0, 5.0],
                    [-1.0, -1.0, -1.0],
                    [841.0, 17.0, 0.0],
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [147.0, 3.0, 12.0],
                ]
            ]
        )

        r = jnp.array([0, 1, 0, 0])
        assert jnp.all(jnp.isclose(zeillinger_fn(pts), r))

    def test_agent(self):
        obs = jnp.array([[[1, 2, 3], [2, 3, 4]], [[0, 1, 2], [-1, -1, -1]]])
        coords = jnp.array([[1, 1, 0], [0, 1, 1]])

        random_agent_fn(make_agent_obs(obs, coords), (2, 3))
        assert jnp.all(choose_first_agent_fn(make_agent_obs(obs, coords), (2, 3)) == jnp.array([[1, 0, 0], [0, 1, 0]]))
        assert jnp.all(choose_last_agent_fn(make_agent_obs(obs, coords), (2, 3)) == jnp.array([[0, 1, 0], [0, 0, 1]]))

    def test_recurrent_fn(self):
        spec = (4, 3)
        host_obs = (
            jnp.array(
                [[[1, 2, 3], [2, 3, 4], [0, 9, 0], [-1, -1, -1]], [[4, 2, 2], [-1, -1, -1], [0, 0, 1], [-1, -1, -1]]])
            .reshape(-1, spec[0] * spec[1])
            .astype(jnp.float32)
        )
        host_actions = jnp.array([2, 3])
        batch_decode = get_batch_decode(spec[1])
        agent_obs = make_agent_obs(host_obs, batch_decode(host_actions))
        agent_actions = jnp.array([1, 0], dtype=jnp.float32)
        # For `recurrent_fn`, it takes
        #   params, key, actions: jnp.ndarray, observations: jnp.ndarray
        #   (warning: observations is already flattened)

        key = jax.random.PRNGKey(42)
        # Test host `recurrent_fn`
        nnet = DResNetMini(4)
        parameters = nnet.init(key, jnp.ones((1, 4 * 3)))
        host_policy = get_apply_fn("host", nnet, spec)
        reward_fn = get_reward_fn("host")
        recurrent_fn = get_recurrent_fn_for_role(
            "host", host_policy, partial(choose_first_agent_fn, spec=spec), reward_fn, spec, dtype=jnp.float32
        )
        print(recurrent_fn(((parameters,), ()), key, host_actions, host_obs))
        # Test agent `recurrent_fn`
        obs_preprocess, coords_preprocess = get_preprocess_fns("host", spec)
        nnet = DResNetMini(3)
        parameters = nnet.init(key, jnp.ones((1, 4 * 3 + 3)))
        agent_policy = get_apply_fn("agent", nnet, spec)
        reward_fn = get_reward_fn("agent")

        def zeillinger_fn_flatten(obs, **_):
            return zeillinger_fn(obs_preprocess(obs))

        recurrent_fn = get_recurrent_fn_for_role(
            "agent", agent_policy, zeillinger_fn_flatten, reward_fn, spec, dtype=jnp.float32
        )
        print(recurrent_fn(((parameters,), ()), key, agent_actions, agent_obs))

    def test_mcts_search(self):
        key = jax.random.PRNGKey(42)
        seed = 4242
        batch_size, max_num_points, dimension = 2, 4, 3
        spec = (max_num_points, dimension)
        points = rescale_jax(
            get_newton_polytope_jax(
                jax.random.randint(key, (batch_size, max_num_points, dimension), 0, 20).astype(jnp.float32)
            )
        )

        # host
        action_dim = 2 ** dimension - dimension - 1
        nnet = DResNetMini(action_dim)
        parameters = nnet.init(key, jnp.ones((1, 4 * 3)))
        host_policy = get_apply_fn("host", nnet, spec)
        reward_fn = get_reward_fn("host")
        recurrent_fn = get_recurrent_fn_for_role(
            "host", host_policy, partial(choose_first_agent_fn, spec=spec), reward_fn, spec, dtype=jnp.float32
        )
        state = flatten(points)
        muzero = jax.jit(
            partial(
                mctx.gumbel_muzero_policy,
                recurrent_fn=recurrent_fn,
                num_simulations=10,
                max_depth=20,
                max_num_considered_actions=10,
            )
        )
        root = mctx.RootFnOutput(
            prior_logits=jnp.array([[0] * (action_dim)] * batch_size, dtype=jnp.float32),
            value=jnp.array([0] * batch_size, dtype=jnp.float32),
            embedding=state,
        )
        policy_output = muzero(
            params=((parameters,), ()),
            rng_key=key,
            root=root,
        )

        print(policy_output)

        # agent
        action_dim = dimension
        del nnet
        nnet = DResNetMini(action_dim)
        parameters = nnet.init(key, jnp.ones((1, 4 * 3 + 3)))
        agent_policy = get_apply_fn("agent", nnet, spec)
        obs_preprocess, coords_preprocess = get_preprocess_fns("host", spec)
        reward_fn = get_reward_fn("agent")

        def zeillinger_fn_flatten(obs, **_):
            return zeillinger_fn(obs_preprocess(obs))

        recurrent_fn = get_recurrent_fn_for_role(
            "agent", agent_policy, zeillinger_fn_flatten, reward_fn, spec, dtype=jnp.float32
        )
        batch_decode_from_one_hot = get_batch_decode_from_one_hot(3)
        state = make_agent_obs(flatten(points), batch_decode_from_one_hot(zeillinger_fn(points)))
        muzero = jax.jit(
            partial(
                mctx.gumbel_muzero_policy,
                recurrent_fn=recurrent_fn,
                num_simulations=10,
                max_depth=200,
                max_num_considered_actions=10,
            )
        )
        root = mctx.RootFnOutput(
            prior_logits=jnp.array([[0] * (action_dim)] * batch_size, dtype=jnp.float32),
            value=jnp.array([0] * batch_size, dtype=jnp.float32),
            embedding=state,
        )
        policy_output = muzero(
            params=((parameters,), ()),
            rng_key=key,
            root=root,
        )

        print(policy_output)

        # test agent action mask
        tree = policy_output.search_tree
        # Make sure those that are visited do not have -jnp.inf policy prior
        assert jnp.all(~jnp.isneginf(tree.children_prior_logits) | (tree.children_visits == 0))

    def test_simulation(self):
        # simulation functions and evaluation loop functions are just wrappers of expanding/simulating process
        batch_size, max_num_points, dimension = 10, 10, 3
        spec = (max_num_points, dimension)

        key = jax.random.PRNGKey(42)
        # key, policy_key = jax.random.split(key)

        config = {
            "eval_batch_size": batch_size,
            "max_num_points": max_num_points,
            "dimension": dimension,
            "max_value": 20,
            "max_length_game": 20,
            "scale_observation": True,
        }

        action_dim = dimension
        nnet = DResNetMini(action_dim)
        nnet_host = DResNetMini(2 ** dimension - dimension - 1)
        parameters = flax.jax_utils.replicate(nnet.init(key, jnp.ones((1, 10 * 3 + 3))))
        host_params = flax.jax_utils.replicate(nnet_host.init(key, jnp.ones((1, 10 * 3))))
        agent_policy = apply_agent_action_mask(get_apply_fn("agent", nnet, spec), dimension)
        host_policy = get_apply_fn("host", nnet_host, spec)

        reward_fn = get_reward_fn("agent")
        all_coord_flattened = get_host_with_flattened_obs(spec, all_coord_host_fn)
        eval_loop = get_evaluation_loop(
            "agent",
            agent_policy,
            partial(all_coord_flattened, spec=spec),
            reward_fn,
            spec=spec,
            num_evaluations=10,
            max_depth=10,
            max_num_considered_actions=10,
            discount=0.99,
            rescale_points=True,
            reposition=False
        )
        pkey = jax.random.split(key, num=len(jax.devices()))

        root_state = generate_pts(pkey, (batch_size, max_num_points, dimension), config["max_value"], jnp.float32,
                                  True, False)
        coords, _ = jax.pmap(host_policy)(replicate(root_state), host_params)
        batch_decode_from_one_hot = get_batch_decode_from_one_hot(dimension)
        coordinate_mask = jax.pmap(batch_decode_from_one_hot)(coords)
        root_state = jnp.concatenate([jax.pmap(flatten)(root_state), coordinate_mask], axis=-1)

        # jax.config.update('jax_disable_jit', True)
        sim = get_simulation("agent", eval_loop, dtype=jnp.float32, **config)
        rollout = jax.pmap(sim)(pkey, jax.pmap(flatten)(root_state), role_fn_args=(parameters,), opponent_fn_args=())

        assert rollout_sanity_tests(rollout, spec)

    def test_jax_util(self):
        obs = jnp.ones((32, 60), dtype=jnp.float32)
        action = jnp.ones((32, 3), dtype=jnp.float32)
        out = make_agent_obs(obs, action)
        assert out.shape == (32, 63)

        host_obs = jnp.array(
            [[[1, 2, 3], [2, 3, 4], [0, 9, 0], [-1, -1, -1]], [[4, 2, 2], [-1, -1, -1], [0, 0, 1], [-1, -1, -1]]]
        ).astype(jnp.float32)
        agent_obs = {"points": jnp.copy(host_obs), "coords": jnp.array([[0, 1, 1], [1, 1, 1]])}
        combined = jnp.concatenate([agent_obs["points"].reshape(2, -1), agent_obs["coords"]], axis=1)
        take_actions = get_take_actions("host", (4, 3), rescale_points=True, reposition=False)
        # For host, `take_actions` directly receives *(obs, coords, axis)
        out = take_actions(host_obs.reshape(2, -1), agent_obs["coords"], jnp.ones(2, dtype=jnp.float32))
        assert jnp.all(
            flatten(
                rescale_jax(
                    get_newton_polytope_jax(shift_jax(host_obs, agent_obs["coords"], jnp.ones(2, dtype=jnp.float32))))
            )
            == out
        )
        # For agent, `take_actions` receives *(combined, axis, axis)
        take_actions = get_take_actions("agent", (4, 3), rescale_points=False, reposition=False)
        out = take_actions(combined, jnp.ones(2, dtype=jnp.float32), jnp.ones(2, dtype=jnp.float32))
        assert jnp.all(
            flatten(get_newton_polytope_jax(
                shift_jax(host_obs, agent_obs["coords"], jnp.ones(2, dtype=jnp.float32)))) == out
        )

    def test_uniform_recurrent_fn(self):
        spec = (4, 3)
        batch_size, max_num_points, dimension = 2, 4, 3

        host_obs = (
            jnp.array(
                [[[1, 2, 3], [2, 3, 4], [0, 9, 0], [-1, -1, -1]], [[4, 2, 2], [-1, -1, -1], [0, 0, 1], [-1, -1, -1]]])
            .reshape(-1, spec[0] * spec[1])
            .astype(jnp.float32)
        )
        host_obs = jnp.concatenate([host_obs, jnp.zeros((2, 3))], axis=-1)
        host_actions = jnp.array([2, 3])

        key = jax.random.PRNGKey(42)

        nnet = DResNetMini(4)
        nnet_a = DResNetMini(3)
        host_parameters = nnet.init(key, jnp.ones((1, 4 * 3)))
        agent_parameters = nnet_a.init(key, jnp.ones((1, 4 * 3 + 3)))
        host_fn = get_apply_fn("host", nnet, spec)

        def host_fn_truncate_input(x, param, *args, **kwargs):
            return host_fn(x[:, :-dimension], param, *args, **kwargs)

        agent_fn = apply_agent_action_mask(get_apply_fn("agent", nnet_a, spec), 3)
        reward_fn = get_reward_fn("host")
        recurrent_fn = get_unified_recurrent_fn(
            host_fn_truncate_input, agent_fn, reward_fn, spec, dtype=jnp.float32
        )
        output, next_obs = recurrent_fn(((host_parameters,), (agent_parameters,)), key, host_actions, host_obs)
        output2, next_obs2 = recurrent_fn(((host_parameters,), (agent_parameters,)), key, host_actions, next_obs)

        # Agent action is padded to the length of host action by -jnp.inf
        assert jnp.all(output.prior_logits[:, -1] == -jnp.inf)
        # When it comes back to host node, the last three coordinates of observations are padded with 0
        assert jnp.all(jnp.isclose(next_obs2[:, -3:], 0))

        muzero = partial(
            mctx.gumbel_muzero_policy,
            recurrent_fn=recurrent_fn,
            num_simulations=10,
            max_depth=20,
            max_num_considered_actions=10,
        )

        # Host search
        policy, value = host_fn(host_obs[:, :4 * 3], host_parameters)
        root = mctx.RootFnOutput(
            prior_logits=policy,
            value=value,
            embedding=host_obs,
        )
        policy_output = muzero(
            params=((host_parameters,), (agent_parameters,)),
            rng_key=key,
            root=root,
        )
        # Agent search
        policy, value = agent_fn(next_obs, agent_parameters)
        policy = jnp.pad(policy, ((0, 0), (0, 1)), mode='constant', constant_values=(0, -jnp.inf))
        root = mctx.RootFnOutput(
            prior_logits=policy,
            value=value,
            embedding=next_obs,
        )
        policy_output = muzero(
            params=((host_parameters,), (agent_parameters,)),
            rng_key=key,
            root=root,
        )
        # The last action weight must be zero: it is only padded to be of the same length as host.
        assert jnp.all(jnp.isclose(policy_output.action_weights[:, -1], 0))

        # ----- below it continues to test eval_loop and sim_fn ----- #
        eval_loop = get_evaluation_loop(
            "host",
            host_fn_truncate_input,
            agent_fn,
            reward_fn,
            spec=spec,
            num_evaluations=10,
            max_depth=10,
            max_num_considered_actions=10,
            discount=0.99,
            role_agnostic=True,  # <-- this is for the unified MC search tree.
            rescale_points=True,
            reposition=False
        )

        config = {
            "eval_batch_size": batch_size,
            "max_num_points": max_num_points,
            "dimension": dimension,
            "max_value": 20,
            "max_length_game": 20,
            "scale_observation": True,
        }
        # jax.config.update('jax_disable_jit', True)
        sim = get_simulation("host", eval_loop, dtype=jnp.float32, **config)
        rollout = sim(key, host_obs, role_fn_args=(host_parameters,), opponent_fn_args=(agent_parameters,))
        assert rollout_sanity_tests(rollout, spec)
        rollout = sim(key, rollout[0][:, 1], role_fn_args=(host_parameters,), opponent_fn_args=(agent_parameters,))
        assert rollout_sanity_tests(rollout, spec)
