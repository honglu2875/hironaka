"""
Utility functions for the actual game playing or tree searching.
"""
import logging
import sys
from collections import deque, namedtuple
from functools import partial
from typing import Optional, Any, Callable, List, Tuple

from dataclasses import dataclass

import jax
import pygraphviz as pgv
import jax.numpy as jnp
from flax.jax_utils import unreplicate

from hironaka.jax import JAXTrainer
from hironaka.jax.host_action_preprocess import decode_from_one_hot, get_batch_decode_from_one_hot
from hironaka.jax.players import get_host_with_flattened_obs, zeillinger_fn
from hironaka.jax.util import get_take_actions, get_preprocess_fns, get_done_from_flatten, mcts_wrapper, action_wrapper
from hironaka.src import rescale_jax


@dataclass
class TreeNode:
    children: Optional[List["TreeNode"]] = None
    parent: Optional["TreeNode"] = None
    action_from_parent: Optional[int] = None
    data: Any = None

    def to_graphviz(self, max_depth: Optional[int] = None, label_fn: Optional[Callable] = None) -> pgv.AGraph:
        """
        Convert the subtree from the current node into a pygraphviz graph.
        Use BFS so that the depth can be controlled.
        Parameters:
            max_depth: (Optional) maximum of depth allowed.
            label_fn: (Optional) a function that applies to a TreeNode object to provide the label for each node.
        Returns:
            pygraphviz AGraph object.
        """
        pgv_graph = pgv.AGraph()
        self.add_node(pgv_graph, id=0, node=self, parent_id=None, label_fn=label_fn, edge_label=None)
        num_nodes = 1

        Data = namedtuple("Data", ["node", "depth", "id"])
        queue = deque([Data(self, 0, 0)])
        while queue:
            curr_node, curr_depth, curr_id = queue.popleft()
            if curr_node.children is None or (max_depth is not None and curr_depth >= max_depth):
                continue
            for child in curr_node.children:
                queue.append(Data(child, curr_depth + 1, num_nodes))
                self.add_node(pgv_graph, id=num_nodes, node=child, parent_id=curr_id, label_fn=label_fn,
                              edge_label=str(child.action_from_parent))
                num_nodes += 1

        return pgv_graph

    @staticmethod
    def add_node(graph: pgv.AGraph, id, node, parent_id=None, label_fn=None, edge_label=None):
        label = label_fn(node) if label_fn is not None else str(id)
        graph.add_node(id, label=label)
        if parent_id is not None:
            graph.add_edge(parent_id, id, label=edge_label)


def default_label_fn(node):
    points = node.data[:, :-spec[1]].reshape(spec)
    points = points[points[:, 0] >= 0]
    return str(points)


def search_tree_fix_host(node: TreeNode, spec: Tuple, host: Callable,
                         depth: int, key: jnp.ndarray, scale_observation=True, max_depth=1000) -> TreeNode:
    """
    Node data is assumed to be 1d with length (max_num_points + 1) * dimension.
    Parameters:
        node: the TreeNode object.
        spec: (max_num_points, dimension).
        host: the host function that outputs policy logits (not converted to multi-binaries yet).
        depth: the depth of the node.
        key: the PRNGKey
        scale_observation: (Optional) whether to scale the observation.
        max_depth: (Optional) the maximal depth.
    """
    decode = get_batch_decode_from_one_hot(spec[1])
    if node.children is None:
        node.children = []

    if get_done_from_flatten(node.data, 'agent', spec[1])[0] or depth > max_depth:
        return node

    # Truncate the observation to (1, max_num_points * dimension)
    obs_preprocess, _ = get_preprocess_fns('agent', spec)
    # Change the observation according to host and agent actions
    take_action = get_take_actions('host', spec, rescale_points=False)

    # Make inference and decode into a multi-binary array showing host's choice of axis
    key, subkey = jax.random.split(key)
    multi_bin = decode(host(node.data, key=subkey))  # shape: (1, dimension)

    for i in range(spec[1]):
        if multi_bin[0, i] > 0.5:
            new_state = take_action(obs_preprocess(node.data).astype(jnp.float32),
                                    multi_bin.astype(jnp.float32),
                                    jnp.array([i], dtype=jnp.float32))
            new_obs = jnp.concatenate([new_state, jnp.zeros((1, spec[1]))], axis=-1)
            key, subkey = jax.random.split(key)
            new_node = search_tree_fix_host(TreeNode(parent=node, action_from_parent=i, data=new_obs),
                                            spec, host, depth+1, subkey, scale_observation, max_depth)
            node.children.append(new_node)

    return node


if __name__ == '__main__':
    # zeillinger_flattened = get_host_with_flattened_obs(spec, zeillinger_fn, truncate_input=True)

    # ------------ Get the MCTS policy functions ------------ #
    trainer = JAXTrainer(jax.random.PRNGKey(42), 'train/jax_mcts.yml')
    spec = (trainer.max_num_points, trainer.dimension)
    assert spec[1] == 3  # Below we run search for a few A,D,E singularities.
    # pt = jnp.array([[3, 0, 0, 0, 5, 0, 0, 0, 2] + [-1] * (spec[0]-3) * spec[1] + [0, 0, 0]], dtype=jnp.float32)
    # pt = jnp.array([[2, 0, 0, 0, 3, 0, 0, 0, 3] + [-1] * (spec[0]-3) * spec[1] + [0, 0, 0]], dtype=jnp.float32)
    # pt = jnp.array([[2, 0, 0, 0, 3, 0, 0, 0, 4] + [-1] * (spec[0]-3) * spec[1] + [0, 0, 0]], dtype=jnp.float32)
    # pt = jnp.array([[2, 0, 0, 0, 2, 1, 0, 0, 5] + [-1] * (spec[0]-3) * spec[1] + [0, 0, 0]], dtype=jnp.float32)
    # pt = jnp.array([[2, 0, 0, 0, 2, 0, 0, 0, 4] + [-1] * (spec[0]-3) * spec[1] + [0, 0, 0]], dtype=jnp.float32)
    pt = jnp.array([[3, 0, 0, 0, 5, 0, 0, 2, 2] + [-1] * (spec[0] - 3) * spec[1] + [0, 0, 0]], dtype=jnp.float32)
    root = TreeNode(parent=None, action_from_parent=None, data=pt)

    logger = trainer.logger
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
    trainer.load_checkpoint('train/models')
    host_fn = mcts_wrapper(trainer.unified_eval_loop)

    host = jax.jit(action_wrapper(
        partial(host_fn,
                params=unreplicate(trainer.host_state.params),
                opp_params=unreplicate(trainer.agent_state.params))))

    search_tree_fix_host(root, spec, host, 0, jax.random.PRNGKey(422), scale_observation=True, max_depth=4)

    graph = root.to_graphviz(10, label_fn=default_label_fn)
    graph.layout('dot')
    graph.draw('runs/tree.png')
