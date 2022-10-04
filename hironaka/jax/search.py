"""
Utility functions for the actual game playing or tree searching.
"""
from collections import deque, namedtuple
from typing import Optional, Any, Callable, List, Tuple

from dataclasses import dataclass
import pygraphviz as pgv
import jax.numpy as jnp

from hironaka.jax.host_action_preprocess import decode_from_one_hot, get_batch_decode_from_one_hot
from hironaka.jax.players import get_host_with_flattened_obs, zeillinger_fn
from hironaka.jax.util import get_take_actions, get_preprocess_fns, get_done_from_flatten
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


def search_tree_fix_host(node: TreeNode, spec: Tuple, host: Callable, depth: int, scale_observation=True) -> TreeNode:
    """
    Node data is assumed to be 1d with length (max_num_points + 1) * dimension.
    Parameters:
        node: the TreeNode object.
        spec: (max_num_points, dimension).
        host: the host function that outputs policy logits (not converted to multi-binaries yet).
        depth: the depth of the node.
        scale_observation: (Optional) whether to scale the observation.
    """
    decode = get_batch_decode_from_one_hot(spec[1])
    if node.children is None:
        node.children = []

    if get_done_from_flatten(node.data, 'agent', spec[1])[0]:
        return node

    # Truncate the observation to (1, max_num_points * dimension)
    obs_preprocess, _ = get_preprocess_fns('agent', spec)
    # Change the observation according to host and agent actions
    take_action = get_take_actions('host', spec, rescale_points=False)

    # Make inference and decode into a multi-binary array showing host's choice of axis
    multi_bin = decode(host(node.data, axis=0))  # shape: (1, dimension)

    for i in range(spec[1]):
        if multi_bin[0, i] > 0.5:
            new_state = take_action(obs_preprocess(node.data).astype(jnp.float32),
                                    multi_bin.astype(jnp.float32),
                                    jnp.array([i], dtype=jnp.float32))
            new_obs = jnp.concatenate([new_state, jnp.zeros((1, spec[1]))], axis=-1)
            new_node = search_tree_fix_host(TreeNode(parent=node, action_from_parent=i, data=new_obs),
                                            spec, host, depth+1, scale_observation)
            node.children.append(new_node)

    return node


if __name__ == '__main__':
    #pt = jnp.array([[3, 0, 0, 0, 5, 0, 0, 0, 2, 0, 0, 0]])  # E8-singularity
    pt = jnp.array([[2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0]])  # D4-singularity
    root = TreeNode(parent=None, action_from_parent=None, data=pt)
    spec = (3, 3)
    zeillinger_flattened = get_host_with_flattened_obs(spec, zeillinger_fn, truncate_input=True)
    search_tree_fix_host(root, spec, zeillinger_flattened, 0, scale_observation=True)

    def label_fn(node):
        points = node.data[:, :-spec[1]].reshape(spec)
        points = points[points[:, 0] >= 0]
        return str(points)

    graph = root.to_graphviz(100, label_fn=label_fn)
    graph.layout('dot')
    graph.draw('runs/tree.png')
