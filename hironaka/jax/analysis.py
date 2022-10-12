"""
Helper functions for fine-grained analysis, mostly with visualizations.
"""
import logging
from functools import partial
from typing import Tuple, Union, List
import jax
import jax.numpy as jnp

import mctx
import pygraphviz as pgv


def draw_mcts_output(output: mctx._src.base.PolicyOutput, spec: Tuple[int, int]) -> List[pgv.AGraph]:
    """
    Given an mctx output, it returns pygraphviz AGraphs which allow for saving png pictures.
    Parameters:
        output: the mctx PolicyOutput
        spec: (max_num_point, dimension)
    Returns:
        a list of AGraph (size 1 if unparallelized, or multiple if parallelized across several devices)
    """
    search_tree = output.search_tree
    max_num_points, dimension = spec

    if len(search_tree.children_index.shape) not in [2, 3]:
        raise ValueError('Format error in output.')
    if len(search_tree.children_index.shape) == 2:
        search_tree = jax.tree_util.tree_map(partial(jnp.expand_dims, axis=0), search_tree)

    def node_label_fn(idx):
        return f"id:{idx}\nnum>0:{jnp.sum(embedding[idx] >= 0)}\nvisits:{node_visits[idx]}"

    def edge_label_fn(node, action):
        return f"logit: {children_logit[node, action]:.2f}\nvalue: {children_value[node, action]:.2f}"

    def draw(graph: pgv.AGraph, node: int):
        for action in range(2**dimension - dimension - 1):
            if children_id[node, action] != -1:
                next_node = children_id[node, action]
                graph.add_node(next_node, label=node_label_fn(next_node))
                graph.add_edge(node, next_node, label=edge_label_fn(node, action))
                draw(graph, next_node)

    num_device = search_tree.children_index.shape[0]
    graphs = []
    for i in range(num_device):
        children_id = output.search_tree.children_index[i]
        children_logit = output.search_tree.children_prior_logits[i]
        children_value = output.search_tree.children_values[i]
        node_visits = output.search_tree.node_visits[i]
        embedding = output.search_tree.embeddings[i]

        graphs.append(pgv.AGraph())
        graphs[-1].add_node(0, label=node_label_fn(0))
        draw(graphs[-1], 0)

    return graphs

