# src/celltreebench/utils/tree_operations.py

from ete3 import Tree
import pandas as pd
import numpy as np
import torch


def generate_topology_tree(num_leaves):
    if not (num_leaves & (num_leaves - 1) == 0):
        raise ValueError("Number of leaves must be a power of 2 for full binary tree.")
    return _make_full_binary_tree(num_leaves)


def generate_model_tree(topology_tree, max_walk):
    tree = topology_tree.copy()
    _walk_tree(node=tree, func=_walk_tree_func_edge_length, max_walk=max_walk)
    return tree


def generate_sample_tree(model_tree, n_signals, n_noise, noise_std=1):
    tree = model_tree.copy()
    _walk_tree(
        tree,
        _walk_tree_func_node_vector,
        n_signals=n_signals,
        n_noise=n_noise,
        noise_std=noise_std,
    )
    return tree


def get_topology_tree_dm(tree, only_leaves=True):
    def _dist_func(node1, node2):
        return node1.get_distance(node2, topology_only=True)

    return _get_tree_dist(tree, _dist_func, only_leaves)


def get_model_tree_dm(tree, only_leaves=True):
    def _dist_func(node1, node2):
        return node1.get_distance(node2, topology_only=False)

    return _get_tree_dist(tree, _dist_func, only_leaves)


def get_sample_tree_dm(tree, only_leaves=True):
    def _dist_func(node1, node2):
        return np.linalg.norm(node1.node_vector - node2.node_vector)

    return _get_tree_dist(tree, _dist_func, only_leaves)


def get_node_mtx(tree, only_leaves=True):
    nodes = tree.get_leaves() if only_leaves else tree.traverse()
    data = []
    for node in nodes:
        if hasattr(node, "node_vector"):
            entry = {f"feature_{i+1}": v for i, v in enumerate(node.node_vector)}
            entry["node_name"] = node.name
            data.append(entry)
        else:
            raise ValueError("Node does not have node_vector attribute")
    return pd.DataFrame(data).set_index("node_name")


def _get_tree_dist(tree, dist_func, only_leaves=True):
    nodes = tree.get_leaves() if only_leaves else list(tree.traverse())
    names = [n.name for n in nodes]
    mat = pd.DataFrame(index=names, columns=names)
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i < j:
                dist = dist_func(n1, n2)
                mat.iloc[i, j] = mat.iloc[j, i] = dist
    return mat.astype(float)


def _make_full_binary_tree(num_leaves):
    total_nodes = 2 * num_leaves - 1

    def add_children(parent, current_index):
        left_idx = 2 * current_index + 1
        right_idx = 2 * current_index + 2
        if left_idx >= total_nodes:
            return
        left = parent.add_child(name=f"Node{left_idx}")
        right = parent.add_child(name=f"Node{right_idx}")
        add_children(left, left_idx)
        add_children(right, right_idx)

    root = Tree()
    root.name = "Node0"
    add_children(root, 0)
    return root


def _walk_tree(node, func, **kwargs):
    func(node, **kwargs)
    for child in node.children:
        _walk_tree(child, func, **kwargs)


def _walk_tree_func_edge_length(node, max_walk):
    if node.is_root():
        node.add_features(dist=0, dist_format="{:.3f}".format(0))
    else:
        length = np.random.uniform(1, max_walk)
        node.dist = length
        node.dist_format = "{:.3f}".format(length)


def _walk_tree_func_node_vector(node, n_signals, n_noise=0, noise_std=1):
    if node.is_root():
        node_vector = np.random.normal(size=n_signals + n_noise)
    else:
        edge_length = node.dist
        parent_vector = node.parent.node_vector
        diff_part = np.random.normal(
            loc=parent_vector[:n_signals], scale=edge_length, size=n_signals
        )
        noise_part = np.random.normal(loc=0, scale=noise_std, size=n_noise)
        node_vector = np.concatenate([diff_part, noise_part])
    node.add_features(node_vector=node_vector)


def get_path_distance_matrix(tree, leaf_names):
    # return topology distance metrix and corresponding node names for indexing
    leaf_map = get_leaves_dict(tree)
    n = len(leaf_names)
    dm = torch.zeros((n, n))
    for i, leaf1_name in enumerate(leaf_names):
        for j, leaf2_name in enumerate(leaf_names):
            leaf1 = leaf_map[leaf1_name]
            leaf2 = leaf_map[leaf2_name]
            dm[i, j] = tree.get_distance(leaf1, leaf2, topology_only=True)
            # logging.info(f"dm[{i},{j}] = {dm[i,j]}, leaves: {leaf1_name}, {leaf2_name}")
    return dm


def get_leaves_dict(tree):
    # Get all leaves
    leaves = tree.get_leaves()
    # Create a dictionary with leaf names as keys and leaf nodes as values
    leaves_dict = {leaf.name: leaf for leaf in leaves}
    return leaves_dict


__all__ = [
    "generate_topology_tree",
    "generate_model_tree",
    "generate_sample_tree",
    "get_topology_tree_dm",
    "get_model_tree_dm",
    "get_sample_tree_dm",
    "get_node_mtx",
    "get_path_distance_matrix",
    "get_leaves_dict",
]
