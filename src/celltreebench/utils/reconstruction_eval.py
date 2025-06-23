# src/celltreebench/utils/reconstruction_eval.py

import torch
import logging
from ete3 import Tree
from .distance_metrics import pairwise_distances

from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from io import StringIO
import Bio.Phylo as Phylo
import numpy as np


def reconstruct_from_dm(dm, node_names, method="nj", unrooted=True):
    tree = _dm_to_etree(dm, node_names=node_names, method=method)
    for node in tree.traverse():
        node.dist_format = "{:.2f}".format(node.dist)
    if unrooted:
        tree.unroot()
    return tree


def compare_trees(tree1, tree2, unrooted_trees=False):
    if unrooted_trees:
        if not _is_unrooted(tree1):
            tree1 = tree1.copy()
            tree1.unroot()
        if not _is_unrooted(tree2):
            tree2 = tree2.copy()
            tree2.unroot()
    else:
        if _is_unrooted(tree1) or _is_unrooted(tree2):
            raise ValueError("Both trees must be rooted for rooted comparison")

    res = tree1.compare(tree2, unrooted=unrooted_trees)
    return {
        "rf": res["rf"],
        "relative_rf": res["norm_rf"],
        "max_rf": res["max_rf"],
        "effective_tree_size": res["effective_tree_size"],
        "ref_edges_in_source": res["ref_edges_in_source"],
        "source_edges_in_ref": res["source_edges_in_ref"],
    }


def evaluate_reconstruction(
    dataset,
    model,
    res_dict,
    phase="test",
    dist_metric="euclidean",
    device="cpu",
    method="nj",
):
    model.eval()
    node_mtx_dict = dataset.get_node_mtx()
    node_mtx = torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float).to(device)
    node_names = node_mtx_dict["node_names"]

    with torch.no_grad():
        transformed = model(node_mtx)
    emb_dm = pairwise_distances(transformed.cpu(), metric=dist_metric)
    emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

    # Map names if needed
    if hasattr(dataset, "species_to_scientific_name_map_df"):
        name_map_df = dataset.species_to_scientific_name_map_df
        for leaf in emb_tree.get_leaves():
            leaf.name = name_map_df[
                name_map_df.species == leaf.name
            ].scientific_name.values[0]

    comp_result = dataset.compare_trees(emb_tree, ref_tree="topology_tree")
    key = f"rf_emb_topo_{phase}"
    res_dict[key][method].append(comp_result["relative_rf"])
    return {key: comp_result["relative_rf"]}


def train_reconstruct_eval(*args, **kwargs):
    return evaluate_reconstruction(*args, phase="train", **kwargs)


def val_reconstruct_eval(*args, **kwargs):
    return evaluate_reconstruction(*args, phase="val", **kwargs)


def test_reconstruct_eval(*args, **kwargs):
    return evaluate_reconstruction(*args, phase="test", **kwargs)


def test_unknown_reconstruct_eval(*args, **kwargs):
    return evaluate_reconstruction(*args, phase="test_unknown", **kwargs)


def _is_unrooted(tree):
    return len(tree.get_children()) != 2


def _dm_to_etree(dm, node_names, method="nj"):
    if hasattr(dm, "numpy"):
        dm = dm.cpu().numpy() if dm.is_cuda else dm.numpy()

    method_map = {"nj": "nj", "upgma": "average", "single": "single", "ward": "ward"}
    method = method_map.get(method, method)

    if method == "nj":
        return _nj_reconstruct(dm, node_names)
    else:
        from scipy.cluster.hierarchy import linkage, to_tree

        X = _full_to_condensed(dm)
        Z = linkage(X, method)
        root_node, nodes = to_tree(Z, rd=True)

        for node in nodes:
            node.name = str(node.id)
        for i, name in enumerate(node_names):
            nodes[i].name = name

        ete_root = Tree()
        ete_root.name = str(root_node.id)
        ete_root.dist = 0
        node_map = {root_node.id: ete_root}
        queue = [root_node]
        while queue:
            node = queue.pop(0)
            for child in [node.left, node.right]:
                if child:
                    new_node = Tree()
                    new_node.name = child.name
                    new_node.dist = child.dist / 2.0
                    node_map[node.id].add_child(new_node)
                    node_map[child.id] = new_node
                    queue.append(child)
        return ete_root


def _nj_reconstruct(dm, names):
    lower_matrix = _lower_triangle_list(dm)
    dm = DistanceMatrix(names=names, matrix=lower_matrix)
    tree = DistanceTreeConstructor().nj(dm)
    buf = StringIO()
    Phylo.write(tree, buf, "newick")
    buf.seek(0)
    return Tree(buf.getvalue(), format=1)


def _lower_triangle_list(matrix):
    n = matrix.shape[0]
    return [[matrix[i, j] for j in range(i + 1)] for i in range(n)]


def _full_to_condensed(dm):
    return dm[np.triu_indices(dm.shape[0], k=1)]


__all__ = [
    "reconstruct_from_dm",
    "compare_trees",
    "evaluate_reconstruction",
    "train_reconstruct_eval",
    "val_reconstruct_eval",
    "test_reconstruct_eval",
    "test_unknown_reconstruct_eval",
]
