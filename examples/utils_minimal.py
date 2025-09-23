import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
from ete3 import Tree
import tqdist
from scipy.cluster import hierarchy as hcluster
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
import Bio.Phylo as Phylo
from io import StringIO

def _triu_vector(dm: torch.Tensor) -> torch.Tensor:
    """把成对距离矩阵提取为上三角向量。
    支持 (B,N,N) 或 (N,N)，返回形状为 (B, P) 的向量，其中 P=N*(N-1)/2。
    """
    if dm.dim() == 2:
        dm = dm.unsqueeze(0)  # (1,N,N)
    B, N, _ = dm.shape
    idx = torch.triu_indices(N, N, offset=1, device=dm.device)
    vec = dm[:, idx[0], idx[1]]  # (B, P)
    return vec

def _align_scale_vec(vec_est: torch.Tensor, vec_ref: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """对齐尺度：给定两个 (B,P) 的距离向量，求 s 使得 || s*vec_est - vec_ref ||_2 最小。
    返回 (s, vec_est_aligned) 其中 s 形状为 (B,)。
    """
    # s = (e·r) / (e·e)
    num = (vec_est * vec_ref).sum(dim=1)                # (B,)
    den = (vec_est * vec_est).sum(dim=1).clamp_min(eps) # (B,)
    s = num / den                                       # (B,)
    vec_est_aligned = vec_est * s.unsqueeze(1)          # (B,P)
    return s, vec_est_aligned

def dm_to_etree(dm, node_names=None, method="nj"):
    """
    Construct ete3 tree from the distance matrix of the leaves.

    Args:
        dm: distance matrix of the leaves
        node_names: names of the leaves
        method: reconstruction method ("nj", "upgma", "ward", "single")

    Returns:
        ete3.Tree: Reconstructed phylogenetic tree
    """
    # If dm is torch.Tensor, transform it to numpy ndarray
    if hasattr(dm, "numpy"):
        # if dm is in GPU, transform it to CPU
        if dm.is_cuda:
            dm = dm.cpu()
        dm = dm.numpy()

    # If node_names is dataframe.index, transform it to list
    if hasattr(node_names, "to_list"):
        node_names = node_names.to_list()

    method_name_map = {
        "ward": "ward",
        "upgma": "average",
        "single": "single",
        "nj": "nj",
    }
    method = method_name_map[method]

    if method == "nj":
        return _nj_reconstruct(dm, node_names)
    else:
        # Hierarchical clustering methods
        n = dm.shape[0]
        X = _full_to_condensed(dm)
        Z = hcluster.linkage(X, method)
        T = hcluster.to_tree(Z, rd=True)

        scipy_tree_root = T[0]
        scipy_tree_node_list = T[1]

        for node in scipy_tree_node_list:
            node.name = str(node.id)

        if node_names is not None:
            for i in range(len(node_names)):
                scipy_tree_node_list[i].name = node_names[i]

        # Create the root for ete3 tree and initialize mapping with node ids
        ete3_root = Tree()
        ete3_root.name = str(scipy_tree_root.id)
        ete3_root.dist = 0
        node_map = {scipy_tree_root.id: ete3_root}

        # BFS to copy from scipy tree to ete3 tree
        to_visit = [scipy_tree_root]
        while to_visit:
            current_scipy_node = to_visit.pop(0)
            cl_dist = current_scipy_node.dist / 2.0
            current_ete3_node = node_map[current_scipy_node.id]

            # Add children nodes
            for child in [current_scipy_node.left, current_scipy_node.right]:
                if child:
                    new_ete3_node = Tree()
                    new_ete3_node.add_features(name=child.name)
                    new_ete3_node.add_features(dist=cl_dist)
                    new_ete3_node.add_features(dist_format="{:.3f}".format(cl_dist))
                    current_ete3_node.add_child(new_ete3_node)
                    node_map[child.id] = new_ete3_node
                    to_visit.append(child)

        return ete3_root


def _nj_reconstruct(dm, names):
    """Reconstruct tree using Neighbor Joining algorithm."""
    lower_matrix = _lower_triangle_list(dm)
    dm_bio = DistanceMatrix(names=names, matrix=lower_matrix)
    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(dm_bio)
    newick_str = StringIO()
    Phylo.write(nj_tree, newick_str, "newick")
    newick_str.seek(0)
    ete_tree = Tree(newick_str.getvalue(), format=1)
    return ete_tree


def _lower_triangle_list(matrix):
    """Convert distance matrix to lower triangle list format for Bio.Phylo."""
    n = matrix.shape[0]
    lower_triangle = []

    # Iterate over each row and gather the elements below the diagonal
    for i in range(0, n):
        row_values = []
        for j in range(i + 1):  # j goes from 0 to i
            row_values.append(matrix[i, j])
        lower_triangle.append(row_values)

    return lower_triangle


def _full_to_condensed(distance_matrix):
    """Convert full distance matrix to condensed format for scipy hierarchical clustering."""
    n = distance_matrix.shape[0]
    # Get the indices for the upper triangle, excluding the diagonal
    upper_triangle_indices = np.triu_indices(n, k=1)
    # Extract the distances using these indices
    condensed_matrix = distance_matrix[upper_triangle_indices]
    return condensed_matrix

def _zscore_vec(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对 (B,P) 向量做逐 batch z-score。"""
    m = v.mean(dim=1, keepdim=True)
    sd = v.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    return (v - m) / sd

def distance_error_from_dm(
    dm_est: torch.Tensor,
    dm_ref: torch.Tensor,
    diff_norm="fro",             # "fro" | 1 | 2 | "inf"
    alpha: float = 0.5,          # 与 distance_error 相同的 blend
    align: str = "scale",        # "scale" | "zscore" | "none"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    直接以“距离矩阵”为输入，比较 dm_est 与 dm_ref。
    - matrix 项：上三角向量对齐后做 L2/L1/Linf 误差
    - stats  项：上三角均值/标准差的 MSE（与你现有 _distance_error_stats 一致）
    """
    if dm_est.dim() == 2: dm_est = dm_est.unsqueeze(0)
    if dm_ref.dim() == 2: dm_ref = dm_ref.unsqueeze(0)
    assert dm_est.shape == dm_ref.shape and dm_est.dim() == 3, "dm_est/dm_ref 需为 (B,N,N) 同形状"

    v_est = _triu_vector(dm_est)   # (B,P)
    v_ref = _triu_vector(dm_ref)   # (B,P)

    # 对齐（只作用在 matrix 项；与你 distance_error 的行为保持一致）
    if align == "scale":
        _, v_est_aligned = _align_scale_vec(v_est, v_ref, eps=eps)
    elif align == "zscore":
        v_est_aligned = _zscore_vec(v_est, eps)
        v_ref        = _zscore_vec(v_ref, eps)
    elif align == "none":
        v_est_aligned = v_est
    else:
        raise ValueError("align must be 'scale', 'zscore' or 'none'.")

    diff = v_est_aligned - v_ref
    if diff_norm in ("fro", 2):
        loss_matrix = (diff.pow(2).mean(dim=1)).sqrt().mean()
    elif diff_norm == 1:
        loss_matrix = diff.abs().mean(dim=1).mean()
    elif diff_norm == "inf":
        loss_matrix = diff.abs().amax(dim=1).mean()
    else:
        raise ValueError(f"Unsupported norm type '{diff_norm}'.")

    # 统计项（均值/标准差）
    est_mean = v_est.mean(dim=1)
    ref_mean = v_ref.mean(dim=1)
    est_std  = v_est.std (dim=1, unbiased=False)
    ref_std  = v_ref.std (dim=1, unbiased=False)
    loss_stats = ((est_mean - ref_mean)**2 + (est_std - ref_std)**2).mean()

    return alpha * loss_matrix + (1.0 - alpha) * loss_stats

def _distance_error_matrix(
    orig_point_matrix,
    transformed_point_matrix,
    diff_norm="fro",
    dist_metric="euclidean",
    align: str = "scale",      # 新增：'scale' | 'zscore' | 'none'
    eps: float = 1e-8,
):
    """
    比较原始与嵌入的成对距离矩阵的误差；支持先做尺度对齐（'scale'），
    或 z-score 对齐（'zscore'），或不对齐（'none'）。
    返回一个标量损失（对 batch 做平均）。
    """
    valid_norms = ["fro", 1, 2, "inf"]
    if diff_norm not in valid_norms:
        raise ValueError(f"Unsupported norm type '{diff_norm}'.")

    if orig_point_matrix.dim() != 3 or transformed_point_matrix.dim() != 3:
        raise ValueError("The shape must be (B, M, N).")
    if orig_point_matrix.size(1) != transformed_point_matrix.size(1):
        raise ValueError("The second dimension (number of points) must match.")

    # 成对距离（B,N,N）
    dis_orig = pairwise_distances(orig_point_matrix, metric=dist_metric)
    dis_trans = pairwise_distances(transformed_point_matrix, metric=dist_metric)

    # 上三角向量 (B,P)
    v_ref = _triu_vector(dis_orig)
    v_est = _triu_vector(dis_trans)

    # 对齐
    if align == "scale":
        s, v_est_aligned = _align_scale_vec(v_est, v_ref, eps=eps)  # s 可用于诊断
    elif align == "zscore":
        # 双方做 z-score 归一化（移除平移与缩放）
        def _z(x):
            m = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
            return (x - m) / sd
        v_ref = _z(v_ref)
        v_est_aligned = _z(v_est)
    elif align == "none":
        v_est_aligned = v_est
    else:
        raise ValueError("align must be 'scale', 'zscore' or 'none'.")

    # 误差（按 diff_norm 聚合）。为数值稳定及可解释，这里用“每样本 RMS/L1/Linf 再取 batch 平均”
    diff = v_est_aligned - v_ref  # (B,P)
    if diff_norm in ("fro", 2):
        # root-mean-square over pairs, then mean over batch
        err = (diff.pow(2).mean(dim=1)).sqrt().mean()
    elif diff_norm == 1:
        err = diff.abs().mean(dim=1).mean()
    elif diff_norm == "inf":
        err = diff.abs().amax(dim=1).mean()

    return err


# def _distance_error_matrix(
#     orig_point_matrix,
#     transformed_point_matrix,
#     diff_norm="fro",
#     dist_metric="euclidean",
# ):
#     """
#     Compute the error between pairwise distance matrices of the original and transformed points.

#     Args:
#         orig_point_matrix (Tensor): Tensor of shape (B, M, N), representing the original points.
#         transformed_point_matrix (Tensor): Tensor of shape (B, M, K), representing the transformed points.
#         diff_norm (str or int): Norm type to use for computing the error. Options are 'fro' (Frobenius norm) or any valid p-norm (e.g., 1, 2).
#         dist_metric (str): Distance metric to use when computing pairwise distances. Options are 'cosine', 'euclidean', 'manhattan'.

#     Returns:
#         Tensor: A scalar tensor representing the mean distance error between the original and transformed points.
#     """
#     valid_norms = ["fro", 1, 2, "inf"]
#     if diff_norm not in valid_norms:
#         raise ValueError(
#             f"Unsupported norm type '{diff_norm}'. Supported options are 'fro', 1, or 2."
#         )
#     elif diff_norm == "inf":
#         diff_norm = np.inf

#     if len(orig_point_matrix.size()) != 3 or len(transformed_point_matrix.size()) != 3:
#         raise ValueError("The shape must be (B, M, N).")
#     else:
#         if orig_point_matrix.size(1) != transformed_point_matrix.size(1):
#             raise ValueError(
#                 "The second dimension (number of points) must be the same for both input matrices."
#             )

#     dis_orig = pairwise_distances(orig_point_matrix, metric=dist_metric)
#     dis_transformed = pairwise_distances(transformed_point_matrix, metric=dist_metric)

#     # Get the number of features (dimensions) for normalization
#     orig_num_features = orig_point_matrix.size(2)
#     if not isinstance(orig_num_features, torch.Tensor):
#         orig_num_features = torch.tensor(orig_num_features, dtype=torch.float32)

#     transformed_num_features = transformed_point_matrix.size(2)

#     # Normalize the pairwise distances by the square root of the number of features
#     dis_orig = dis_orig / torch.sqrt(orig_num_features)
#     dis_transformed = dis_transformed / torch.sqrt(
#         torch.tensor(transformed_num_features, dtype=torch.float32)
#     )
#     error = torch.linalg.matrix_norm(dis_orig - dis_transformed, ord=diff_norm)

#     # Average the error across all samples in the batch
#     error = torch.mean(error)
#     return error

def _distance_error_stats(
    orig_point_matrix,
    transformed_point_matrix,
    diff_norm="fro",
    dist_metric="euclidean",
):
    """Encourage embeddings to retain global pairwise distance statistics.

    Rather than matching every entry in the distance matrix, this loss only
    aligns the mean and standard deviation of the pairwise distances. The
    softer constraint helps avoid embedding collapse while leaving supervised
    losses to determine fine-grained structure.

    Args:
        orig_point_matrix (Tensor): Tensor of shape (B, M, N) for the original data.
        transformed_point_matrix (Tensor): Tensor of shape (B, M, K) for the transformed embeddings.
        diff_norm (str or int): Ignored, kept for API compatibility.
        dist_metric (str): Distance metric passed to :func:`pairwise_distances`.

    Returns:
        Tensor: Scalar loss encouraging similar distance statistics.
    """

    if orig_point_matrix.dim() != 3 or transformed_point_matrix.dim() != 3:
        raise ValueError("The shape must be (B, M, N).")
    if orig_point_matrix.size(1) != transformed_point_matrix.size(1):
        raise ValueError(
            "The second dimension (number of points) must be the same for both input matrices."
        )

    dis_orig = pairwise_distances(orig_point_matrix, metric=dist_metric)
    dis_transformed = pairwise_distances(transformed_point_matrix, metric=dist_metric)

    if dis_orig.size(-1) < 2:
        return torch.tensor(0.0, device=orig_point_matrix.device)

    idx = torch.triu_indices(
        dis_orig.size(-2),
        dis_orig.size(-1),
        offset=1,
        device=dis_orig.device,
    )

    orig_vals = dis_orig[..., idx[0], idx[1]]
    trans_vals = dis_transformed[..., idx[0], idx[1]]

    orig_mean = orig_vals.mean(dim=-1)
    trans_mean = trans_vals.mean(dim=-1)
    mean_loss = (trans_mean - orig_mean) ** 2

    orig_std = orig_vals.std(dim=-1, unbiased=False)
    trans_std = trans_vals.std(dim=-1, unbiased=False)
    std_loss = (trans_std - orig_std) ** 2

    return (mean_loss + std_loss).mean()

def distance_error(
    orig_point_matrix,
    transformed_point_matrix,
    diff_norm="fro",
    dist_metric="euclidean",
    alpha=0.5,
    align: str = "scale",   # 新增，默认做尺度对齐    
):
    """Blend matrix-level and statistical distance regularisers.

    Args:
        orig_point_matrix (Tensor): Original features of shape (B, M, N).
        transformed_point_matrix (Tensor): Transformed embeddings of shape (B, M, K).
        diff_norm (str or int): Passed to the matrix-based loss for compatibility.
        dist_metric (str): Distance metric passed to the underlying losses.
        alpha (float): Weight for the matrix loss; statistical loss uses (1-alpha).

    Returns:
        Tensor: Scalar loss combining both regularisers.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    
    loss_matrix = _distance_error_matrix(
        orig_point_matrix,
        transformed_point_matrix,
        diff_norm=diff_norm,
        dist_metric=dist_metric,
        align=align,
    )
    loss_stats = _distance_error_stats(
        orig_point_matrix,
        transformed_point_matrix,
        diff_norm=diff_norm,
        dist_metric=dist_metric,
    )
    return alpha * loss_matrix + (1 - alpha) * loss_stats


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distances.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, D) or (N, D).

    Returns:
        torch.Tensor: Pairwise cosine distance matrix.
    """
    # Normalize the vectors to unit length
    x_norm = F.normalize(x, p=2, dim=-1)

    # Compute cosine similarity
    if x.dim() == 3:  # Batched case (B, N, D)
        cosine_sim = torch.bmm(x_norm, x_norm.transpose(-2, -1))
    else:  # Single case (N, D)
        cosine_sim = torch.mm(x_norm, x_norm.t())

    # Convert similarity to distance
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def pairwise_distances(embeddings, metric="euclidean", epsilon=1e-6):
    """
    Compute pairwise distances between embeddings using various distance metrics.

    Args:
        embeddings (Tensor): Input tensor of shape (B, N, n_features) representing the embeddings.
        metric (str): Distance metric to use. Options are "euclidean", "manhattan", "inf", "cosine", "poincare".
        epsilon (float): Values below this threshold will be clamped to zero.

    Returns:
        Tensor: Pairwise distance matrix of shape (B, N, N).

    Raises:
        ValueError: If the metric is unsupported.
    """
    p_norm_dict = {"euclidean": 2, "manhattan": 1, "inf": float("inf")}

    # If embeddings is numpy array, transform it to tensor
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    if metric in p_norm_dict:
        distance_matrix = torch.cdist(embeddings, embeddings, p=p_norm_dict[metric])

    elif metric == "cosine":
        return pairwise_cosine_distance(embeddings)

    # elif metric == "poincare":
    #     return pairwise_poincare_distance(embeddings, eps=epsilon)
    else:
        raise ValueError(
            "Unsupported metric. Choose 'cosine', 'euclidean', 'manhattan', or 'poincare'."
        )

    return distance_matrix


# def pairwise_distances(embeddings, metric="euclidean", epsilon=1e-6):
#     """
#     Compute pairwise distances between embeddings using the specified metric.
#     Memory-efficient implementation that avoids creating large intermediate tensors.

#     Args:
#         embeddings (torch.Tensor): Tensor of shape (B, N, D) or (N, D) representing the embeddings.
#         metric (str): Distance metric ('euclidean', 'cosine', 'manhattan').
#         epsilon (float): Small value for numerical stability.

#     Returns:
#         torch.Tensor: Pairwise distance matrix of shape (B, N, N) or (N, N).
#     """
#     if metric == "euclidean":
#         if embeddings.dim() == 3:  # Batched case (B, N, D)
#             B, N, D = embeddings.shape
#             # Use the mathematical identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
#             # This avoids creating the large (B, N, D, N) tensor

#             # Compute squared norms: (B, N)
#             norms_sq = torch.sum(embeddings**2, dim=2)

#             # Compute dot products: (B, N, N)
#             dot_products = torch.bmm(embeddings, embeddings.transpose(1, 2))

#             # Use broadcasting to compute ||a||^2 + ||b||^2 - 2<a,b>
#             distances_sq = (
#                 norms_sq.unsqueeze(2) + norms_sq.unsqueeze(1) - 2 * dot_products
#             )

#             # Clamp to avoid negative values due to numerical errors
#             distances_sq = torch.clamp(distances_sq, min=0)
#             distances = torch.sqrt(distances_sq + epsilon)

#         else:  # Single case (N, D)
#             N, D = embeddings.shape
#             # Same approach for non-batched case
#             norms_sq = torch.sum(embeddings**2, dim=1)  # (N,)
#             dot_products = torch.mm(embeddings, embeddings.t())  # (N, N)
#             distances_sq = (
#                 norms_sq.unsqueeze(1) + norms_sq.unsqueeze(0) - 2 * dot_products
#             )
#             distances_sq = torch.clamp(distances_sq, min=0)
#             distances = torch.sqrt(distances_sq + epsilon)

#     elif metric == "cosine":
#         distances = pairwise_cosine_distance(embeddings)

#     elif metric == "manhattan":
#         if embeddings.dim() == 3:  # Batched case (B, N, D)
#             # For Manhattan distance, we still need to expand, but we can do it more efficiently
#             # by processing in chunks if needed
#             B, N, D = embeddings.shape
#             if D > 1000:  # For high-dimensional data, use chunked computation
#                 distances = torch.zeros(B, N, N, device=embeddings.device)
#                 chunk_size = 1000
#                 for i in range(0, D, chunk_size):
#                     end_idx = min(i + chunk_size, D)
#                     chunk = embeddings[:, :, i:end_idx]  # (B, N, chunk_size)
#                     x1 = chunk.unsqueeze(3)  # (B, N, chunk_size, 1)
#                     x2 = chunk.unsqueeze(2)  # (B, N, 1, chunk_size)
#                     distances += torch.sum(torch.abs(x1 - x2), dim=2)
#             else:
#                 x1 = embeddings.unsqueeze(3)  # (B, N, D, 1)
#                 x2 = embeddings.unsqueeze(2)  # (B, N, 1, D)
#                 distances = torch.sum(torch.abs(x1 - x2), dim=2)
#         else:  # Single case (N, D)
#             N, D = embeddings.shape
#             if D > 1000:  # For high-dimensional data, use chunked computation
#                 distances = torch.zeros(N, N, device=embeddings.device)
#                 chunk_size = 1000
#                 for i in range(0, D, chunk_size):
#                     end_idx = min(i + chunk_size, D)
#                     chunk = embeddings[:, i:end_idx]  # (N, chunk_size)
#                     x1 = chunk.unsqueeze(1)  # (N, 1, chunk_size)
#                     x2 = chunk.unsqueeze(0)  # (1, N, chunk_size)
#                     distances += torch.sum(torch.abs(x1 - x2), dim=2)
#             else:
#                 x1 = embeddings.unsqueeze(1)  # (N, 1, D)
#                 x2 = embeddings.unsqueeze(0)  # (1, N, D)
#                 distances = torch.sum(torch.abs(x1 - x2), dim=2)

#     else:
#         raise ValueError(f"Unsupported distance metric: {metric}")

#     return distances


def reconstruct_from_dm(dm, node_names, method, unrooted=True):
    """
    Reconstruct a tree from a distance matrix using the specified method.

    Args:
        dm (numpy.ndarray): Distance matrix.
        node_names (list): List of node names corresponding to the distance matrix.
        method (str): Reconstruction method ('nj' for neighbor joining).
        unrooted (bool): Whether to return an unrooted tree.

    Returns:
        ete3.Tree: Reconstructed tree.
    """
    # The dm_to_etree function may not support the unrooted parameter
    # Let's call it without that parameter for compatibility
    return dm_to_etree(dm, node_names, method=method)


def compare_trees(tree1, tree2, unrooted_trees=False):
    """
    Compare two trees using Robinson-Foulds distance.

    Args:
        tree1, tree2: ete3.Tree objects to compare.
        unrooted_trees (bool): Whether trees are unrooted.

    Returns:
        dict: Dictionary containing RF distance and related metrics.
    """
    if unrooted_trees:
        rf_distance = tree1.robinson_foulds(tree2, unrooted_trees=True)[0]
        # For unrooted trees, the maximum RF distance is 2 * (n - 3)
        # where n is the number of leaves
        max_rf = 2 * (len(tree1.get_leaves()) - 3)
    else:
        rf_distance = tree1.robinson_foulds(tree2)[0]
        # For rooted trees, the maximum RF distance is 2 * (n - 2)
        max_rf = 2 * (len(tree1.get_leaves()) - 2)

    if max_rf == 0:
        relative_rf = 0.0
    else:
        relative_rf = rf_distance / max_rf

    return {
        "rf_distance": rf_distance,
        "max_rf": max_rf,
        "relative_rf": relative_rf,
    }


def check_embedding(dataset, model, dist_metric, device):
    """
    Check the embedding produced by the model for a given dataset.

    Args:
        dataset: Dataset object containing the data.
        model: Neural network model.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.

    Returns:
        tuple: (embeddings, distance_matrix, node_names)
    """
    model.eval()
    with torch.no_grad():
        # Get the node matrix data
        node_mtx_dict = dataset.get_node_mtx()
        pts_mtx = (
            torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float)
            .unsqueeze(0)
            .to(device)
        )
        node_names = node_mtx_dict["node_names"]

        # Get embeddings
        embeddings = model(pts_mtx)  # Shape: (1, N, D)

        # Compute distance matrix
        dm = pairwise_distances(embeddings, metric=dist_metric)
        dm = dm.squeeze(0).cpu().numpy()  # Shape: (N, N)

        return embeddings, dm, node_names


def train_reconstruct_eval(
    dataset, model, res_dict, dist_metric="euclidean", device="cpu", method="nj"
):
    """
    Evaluate reconstruction performance on training data.

    Args:
        dataset: Training dataset.
        model: Neural network model.
        res_dict: Results dictionary to update.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.
        method (str): Tree reconstruction method.
    """
    model.eval()
    with torch.no_grad():
        _, emb_dm, node_names = check_embedding(dataset, model, dist_metric, device)

        # Reconstruct tree from embedding
        emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

        # Compare with reference tree
        emb_topo_res = dataset.compare_trees(emb_tree, ref_tree="topology_tree")

        # Store result
        res_key = f"rf_emb_topo_train"
        res_dict[res_key][method].append(emb_topo_res["relative_rf"])


def test_reconstruct_eval(
    dataset, model, res_dict, dist_metric="euclidean", device="cpu", method="nj"
):
    """
    Evaluate reconstruction performance on test data.

    Args:
        dataset: Test dataset.
        model: Neural network model.
        res_dict: Results dictionary to update.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.
        method (str): Tree reconstruction method.
    """
    model.eval()
    with torch.no_grad():
        _, emb_dm, node_names = check_embedding(dataset, model, dist_metric, device)

        # Reconstruct tree from embedding
        emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

        # Compare with reference tree
        emb_topo_res = dataset.compare_trees(emb_tree, ref_tree="topology_tree")

        # Store result
        res_key = f"rf_emb_topo_test"
        res_dict[res_key][method].append(emb_topo_res["relative_rf"])


def test_unknown_reconstruct_eval(
    dataset, model, res_dict, dist_metric="euclidean", device="cpu", method="nj"
):
    """
    Evaluate reconstruction performance on unknown test data.

    Args:
        dataset: Unknown test dataset.
        model: Neural network model.
        res_dict: Results dictionary to update.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.
        method (str): Tree reconstruction method.
    """
    model.eval()
    with torch.no_grad():
        _, emb_dm, node_names = check_embedding(dataset, model, dist_metric, device)

        # Reconstruct tree from embedding
        emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

        # Compare with reference tree
        emb_topo_res = dataset.compare_trees(emb_tree, ref_tree="topology_tree")

        # Store result
        res_key = f"rf_emb_topo_test_unknown"
        res_dict[res_key][method].append(emb_topo_res["relative_rf"])
