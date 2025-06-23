# src/celltreebench/utils/distance_metrics.py

import torch
import numpy as np


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return 1 - x_norm @ x_norm.T
    elif x.dim() == 3:
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return 1 - torch.bmm(x_norm, x_norm.transpose(1, 2))
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {x.shape}")


def ensure_within_unit_ball(x: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True)
    return x / torch.clamp(norms, min=1.0) * (1.0 - margin)


def pairwise_poincare_distance(x: torch.Tensor, eps=1e-6, margin=1e-3) -> torch.Tensor:
    x = ensure_within_unit_ball(x, margin=margin)
    if x.dim() == 2:
        sq = x.pow(2).sum(-1).unsqueeze(1)
        dist_sq = ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(-1)
        denom = (1 - sq) * (1 - sq.T)
        return torch.acosh(torch.clamp(1 + 2 * dist_sq / (denom + eps), min=1 + eps))
    elif x.dim() == 3:
        sq = x.pow(2).sum(-1)
        xi, xj = sq.unsqueeze(2), sq.unsqueeze(1)
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist_sq = diff.pow(2).sum(-1)
        denom = (1 - xi) * (1 - xj)
        return torch.acosh(torch.clamp(1 + 2 * dist_sq / (denom + eps), min=1 + eps))
    else:
        raise ValueError("Unsupported input dimensions")


def pairwise_distances(embeddings, metric="euclidean", epsilon=1e-6):
    p_norms = {"euclidean": 2, "manhattan": 1, "inf": float("inf")}
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    if metric in p_norms:
        return torch.cdist(embeddings, embeddings, p=p_norms[metric])
    elif metric == "cosine":
        return pairwise_cosine_distance(embeddings)
    elif metric == "poincare":
        return pairwise_poincare_distance(embeddings, eps=epsilon)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def flatten_symmetric_matrix(D):
    n = D.shape[0]
    return np.array([D[i, j] for i in range(n) for j in range(i + 1, n)]).reshape(-1, 1)


def unflatten_symmetric_matrix(vector):
    m = len(vector)
    n = int((1 + np.sqrt(1 + 8 * m)) // 2)
    D = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = vector[idx]
            idx += 1
    return D


__all__ = [
    "pairwise_distances",
    "pairwise_cosine_distance",
    "pairwise_poincare_distance",
    "ensure_within_unit_ball",
    "flatten_symmetric_matrix",
    "unflatten_symmetric_matrix",
]
