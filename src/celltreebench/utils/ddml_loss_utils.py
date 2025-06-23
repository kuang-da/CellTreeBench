# src/celltreebench/utils/ddml_loss_utils.py

import torch
from .distance_metrics import pairwise_distances


def distance_error(orig, transformed, diff_norm="fro", dist_metric="euclidean"):
    dis_orig = pairwise_distances(orig, metric=dist_metric)
    dis_trans = pairwise_distances(transformed, metric=dist_metric)

    orig_features = orig.size(-1)
    trans_features = transformed.size(-1)

    dis_orig = dis_orig / torch.sqrt(torch.tensor(orig_features, dtype=torch.float32))
    dis_trans = dis_trans / torch.sqrt(
        torch.tensor(trans_features, dtype=torch.float32)
    )

    error = torch.linalg.matrix_norm(dis_orig - dis_trans, ord=diff_norm)
    return torch.mean(error)


def additivity_direction_error(dm1, dm2):
    dm1_sums = _quartet_sums(dm1)
    dm2_sums = _quartet_sums(dm2)

    _, idx1 = torch.topk(dm1_sums, 2, dim=1)
    _, idx2 = torch.topk(dm2_sums, 2, dim=1)

    min_idx1 = 3 - idx1[:, 0] - idx1[:, 1]
    min_idx2 = 3 - idx2[:, 0] - idx2[:, 1]
    return (min_idx1 != min_idx2).float().mean()


def additivity_error_ddml_tensor(
    point_matrix,
    dm_ref,
    quartet_idx,
    weight_close=1.0,
    weight_push=10.0,
    push_margin=0.5,
    dist_metric="euclidean",
    device="cpu",
    matching_mode="mismatched",
):
    point_matrix = point_matrix.to(device)
    dm_ref = dm_ref.to(device)
    dm = pairwise_distances(point_matrix, metric=dist_metric).squeeze()
    dm_ref = dm_ref.squeeze()

    idx_rows = quartet_idx.unsqueeze(2).expand(-1, -1, 4)
    idx_cols = quartet_idx.unsqueeze(1).expand(-1, 4, -1)
    dm_q = dm[idx_rows, idx_cols]
    dm_r = dm_ref[idx_rows, idx_cols]

    sum_q = _quartet_sums(dm_q)
    sum_r = _quartet_sums(dm_r)

    _, idx_q = torch.topk(sum_q, 2, dim=1)
    _, idx_r = torch.topk(sum_r, 2, dim=1)

    if matching_mode == "mismatched":
        mask = torch.any(idx_q != idx_r, dim=1)
    elif matching_mode == "matched":
        mask = torch.all(idx_q == idx_r, dim=1)
    elif matching_mode == "all":
        mask = torch.ones(sum_q.size(0), dtype=torch.bool, device=device)
    else:
        raise ValueError("Invalid matching_mode")

    loss_close = loss_push = torch.tensor(0.0, device=device)
    if mask.any():
        sums = sum_q[mask]
        ref_idx = idx_r[mask]
        top2 = sums.gather(1, ref_idx)
        lowest_idx = 3 - ref_idx.sum(1)
        lowest = sums[torch.arange(sums.size(0)), lowest_idx]

        loss_close = torch.mean(torch.abs(top2[:, 0] - top2[:, 1]))
        avg_top2 = top2.mean(1)
        loss_push = torch.mean(torch.relu((avg_top2 + push_margin) - lowest))

    total = weight_close * loss_close + weight_push * loss_push
    return total, loss_close, loss_push, idx_r


def _quartet_sums(dm):
    return torch.stack(
        [
            dm[:, 0, 1] + dm[:, 2, 3],
            dm[:, 0, 2] + dm[:, 1, 3],
            dm[:, 0, 3] + dm[:, 1, 2],
        ],
        dim=1,
    )


__all__ = [
    "distance_error",
    "additivity_direction_error",
    "additivity_error_ddml_tensor",
]
