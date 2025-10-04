import torch
from typing import Dict, Optional, Tuple


def _flatten_triu(dm: torch.Tensor) -> torch.Tensor:
    if dm.dim() == 2:
        dm = dm.unsqueeze(0)
    B, N, _ = dm.shape
    iu = torch.triu_indices(N, N, 1, device=dm.device)
    v = dm[:, iu[0], iu[1]]  # (B, P)
    return v


def _pairwise_sqeuclidean(E: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return pairwise squared Euclidean distances for E of shape (B,N,D) or (N,D)."""
    if E.dim() == 2:
        E = E.unsqueeze(0)
    # Use identity ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # This is numerically stable and avoids sqrt.
    B, N, D = E.shape
    norms = (E ** 2).sum(dim=2)  # (B,N)
    dots = torch.bmm(E, E.transpose(1, 2))  # (B,N,N)
    D2 = norms.unsqueeze(2) + norms.unsqueeze(1) - 2.0 * dots
    if eps > 0.0:
        D2 = D2.clamp_min(eps)
    return D2


def _linreg_affine(x: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve y ≈ α x + β in least-squares sense per-batch (x,y: (B,P)). Optional weights w: (B,P)."""
    B, P = x.shape
    if w is not None:
        # Weighted regression
        W = w
        sumW = W.sum(dim=1, keepdim=True).clamp_min(eps)
        x_bar = (W * x).sum(dim=1, keepdim=True) / sumW
        y_bar = (W * y).sum(dim=1, keepdim=True) / sumW
        x_c = x - x_bar
        y_c = y - y_bar
        Sxx = (W * x_c * x_c).sum(dim=1, keepdim=True).clamp_min(eps)
        Sxy = (W * x_c * y_c).sum(dim=1, keepdim=True)
        alpha = (Sxy / Sxx).squeeze(1)
        beta = (y_bar - alpha.unsqueeze(1) * x_bar).squeeze(1)
    else:
        x_bar = x.mean(dim=1, keepdim=True)
        y_bar = y.mean(dim=1, keepdim=True)
        x_c = x - x_bar
        y_c = y - y_bar
        Sxx = (x_c * x_c).sum(dim=1, keepdim=True).clamp_min(eps)
        Sxy = (x_c * y_c).sum(dim=1, keepdim=True)
        alpha = (Sxy / Sxx).squeeze(1)
        beta = (y_bar - alpha.unsqueeze(1) * x_bar).squeeze(1)
    return alpha, beta


def _linreg_scale(x: torch.Tensor, y: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """Solve y ≈ α x in least-squares sense per-batch. Optional weights w: (B,P). Returns α:(B,)."""
    if w is not None:
        Sxx = (w * x * x).sum(dim=1).clamp_min(eps)
        Sxy = (w * x * y).sum(dim=1)
    else:
        Sxx = (x * x).sum(dim=1).clamp_min(eps)
        Sxy = (x * y).sum(dim=1)
    alpha = Sxy / Sxx
    return alpha


def _huber(res: torch.Tensor, delta: float) -> torch.Tensor:
    a = res.abs()
    quad = 0.5 * a.pow(2)
    lin = delta * (a - 0.5 * delta)
    return torch.where(a <= delta, quad, lin)


def _bin_weights(d_vec: torch.Tensor, bins: int = 10, mode: str = "none", long_pair_boost: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pair weights for equalized distance bins (per-batch).
    Returns (weights, bin_ids). weights has shape (B,P), bin_ids (B,P) long.
    """
    if mode == "none" or bins <= 1:
        w = torch.ones_like(d_vec)
        return w, torch.zeros_like(d_vec, dtype=torch.long)

    B, P = d_vec.shape
    w = torch.ones_like(d_vec)
    bin_ids = torch.zeros_like(d_vec, dtype=torch.long)
    for b in range(B):
        v = d_vec[b]
        qs = torch.quantile(v, torch.linspace(0, 1, steps=bins + 1, device=v.device), interpolation="linear")
        qs[0] = v.min() - 1e-12
        qs[-1] = v.max() + 1e-12
        # Assign bins
        # bin k if in (qs[k], qs[k+1]]
        for k in range(bins):
            m = (v > qs[k]) & (v <= qs[k + 1])
            bin_ids[b, m] = k
        # equalize weights inversely to counts
        for k in range(bins):
            cnt = (bin_ids[b] == k).sum().item()
            if cnt > 0:
                w[b, bin_ids[b] == k] = float(P) / float(bins * cnt)
        # long-pair boost on the farthest bin
        if long_pair_boost and long_pair_boost > 1.0:
            m_last = bin_ids[b] == (bins - 1)
            if m_last.any():
                w[b, m_last] = w[b, m_last] * long_pair_boost
    return w, bin_ids


@torch.no_grad()
def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, w: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    if w is not None:
        sumW = w.sum(dim=1, keepdim=True).clamp_min(eps)
        y_bar = (w * y_true).sum(dim=1, keepdim=True) / sumW
        ss_tot = (w * (y_true - y_bar).pow(2)).sum(dim=1)
        ss_res = (w * (y_true - y_pred).pow(2)).sum(dim=1)
    else:
        y_bar = y_true.mean(dim=1, keepdim=True)
        ss_tot = ((y_true - y_bar).pow(2)).sum(dim=1)
        ss_res = ((y_true - y_pred).pow(2)).sum(dim=1)
    return 1.0 - (ss_res / ss_tot.clamp_min(eps))


def pairwise_l2sq_regression(
    E: torch.Tensor,
    d_true: torch.Tensor,
    align: str = "affine",  # 'affine' | 'scale' | 'none'
    huber_delta: float = 0.05,
    bins: int = 10,
    sampling_mode: str = "bucket_equal",  # 'bucket_equal' | 'none'
    long_pair_boost: float = 1.5,
    center: bool = True,
    ensure_positive_alpha: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Fully-supervised D-loss: regress squared Euclidean distances to true tree distances.

    Args:
        E: (B,N,D) or (N,D) embeddings
        d_true: (B,N,N) or (N,N) true tree distances
        align: 'affine' to solve α,β; 'scale' to solve α with β=0; 'none' uses α=1,β=0
        huber_delta: Huber threshold on residuals
        bins: number of quantile bins for equalized weighting
        sampling_mode: 'bucket_equal' for equalized per-bin weights, else 'none'
        long_pair_boost: multiplier for the farthest bin
        center: if True, zero-mean center embeddings per batch

    Returns:
        dict with keys: loss, alpha, beta, r2, rmse_by_bin (tensor[bins])
    """
    if E.dim() == 2:
        E = E.unsqueeze(0)
    if d_true.dim() == 2:
        d_true = d_true.unsqueeze(0)

    if center:
        E = E - E.mean(dim=1, keepdim=True)

    # Δ = ||xi - xj||^2
    D2 = _pairwise_sqeuclidean(E)
    y = _flatten_triu(D2)  # (B,P)
    x = _flatten_triu(d_true)  # (B,P) true tree distances

    # weights by distance bins
    if sampling_mode == "bucket_equal":
        w, bin_ids = _bin_weights(x, bins=bins, mode="bucket_equal", long_pair_boost=long_pair_boost)
    else:
        w = torch.ones_like(x)
        bin_ids = torch.zeros_like(x, dtype=torch.long)

    # alignment
    if align == "affine":
        alpha, beta = _linreg_affine(x, y, w=w)
    elif align == "scale":
        alpha = _linreg_scale(x, y, w=w)
        beta = torch.zeros_like(alpha)
    elif align == "none":
        alpha = torch.ones(y.shape[0], device=y.device, dtype=y.dtype)
        beta = torch.zeros_like(alpha)
    else:
        raise ValueError("align must be 'affine' | 'scale' | 'none'")

    if ensure_positive_alpha:
        alpha = alpha.clamp_min(1e-6)

    y_hat = alpha.unsqueeze(1) * x + beta.unsqueeze(1)
    resid = y - y_hat

    if huber_delta and huber_delta > 0:
        losses = _huber(resid, huber_delta)
    else:
        losses = 0.5 * resid.pow(2)

    # weighted mean
    loss = (w * losses).sum(dim=1) / (w.sum(dim=1).clamp_min(1e-8))
    loss = loss.mean()

    # R^2 per-batch then average
    r2 = _r2_score(y, y_hat, w=w).mean()

    # RMSE by bin (report only)
    rmse_bins = []
    B, P = y.shape
    for k in range(bins):
        mask = (bin_ids == k)
        if mask.any():
            rmse_k = torch.sqrt((resid[mask] ** 2).mean())
        else:
            rmse_k = torch.tensor(float('nan'), device=y.device, dtype=y.dtype)
        rmse_bins.append(rmse_k)
    rmse_by_bin = torch.stack(rmse_bins, dim=0) if rmse_bins else torch.empty(0, device=y.device)

    return {
        "loss": loss,
        "alpha": alpha.detach(),
        "beta": beta.detach(),
        "alpha_raw": alpha,
        "beta_raw": beta,
        "r2": r2.detach(),
        "rmse_by_bin": rmse_by_bin.detach(),
    }


__all__ = [
    "pairwise_l2sq_regression",
]
