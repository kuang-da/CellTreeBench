import torch
from typing import Dict, Optional, Tuple


class PairBucketWeighter:
    """
    Compute per-pair weights based on quantile bins of the reference distances.
    Intended to be lightweight for small N; uses full matrix and returns weights for the upper triangle.
    """

    def __init__(self, d_true: torch.Tensor, bins: int = 10, long_pair_boost: float = 1.0):
        if d_true.dim() == 3:
            d_true = d_true[0]
        self.N = d_true.shape[0]
        self.device = d_true.device
        self.dtype = d_true.dtype
        self.bins = max(1, int(bins))
        self.long_pair_boost = float(long_pair_boost)
        self._build(d_true)

    def _build(self, d_true: torch.Tensor):
        N = self.N
        iu = torch.triu_indices(N, N, 1, device=self.device)
        v = d_true[iu[0], iu[1]]
        if self.bins <= 1:
            self.weights = torch.ones_like(v)
            self.bin_ids = torch.zeros_like(v, dtype=torch.long)
            return
        qs = torch.quantile(v, torch.linspace(0, 1, steps=self.bins + 1, device=self.device), interpolation="linear")
        qs[0] = v.min() - 1e-12
        qs[-1] = v.max() + 1e-12
        bin_ids = torch.zeros_like(v, dtype=torch.long)
        for k in range(self.bins):
            m = (v > qs[k]) & (v <= qs[k + 1])
            bin_ids[m] = k
        w = torch.ones_like(v)
        P = v.numel()
        for k in range(self.bins):
            cnt = (bin_ids == k).sum().item()
            if cnt > 0:
                w[bin_ids == k] = float(P) / float(self.bins * cnt)
        if self.long_pair_boost and self.long_pair_boost > 1.0:
            m_last = bin_ids == (self.bins - 1)
            if m_last.any():
                w[m_last] *= self.long_pair_boost
        self.weights = w
        self.bin_ids = bin_ids

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weights, self.bin_ids


__all__ = ["PairBucketWeighter"]

