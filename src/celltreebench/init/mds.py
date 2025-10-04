import torch
from typing import Tuple


def classical_mds(D: torch.Tensor, out_dim: int, treat_as_squared: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classical MDS via double-centering.

    Args:
        D: (N,N) or (B,N,N) distance matrix. If batched, only the first is used.
        out_dim: target embedding dimension K
        treat_as_squared: if True, interpret D as squared distances; otherwise square it.

    Returns:
        (Y, evals): Y is (N,out_dim) embedding, evals are top eigenvalues
    """
    if D.dim() == 3:
        D = D[0]
    N = D.shape[0]
    device = D.device
    dtype = D.dtype

    if treat_as_squared:
        D2 = D
    else:
        D2 = D ** 2

    I = torch.eye(N, device=device, dtype=dtype)
    one = torch.ones((N, N), device=device, dtype=dtype) / N
    J = I - one
    # B = -0.5 J D^2 J
    B = -0.5 * J @ D2 @ J

    # eigen-decomposition
    # We take top-K eigenpairs of symmetric matrix
    evals, evecs = torch.linalg.eigh(B)
    evals = evals.clamp_min(0)
    idx = torch.argsort(evals, descending=True)
    idx = idx[: out_dim]
    L = torch.diag(torch.sqrt(evals[idx]))
    V = evecs[:, idx]
    Y = V @ L
    return Y, evals[idx]


__all__ = ["classical_mds"]

