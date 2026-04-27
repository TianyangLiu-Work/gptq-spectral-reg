"""Power iteration for spectral norm estimation of delta_W.

Given a weight error matrix delta_W = q_weight - W,
estimates the top right singular vector v and singular value sigma.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def top_singular_vector(
    delta_w: torch.Tensor,
    n_iter: int = 20,
    seed: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Estimate the top right singular vector and singular value of delta_W.

    Uses power iteration on A^T A where A = delta_W.

    Args:
        delta_w: Weight error matrix [out_dim, in_dim].
        n_iter: Number of power iterations.
        seed: Random seed for initialization (None = no seeding).

    Returns:
        v: Top right singular vector [in_dim] (unit norm).
        sigma: Top singular value (spectral norm of delta_W).
    """
    w = delta_w.detach().float()
    out_dim, in_dim = w.shape

    if out_dim == 0 or in_dim == 0:
        return torch.zeros(in_dim), 0.0

    # Flatten if not a proper 2D matrix
    if out_dim == 1 or in_dim == 1:
        sigma = float(torch.linalg.vector_norm(w.reshape(-1), ord=2).item())
        v = torch.zeros(in_dim)
        v[0] = 1.0
        return v, sigma

    # Initialize random vector
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

    v = torch.randn(in_dim, generator=gen, device=w.device, dtype=torch.float32)

    for _ in range(n_iter):
        v_norm = torch.linalg.vector_norm(v, ord=2)
        if v_norm.item() < 1e-12:
            v = torch.randn(in_dim, device=w.device, dtype=torch.float32)
            continue
        v = v / v_norm

        # v = A^T A v  (power iteration on A^T A)
        u = w @ v           # [out_dim]
        u_norm = torch.linalg.vector_norm(u, ord=2)
        if u_norm.item() < 1e-12:
            break
        u = u / u_norm
        v = w.T @ u         # [in_dim]

    # Normalize and compute sigma
    v_norm = torch.linalg.vector_norm(v, ord=2)
    if v_norm.item() > 1e-12:
        v = v / v_norm
    sigma = float(torch.linalg.vector_norm(w @ v, ord=2).item())

    return v, sigma


@torch.no_grad()
def compute_spectral_norm(
    tensor: torch.Tensor,
    n_iter: int = 20,
) -> float:
    """Compute spectral norm (largest singular value) via power iteration.

    Args:
        tensor: Any tensor. Will be reshaped to 2D if necessary.
        n_iter: Number of power iterations.

    Returns:
        sigma: Spectral norm.
    """
    _, sigma = top_singular_vector(tensor, n_iter=n_iter)
    return sigma


@torch.no_grad()
def compute_all_spectral_norms(
    deltas: dict[str, torch.Tensor],
    n_iter: int = 20,
) -> dict[str, float]:
    """Compute spectral norms for a dict of delta weights.

    Args:
        deltas: Dict of {name: delta_weight}.
        n_iter: Power iterations.

    Returns:
        specs: Dict of {name: spectral_norm}.
    """
    return {
        name: compute_spectral_norm(delta, n_iter=n_iter)
        for name, delta in deltas.items()
    }
