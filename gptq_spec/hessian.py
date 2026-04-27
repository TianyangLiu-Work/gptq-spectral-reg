"""Hessian construction and regularization for GPTQ.

Provides:
- build_hessian: raw Hessian from activations
- add_damping: standard GPTQ damping
- add_frobenius_reg: isotropic Frobenius regularization H' = H + lambda * I
- add_spectral_reg: rank-1 spectral approximation H' = H + lambda * vv^T
"""

from __future__ import annotations

import torch


def build_hessian(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Build GPTQ Hessian from activation matrix.

    H = X^T X   (accumulated outer products of input rows)
    mean_diag = mean(diag(H))

    Args:
        activations: Activation tensor [n_samples, in_dim] or [n_samples, seq_len, in_dim].

    Returns:
        H: Hessian matrix [in_dim, in_dim].
        mean_diag: Mean of Hessian diagonal.
    """
    act = activations.detach().float()
    if act.dim() == 3:
        # [n_samples, seq_len, in_dim] -> [n_samples * seq_len, in_dim]
        n, s, d = act.shape
        act = act.reshape(-1, d)

    H = act.T @ act
    mean_diag = float(torch.diag(H).mean().item())
    return H, mean_diag


def add_damping(
    H: torch.Tensor,
    damp: float,
    mean_diag: float | None = None,
) -> torch.Tensor:
    """Add standard GPTQ damping.

    H' = H + damp * mean(diag(H)) * I

    Args:
        H: Hessian matrix [d, d].
        damp: Damping coefficient.
        mean_diag: Mean of diagonal (computed if None).

    Returns:
        Damped Hessian.
    """
    d = H.shape[0]
    if mean_diag is None:
        mean_diag = float(torch.diag(H).mean().item())
    return H + damp * mean_diag * torch.eye(d, device=H.device, dtype=H.dtype)


def add_frobenius_reg(
    H: torch.Tensor,
    beta: float,
    mean_diag: float | None = None,
    base_damp: float = 0.01,
) -> tuple[torch.Tensor, float]:
    """Add Frobenius regularization to Hessian.

    H' = H + base_damp * mean_diag * I + beta * mean_diag * I
       = H + (base_damp + beta) * mean_diag * I

    Where beta is the regularization strength, normalized by mean_diag
    for scale invariance.

    Args:
        H: Hessian matrix [d, d].
        beta: Frobenius regularization strength.
        mean_diag: Mean of diagonal (computed if None).
        base_damp: Base GPTQ damping (default: 0.01).

    Returns:
        H_reg: Regularized Hessian.
        total_scale: The total damping scale factor (base_damp + beta) * mean_diag.
    """
    d = H.shape[0]
    if mean_diag is None:
        mean_diag = float(torch.diag(H).mean().item())

    total_damp = base_damp + beta
    scale = total_damp * mean_diag
    H_reg = H + scale * torch.eye(d, device=H.device, dtype=H.dtype)
    return H_reg, scale


def add_spectral_reg(
    H: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    mean_diag: float | None = None,
    base_damp: float = 0.01,
) -> tuple[torch.Tensor, float]:
    """Add spectral (rank-1) regularization to Hessian.

    H' = H + base_damp * mean_diag * I + beta * mean_diag * v @ v^T

    Where v is the top right singular vector of the quantization error (delta_W).
    This penalizes the weight error along the worst-case direction.

    Args:
        H: Hessian matrix [d, d].
        v: Top right singular vector of delta_W, shape [d].
        beta: Spectral regularization strength.
        mean_diag: Mean of diagonal (computed if None).
        base_damp: Base GPTQ damping (default: 0.01).

    Returns:
        H_reg: Regularized Hessian.
        scale: The spectral regularization scale.
    """
    d = H.shape[0]
    if mean_diag is None:
        mean_diag = float(torch.diag(H).mean().item())

    # Base damping first
    H_reg = H + base_damp * mean_diag * torch.eye(d, device=H.device, dtype=H.dtype)

    # Rank-1 spectral correction
    scale = beta * mean_diag
    H_reg = H_reg + scale * torch.outer(v, v)

    return H_reg, scale


def build_hessian_with_options(
    activations: torch.Tensor,
    method: str = "none",
    beta: float = 0.01,
    v: torch.Tensor | None = None,
    base_damp: float = 0.01,
) -> torch.Tensor:
    """One-stop shop: build Hessian with optional regularization.

    Args:
        activations: Activation tensor.
        method: One of "none", "frobenius", "spectral", "stronger_damping".
        beta: Regularization strength.
        v: Right singular vector (required for "spectral").
        base_damp: Base damping.

    Returns:
        H: (Regularized) Hessian.
    """
    H, mean_diag = build_hessian(activations)

    if method == "none" or method == "baseline":
        return add_damping(H, base_damp, mean_diag)
    elif method == "stronger_damping":
        return add_damping(H, beta, mean_diag)
    elif method == "frobenius":
        H_reg, _ = add_frobenius_reg(H, beta, mean_diag, base_damp)
        return H_reg
    elif method == "spectral":
        if v is None:
            raise ValueError("Spectral method requires v (top right singular vector).")
        H_reg, _ = add_spectral_reg(H, v, beta, mean_diag, base_damp)
        return H_reg
    else:
        raise ValueError(f"Unknown method: {method}")
