"""GPTQ: standard per-channel quantization + error propagation.

Two-step process:
1. GPTQ with per-channel (group_size=in_dim) quantization — the standard GPTQ algorithm
2. Then re-quantize to per-group for evaluation
"""

from __future__ import annotations

import torch


@torch.no_grad()
def gptq_quantize_layer_perchannel(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int = 4,
    damp: float = 1e-2,
) -> torch.Tensor:
    """GPTQ with per-channel quantization (1 scale per column)."""
    out_dim, in_dim = weight.shape
    w = weight.detach().float().clone()
    device = w.device
    H = hessian.float().to(device).clone()

    # Damping
    mean_diag = torch.diag(H).mean()
    H = H + damp * mean_diag * torch.eye(in_dim, device=H.device, dtype=H.dtype)

    # Cholesky
    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        H = H + damp * mean_diag * torch.eye(in_dim, device=H.device, dtype=H.dtype)
        L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)

    qmax = float(2 ** (bits - 1) - 1)

    # Per-channel scale
    scales = w.abs().max(dim=0, keepdim=True).values  # [1, in_dim]
    scales = scales / qmax
    scales = scales.clamp(min=1e-12)

    # Integer representation
    w_int = torch.round(w / scales)

    # GPTQ column-by-column
    for i1 in range(0, in_dim, 128):
        i2 = min(i1 + 128, in_dim)
        for j in range(i1, i2):
            col_int = w_int[:, j:j+1]
            col_clamped = col_int.clamp(-qmax, qmax)
            err = col_clamped - col_int  # integer error

            w_int[:, j:j+1] = col_clamped

            if j + 1 < in_dim:
                h_diag = H_inv[j, j].clamp(min=1e-12)
                cross = H_inv[j:j+1, j+1:]  # [1, remaining]

                # Propagation in integer space, accounting for scale differences
                s_j = scales[:, j:j+1]  # [1, 1]
                s_rem = scales[:, j+1:]  # [1, remaining]

                err_float = err * s_j  # [out_dim, 1]
                delta_float = (err_float / h_diag) @ cross  # [out_dim, remaining]
                delta_int = delta_float / s_rem  # [out_dim, remaining]
                w_int[:, j+1:] += delta_int

    # Dequantize
    w_q = w_int * scales
    return w_q.to(dtype=weight.dtype)


def requantize_to_groupwise(
    w_q: torch.Tensor,
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """Re-quantize a weight matrix to per-group format."""
    out_dim, in_dim = w_q.shape
    qmax = float(2 ** (bits - 1) - 1)
    num_groups = (in_dim + group_size - 1) // group_size
    pad_len = num_groups * group_size - in_dim

    w = w_q.clone()
    if pad_len > 0:
        w = torch.nn.functional.pad(w, (0, pad_len), "constant", 0.0)

    w_g = w.view(out_dim, num_groups, group_size)
    abs_max = w_g.abs().amax(dim=2, keepdim=True)
    scale = abs_max / qmax
    scale = scale.clamp(min=1e-12)
    q = torch.clamp(torch.round(w_g / scale), -qmax, qmax)
    dq = q * scale
    dq = dq.reshape(out_dim, num_groups * group_size)
    if pad_len > 0:
        dq = dq[:, :in_dim]
    return dq.to(dtype=weight.dtype)


def gptq_quantize_layer(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    blocksize: int = 128,
    act_order: bool = False,
    damp: float = 1e-2,
) -> torch.Tensor:
    """GPTQ: per-channel GPTQ, then re-quantize to per-group."""
    # Step 1: Per-channel GPTQ
    w_q = gptq_quantize_layer_perchannel(weight, hessian, bits, damp)
    # Step 2: Re-quantize to group-wise
    w_g = requantize_to_groupwise(w_q, weight, bits, group_size)
    return w_g


def gather_hessians(handles, activations, device=None):
    hessians = {}
    for handle in handles:
        name = handle.canonical_name
        if name not in activations:
            continue
        act = activations[name].detach().float()
        if act.dim() == 3:
            act = act.reshape(-1, act.shape[-1])
        H = act.T @ act
        if device is not None:
            H = H.to(device)
        hessians[name] = H
    return hessians


def quantize_model_gptq(model, handles, hessians, bits=4, group_size=128,
                        blocksize=128, act_order=False, damp=1e-2, device=None):
    orig_device = next(model.parameters()).device
    compute_device = device or orig_device
    deltas = {}
    for handle in handles:
        name = handle.canonical_name
        if name not in hessians:
            continue
        w_gpu = handle.weight.detach().float().to(compute_device)
        H = hessians[name].to(compute_device)
        q_w = gptq_quantize_layer(w_gpu, H, bits=bits, group_size=group_size,
                                   blocksize=blocksize, act_order=act_order, damp=damp)
        delta = q_w.cpu() - handle.weight.detach().cpu()
        deltas[name] = delta
    return deltas
