"""Group-wise quantization utilities for GPTQ.

Provides symmetric per-group quantization/dequantization of weight matrices.
"""

from __future__ import annotations

import torch


def quantize_weight_groupwise(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    sym: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight matrix using per-group symmetric quantization.

    Args:
        weight: Weight tensor of shape [out_dim, in_dim].
        bits: Number of bits (4 or 3).
        group_size: Group size along the in_dim axis.
        sym: If True, symmetric quantization (zero point = 0).

    Returns:
        q_weight: Dequantized weights (same shape, same dtype as input).
        scale: Scale factors, shape [out_dim, num_groups].
        zero_point: Zero point (None for symmetric), same shape as scale.
    """
    orig_dtype = weight.dtype
    w = weight.detach().float()
    out_dim, in_dim = w.shape

    if group_size <= 0 or group_size > in_dim:
        group_size = in_dim
    num_groups = (in_dim + group_size - 1) // group_size

    # Pad in_dim to be divisible by group_size
    pad_len = num_groups * group_size - in_dim
    if pad_len > 0:
        w = torch.nn.functional.pad(w, (0, pad_len), "constant", 0.0)

    # Reshape to [out_dim, num_groups, group_size]
    w_g = w.view(out_dim, num_groups, group_size)

    if sym:
        qmax = float(2 ** (bits - 1) - 1)
        # Compute scale per group as max absolute value / qmax
        abs_max = w_g.abs().amax(dim=2, keepdim=True)
        scale = abs_max / qmax
        scale = scale.clamp(min=1e-12)
        # Quantize
        q = torch.clamp(torch.round(w_g / scale), -qmax, qmax)
        # Dequantize
        dq = q * scale
        dq = dq.reshape_as(w)
        # Remove padding
        if pad_len > 0:
            dq = dq[:, :in_dim].contiguous()
        zero_point = None
    else:
        raise NotImplementedError("Asymmetric quantization not yet implemented.")

    q_weight = dq.to(dtype=orig_dtype)
    return q_weight, scale.squeeze(-1), zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    sym: bool = True,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Simple per-tensor quantization.

    Args:
        tensor: Input tensor.
        bits: Number of bits.
        sym: Symmetric quantization.

    Returns:
        q_tensor: Dequantized tensor.
        scale: Scale factor.
        zero_point: Zero point (None for symmetric).
    """
    t = tensor.detach().float()
    if sym:
        qmax = float(2 ** (bits - 1) - 1)
        if scale is None:
            scale = t.abs().max() / qmax
            scale = scale.clamp(min=1e-12)
        q = torch.clamp(torch.round(t / scale), -qmax, qmax)
        dq = q * scale
        return dq.to(dtype=tensor.dtype), scale, None
    else:
        raise NotImplementedError("Asymmetric quantization not yet implemented.")
