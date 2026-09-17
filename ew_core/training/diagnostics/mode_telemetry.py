"""Mode-Specific Gradient & Loss Telemetry.

Computes isolated per-mode Bellman loss, masked backpropagated gradients,
and sample counts for Modes 0-4 to detect mode-value suppression or starved updates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_mode_isolated_gradients(
    online_drqn: nn.Module,
    loss_fn: nn.Module,
    q_chosen: torch.Tensor,
    targets: torch.Tensor,
    act_b: torch.Tensor,
    loss_mask: torch.Tensor,
    n_modes: int = 5,
) -> Dict[str, Any]:
    """Compute masked loss and isolated gradient norms for each action mode separately.

    Uses an isolated backward pass per mode (with retain_graph) to verify
    whether Mode m receives non-zero gradients into the advantage head.
    """
    mode_telemetry: Dict[str, Any] = {}
    actions = act_b.squeeze(-1) if act_b.dim() > loss_mask.dim() else act_b
    modes = actions % n_modes

    # Target parameters: check band_advantage_head specifically if present, else whole model
    if hasattr(online_drqn, "band_advantage_head"):
        target_params = [p for p in online_drqn.band_advantage_head.parameters() if p.requires_grad]
    else:
        target_params = [p for p in online_drqn.parameters() if p.requires_grad]

    for m in range(n_modes):
        m_mask = loss_mask & (modes == m)
        n_samples = int(m_mask.sum().item())

        if n_samples == 0:
            mode_telemetry[f"mode_{m}"] = {
                "valid_samples": 0,
                "masked_td_loss": 0.0,
                "masked_grad_norm": 0.0,
                "has_zero_gradient": True,
            }
            continue

        m_loss = loss_fn(q_chosen[m_mask], targets[m_mask].detach())

        # Isolated gradient check
        if target_params and m_loss.requires_grad:
            grads = torch.autograd.grad(
                m_loss,
                target_params,
                retain_graph=True,
                allow_unused=True,
            )
            valid_grads = [g for g in grads if g is not None]
            if valid_grads:
                total_norm = torch.norm(
                    torch.stack([torch.norm(g.detach(), 2) for g in valid_grads]), 2
                ).item()
            else:
                total_norm = 0.0
        else:
            total_norm = 0.0

        mode_telemetry[f"mode_{m}"] = {
            "valid_samples": n_samples,
            "masked_td_loss": float(m_loss.item()),
            "masked_grad_norm": float(total_norm),
            "has_zero_gradient": bool(total_norm < 1e-7),
        }

    return mode_telemetry
