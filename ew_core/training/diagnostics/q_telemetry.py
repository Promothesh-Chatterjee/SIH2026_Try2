"""Q-value, target-Q, and gradient telemetry diagnostics module."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import torch


class QTelemetry:
    """Monitors online Q-values, unclamped target-Q values, TD errors, and gradient norms.

    Staged Limits:
      - max_q < 25.0: Healthy
      - 25.0 <= max_q < 40.0: Diagnostic Warning
      - 40.0 <= max_q <= 50.0: Critical Warning
      - max_q > 50.0: Hard Safety Halt
      - NaN / Inf in any quantity: Hard Safety Halt
    """

    def __init__(self, halt_ceiling: float = 50.0) -> None:
        self.halt_ceiling = halt_ceiling

    def evaluate_step(
        self,
        q_online: torch.Tensor,
        target_q_unclamped: torch.Tensor,
        td_errors: torch.Tensor,
        online_model: torch.nn.Module,
        unclipped_grad_norm: float | None = None,
        clipped_grad_norm: float | None = None,
    ) -> Dict[str, Any]:
        """Compute full Q diagnostic telemetry for one optimization step."""
        q_detached = q_online.detach()
        tq_detached = target_q_unclamped.detach()
        td_detached = td_errors.detach()

        # Check for NaN / Inf
        has_nan = bool(
            torch.isnan(q_detached).any()
            or torch.isnan(tq_detached).any()
            or torch.isnan(td_detached).any()
        )
        has_inf = bool(
            torch.isinf(q_detached).any()
            or torch.isinf(tq_detached).any()
            or torch.isinf(td_detached).any()
        )

        q_mean = float(q_detached.mean().item()) if not has_nan else float("nan")
        q_std = float(q_detached.std().item()) if not has_nan else float("nan")
        q_min = float(q_detached.min().item()) if not has_nan else float("nan")
        q_max = float(q_detached.max().item()) if not has_nan else float("nan")

        target_q_mean = float(tq_detached.mean().item()) if not has_nan else float("nan")
        target_q_max = float(tq_detached.max().item()) if not has_nan else float("nan")
        target_q_min = float(tq_detached.min().item()) if not has_nan else float("nan")

        td_mean = float(td_detached.mean().item()) if not has_nan else float("nan")
        td_std = float(td_detached.std().item()) if not has_nan else float("nan")

        # Parameter norm
        param_norm = 0.0
        for p in online_model.parameters():
            if p.data is not None:
                param_norm += float(p.data.norm(2).item() ** 2)
        param_norm = float(np.sqrt(param_norm))

        # Classify warnings and halts
        halt_reasons: List[str] = []
        warning_reasons: List[str] = []

        if has_nan:
            halt_reasons.append("NaN detected in Q-values, target-Q, or TD errors")
        if has_inf:
            halt_reasons.append("Inf detected in Q-values, target-Q, or TD errors")

        if not has_nan and not has_inf:
            if q_max > self.halt_ceiling:
                halt_reasons.append(f"Hard safety halt: max_q={q_max:.2f} > {self.halt_ceiling:.1f}")
            elif q_max >= 40.0:
                warning_reasons.append(f"Critical warning: max_q={q_max:.2f} in [40.0, 50.0]")
            elif q_max >= 25.0:
                warning_reasons.append(f"Diagnostic warning: max_q={q_max:.2f} in [25.0, 40.0]")

        return {
            "q_mean": q_mean,
            "q_std": q_std,
            "q_min": q_min,
            "q_max": q_max,
            "target_q_unclamped_mean": target_q_mean,
            "target_q_unclamped_max": target_q_max,
            "target_q_unclamped_min": target_q_min,
            "td_mean": td_mean,
            "td_std": td_std,
            "unclipped_grad_norm": unclipped_grad_norm if unclipped_grad_norm is not None else 0.0,
            "clipped_grad_norm": clipped_grad_norm if clipped_grad_norm is not None else 0.0,
            "param_norm": param_norm,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "diagnostic_warning": len(warning_reasons) > 0,
            "safety_halt": len(halt_reasons) > 0,
            "halt_reasons": halt_reasons,
            "warning_reasons": warning_reasons,
        }
