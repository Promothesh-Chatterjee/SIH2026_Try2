"""
Phase 8 Fail-Closed Operational Readiness Validation Helper.

Provides strict, fail-closed metric validation, finite-number enforcement,
and standardized scorecard generation for operational readiness gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional


class ReadinessStatus(str, Enum):
    """Permitted readiness statuses under Phase 8 contract."""
    READY = "READY"
    NOT_READY = "NOT_READY"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    EVALUATION_ERROR = "EVALUATION_ERROR"


class ReadinessError(Exception):
    """Base exception for operational readiness validation failures."""
    pass


class MissingMetricError(ReadinessError):
    """Raised when a required metric is absent or None."""
    pass


class NonFiniteMetricError(ReadinessError):
    """Raised when a metric is NaN, Inf, or non-finite."""
    pass


class IntegrityFailureError(ReadinessError):
    """Raised when checkpoint provenance, signature, or approval fails."""
    pass


def require_present_metric(container: Dict[str, Any], key: str, metric_name: str) -> Any:
    """
    Ensure metric exists in container and is not None. Fail closed otherwise.
    """
    if not isinstance(container, dict):
        raise MissingMetricError(f"Container for '{metric_name}' is not a dict: {type(container)}")
    if key not in container or container[key] is None:
        raise MissingMetricError(f"Missing required metric: '{metric_name}' (key '{key}' absent or None)")
    return container[key]


def require_finite_metric(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """
    Ensure metric is a valid finite float (not boolean, not NaN, not Inf).
    Optionally check min_val and max_val bounds. Fail closed otherwise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NonFiniteMetricError(f"Metric '{name}' must be numeric (int/float), got {type(value).__name__}: {value}")
    val_float = float(value)
    if not math.isfinite(val_float) or math.isnan(val_float):
        raise NonFiniteMetricError(f"Metric '{name}' is non-finite or NaN: {val_float}")
    if min_val is not None and val_float < min_val:
        raise ValueError(f"Metric '{name}' value {val_float} is below minimum allowed {min_val}")
    if max_val is not None and val_float > max_val:
        raise ValueError(f"Metric '{name}' value {val_float} is above maximum allowed {max_val}")
    return val_float


def check_threshold(val: float, target: float, op: str, name: str) -> tuple[bool, str]:
    """
    Check val against target using op ('>=', '<=', '>', '<', '==').
    Returns (passed, explanation).
    """
    if op == ">=":
        passed = bool(val >= target)
        sym = ">="
    elif op == "<=":
        passed = bool(val <= target)
        sym = "<="
    elif op == ">":
        passed = bool(val > target)
        sym = ">"
    elif op == "<":
        passed = bool(val < target)
        sym = "<"
    elif op == "==":
        passed = bool(math.isclose(val, target, rel_tol=1e-5, abs_tol=1e-7))
        sym = "=="
    else:
        raise ValueError(f"Unsupported comparison operator: {op}")

    msg = f"{name}: {val} {sym} {target} -> {'PASS' if passed else 'FAIL'}"
    return passed, msg


@dataclass
class OperationalReadinessVerdict:
    """Standardized Phase 8 readiness scorecard and verdict."""
    schema_version: str = "phase8"
    readiness_status: str = ReadinessStatus.NOT_READY.value
    all_gates_passed: bool = False
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    gates: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
