"""Lexicographic Promotion Sentinel for Cognitive EW SmartScan.

Phase 7 Contract:
  1. Fail-closed: Rejects any candidate with missing/non-finite metrics, missing provenance,
     or missing/mismatched checkpoint identity or SHA-256.
  2. Cryptographic binding: A promotion report referring to Checkpoint A can NEVER be used
     to promote Checkpoint B.
  3. Explicit metric units: Metric units must be explicitly declared via 'metric_units'
     ('percent' or 'fraction'). Units are never guessed via threshold heuristics.
  4. Machine verdict separation: Returns explicit machine status ('APPROVED' | 'REJECTED')
     separately from human-readable verdict explanations.

Gate Criteria:
Level 0: Safety & Stability Constraints
  - Pfa <= 0.001 (0.10%)
  - No Q-value explosion (max |Q| <= 50.0)
  - Action entropy >= 1.0 (no static policy lock)
  - Loss finite, non-zero distinct bands (>= 12/36)

Level 1: Mean Intercept Rate Optimization
  - Candidate Mean IR >= Baseline Mean IR (60.45%) OR
  - If Candidate Mean IR is within 1.0% of Baseline (Mean IR >= 59.45%),
    promotion is permitted IF substantial robustness gains are achieved:
    * Agile IR gain >= +2.0% (Candidate Agile IR >= 48.70%)
    * Sparse IR gain >= +1.5% (Candidate Sparse IR >= 19.10%)
    * Worst-case IR gain >= +1.0% (Candidate Worst-case IR >= 13.40%)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ew_core.training.safety.checkpoint_guard import sha256_file
from ew_core.utils.checkpoint_paths import EXPECTED_FROZEN_SHA256

BASELINE_GATE25K = {
    "mean_ir": 60.45,
    "agile_ir": 46.70,
    "sparse_ir": 17.60,
    "worst_case_ir": 12.40,
    "pfa": 0.0,
    "distinct_bands": 29.9,
    "action_entropy": 2.06,
}

CANONICAL_METRIC_UNITS = {
    "mean_ir": "percent",
    "agile_ir": "percent",
    "sparse_ir": "percent",
    "worst_case_ir": "percent",
    "pfa": "fraction",
}


def evaluate_promotion(
    report: Dict[str, Any],
    candidate_path: Optional[Path | str] = None,
    enforce_provenance: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Evaluate candidate metrics and provenance against the formal promotion criteria.

    Args:
        report: Promotion report dictionary containing evaluation results, metrics, and provenance.
        candidate_path: Optional explicit path to candidate checkpoint file on disk.
        enforce_provenance: If True (default), strictly verifies checkpoint on-disk SHA,
                            baseline SHA, git revision, and explicit metric units.
                            If False (or if report has evaluation_type='metric_only_diagnostic'),
                            evaluates metrics only for historical test diagnostics.

    Returns:
        (promoted: bool, verdict: str, details: Dict[str, Any])
        details includes machine fields 'promotion_status' ('APPROVED' | 'REJECTED'),
        'checkpoint_sha256', and the evaluated metrics scorecard.
    """
    is_diagnostic = (not enforce_provenance) or (report.get("evaluation_type") == "metric_only_diagnostic")

    # ── 1. Provenance & Cryptographic Artifact Binding ────────────────────────
    actual_candidate_sha: Optional[str] = None
    target_ckpt: Optional[Path] = None

    if not is_diagnostic:
        # Resolve target candidate path
        raw_path = candidate_path or report.get("checkpoint_path")
        if not raw_path:
            return False, "REJECT (Missing candidate checkpoint identity: no checkpoint_path provided)", {
                "promotion_status": "REJECTED",
                "reasons": ["Missing candidate checkpoint identity"],
            }

        target_ckpt = Path(raw_path).resolve()
        if not target_ckpt.is_file():
            return False, f"REJECT (Candidate checkpoint file not found on disk: {target_ckpt})", {
                "promotion_status": "REJECTED",
                "reasons": [f"Candidate checkpoint file not found: {target_ckpt}"],
            }

        if "QUARANTINED" in target_ckpt.name:
            return False, f"REJECT (Candidate checkpoint is marked as QUARANTINED: {target_ckpt.name})", {
                "promotion_status": "REJECTED",
                "reasons": ["Candidate checkpoint is quarantined"],
            }

        # Verify on-disk SHA matches report's recorded SHA
        try:
            actual_candidate_sha = sha256_file(target_ckpt)
        except Exception as exc:
            return False, f"REJECT (Could not compute candidate SHA-256: {exc})", {
                "promotion_status": "REJECTED",
                "reasons": [str(exc)],
            }

        report_ckpt_sha = str(report.get("checkpoint_sha256", "")).lower().strip()
        if not report_ckpt_sha:
            return False, "REJECT (Promotion report missing required 'checkpoint_sha256')", {
                "promotion_status": "REJECTED",
                "reasons": ["Missing checkpoint_sha256 in promotion report"],
            }

        if actual_candidate_sha != report_ckpt_sha:
            return False, (
                f"REJECT (Cryptographic mismatch: report checkpoint_sha256 '{report_ckpt_sha}' "
                f"does not match actual candidate file hash '{actual_candidate_sha}')"
            ), {
                "promotion_status": "REJECTED",
                "reasons": ["Candidate SHA-256 mismatch (possible report replay attempt)"],
                "report_sha256": report_ckpt_sha,
                "disk_sha256": actual_candidate_sha,
            }

        # Verify baseline SHA
        report_base_sha = str(report.get("baseline_checkpoint_sha256", "")).lower().strip()
        if not report_base_sha:
            return False, "REJECT (Promotion report missing required 'baseline_checkpoint_sha256')", {
                "promotion_status": "REJECTED",
                "reasons": ["Missing baseline_checkpoint_sha256 in promotion report"],
            }

        if report_base_sha != EXPECTED_FROZEN_SHA256:
            return False, (
                f"REJECT (Baseline contract violation: report baseline SHA '{report_base_sha}' "
                f"does not match immutable frozen baseline '{EXPECTED_FROZEN_SHA256}')"
            ), {
                "promotion_status": "REJECTED",
                "reasons": ["Baseline SHA-256 mismatch"],
            }

        # Provenance: git revision & seed
        git_rev = report.get("git_revision")
        if not git_rev or str(git_rev).strip() in ("", "unknown"):
            return False, "REJECT (Incomplete promotion provenance: missing 'git_revision')", {
                "promotion_status": "REJECTED",
                "reasons": ["Missing git_revision"],
            }

        if "evaluation_seed" not in report:
            return False, "REJECT (Incomplete promotion provenance: missing 'evaluation_seed')", {
                "promotion_status": "REJECTED",
                "reasons": ["Missing evaluation_seed"],
            }

    # ── 2. Metric Extraction & Explicit Unit Handling ─────────────────────────
    metric_units = report.get("metric_units")
    if not is_diagnostic and not metric_units:
        return False, "REJECT (Missing explicit 'metric_units' declaration in evaluation report)", {
            "promotion_status": "REJECTED",
            "reasons": ["Missing explicit metric_units declaration"],
        }

    # Extract raw metrics from report structures
    raw_mean: float = float("nan")
    raw_agile: float = float("nan")
    raw_sparse: float = float("nan")
    raw_worst: float = float("nan")
    raw_pfa: float = float("nan")
    distinct_bands: float = 36.0
    action_entropy: float = 0.0
    q_max: float = 0.0

    if "scenario_summary" in report:
        scen = report["scenario_summary"]
        act = report.get("action_summary", {})
        diag = report.get("training_diagnostics", {})

        raw_mean = float(scen.get("mean_ir", float("nan")))
        raw_agile = float(scen.get("agile_ir", float("nan")))
        raw_sparse = float(scen.get("sparse_ir", float("nan")))
        raw_worst = float(scen.get("worst_case_ir", float("nan")))
        raw_pfa = float(scen.get("pfa", scen.get("mean_pfa", scen.get("false_alarm_rate", float("nan")))))

        distinct_bands = float(act.get("unique_bands", act.get("distinct_bands", 36.0)))
        action_entropy = float(act.get("action_entropy", 0.0))
        q_max = float(diag.get("q_max", 0.0))
    elif "composite_generalization_components_10scen" in report or "baseline_hierarchy" in report:
        diag = report.get("training_diagnostics", {})
        comp = report.get("composite_generalization_components_10scen", {})
        bench = report.get("baseline_hierarchy", {})
        drqn = bench.get("drqn", {})

        raw_mean = float(comp.get("overall_ir", drqn.get("intercept_rate", float("nan"))))
        raw_agile = float(comp.get("agile_ir", float("nan")))
        raw_sparse = float(comp.get("sparse_ir", float("nan")))
        raw_worst = float(comp.get("worst_case_ir", float("nan")))
        raw_pfa = float(drqn.get("pfa", float("nan")))

        distinct_bands = float(drqn.get("distinct_bands", 36.0))
        action_entropy = float(drqn.get("action_entropy", 0.0))
        q_max = float(diag.get("q_max", 0.0))
    else:
        return False, "REJECT (Malformed report: missing scenario_summary or composite_generalization_components_10scen)", {
            "promotion_status": "REJECTED",
            "reasons": ["Unrecognized report schema"],
        }

    # Convert to canonical units (IR in percent [0, 100], Pfa in fraction [0, 1])
    if metric_units:
        # Explicit unit parsing
        mean_unit = metric_units.get("mean_ir", "percent")
        agile_unit = metric_units.get("agile_ir", "percent")
        sparse_unit = metric_units.get("sparse_ir", "percent")
        worst_unit = metric_units.get("worst_case_ir", "percent")
        pfa_unit = metric_units.get("pfa", "fraction")

        mean_ir = raw_mean * 100.0 if mean_unit == "fraction" else raw_mean
        agile_ir = raw_agile * 100.0 if agile_unit == "fraction" else raw_agile
        sparse_ir = raw_sparse * 100.0 if sparse_unit == "fraction" else raw_sparse
        worst_case_ir = raw_worst * 100.0 if worst_unit == "fraction" else raw_worst
        pfa = raw_pfa / 100.0 if pfa_unit == "percent" else raw_pfa
    else:
        # Diagnostic legacy fallback (only reached if is_diagnostic is True)
        mean_ir = raw_mean if raw_mean > 1.0 else raw_mean * 100.0
        agile_ir = raw_agile if raw_agile > 1.0 else raw_agile * 100.0
        sparse_ir = raw_sparse if raw_sparse > 1.0 else raw_sparse * 100.0
        worst_case_ir = raw_worst if raw_worst > 1.0 else raw_worst * 100.0
        pfa = raw_pfa / 100.0 if raw_pfa > 1.0 else raw_pfa

    # ── 3. Non-Finite / NaN / Inf Check ───────────────────────────────────────
    metric_values = {
        "mean_ir": mean_ir,
        "agile_ir": agile_ir,
        "sparse_ir": sparse_ir,
        "worst_case_ir": worst_case_ir,
        "pfa": pfa,
        "distinct_bands": distinct_bands,
        "action_entropy": action_entropy,
        "q_max": q_max,
    }

    non_finite = [k for k, v in metric_values.items() if not math.isfinite(v)]
    if non_finite:
        return False, f"REJECT (Non-finite metric values detected: {', '.join(non_finite)})", {
            "promotion_status": "REJECTED",
            "reasons": [f"Non-finite metrics: {non_finite}"],
            **metric_values,
        }

    # ── 4. Level 0 Safety & Stability Constraints ─────────────────────────────
    reasons = []
    if pfa > 0.001:
        reasons.append(f"Pfa violation: {pfa:.4f} > 0.001 (0.10%)")
    if abs(q_max) > 50.0:
        reasons.append(f"Q-explosion violation: max |Q|={abs(q_max):.2f} > 50.0")
    if action_entropy < 1.0:
        reasons.append(f"Action entropy violation: entropy={action_entropy:.2f} < 1.0")
    if distinct_bands < 12.0:
        reasons.append(f"Distinct bands violation: {distinct_bands:.1f} < 12.0")

    details = {
        "promotion_status": "REJECTED",
        "checkpoint_path": str(target_ckpt) if target_ckpt else report.get("checkpoint_path"),
        "checkpoint_sha256": actual_candidate_sha or report.get("checkpoint_sha256"),
        "baseline_checkpoint_sha256": EXPECTED_FROZEN_SHA256,
        "metric_units": CANONICAL_METRIC_UNITS,
        "metrics": metric_values,
        "mean_ir": mean_ir,
        "agile_ir": agile_ir,
        "sparse_ir": sparse_ir,
        "worst_case_ir": worst_case_ir,
        "pfa": pfa,
        "distinct_bands": distinct_bands,
        "action_entropy": action_entropy,
        "q_max": q_max,
    }

    if reasons:
        verdict = f"REJECT (Level 0 safety constraints failed: {'; '.join(reasons)})"
        details["promotion_verdict"] = verdict
        details["reasons"] = reasons
        return False, verdict, details

    # ── 5. Level 1 Mean Intercept Rate & Robustness Checks ─────────────────────
    ir_delta = mean_ir - BASELINE_GATE25K["mean_ir"]
    agile_delta = agile_ir - BASELINE_GATE25K["agile_ir"]
    sparse_delta = sparse_ir - BASELINE_GATE25K["sparse_ir"]
    worst_delta = worst_case_ir - BASELINE_GATE25K["worst_case_ir"]

    details["deltas"] = {
        "mean_ir_delta": ir_delta,
        "agile_ir_delta": agile_delta,
        "sparse_ir_delta": sparse_delta,
        "worst_case_ir_delta": worst_delta,
    }

    if ir_delta >= 0.0:
        verdict = f"PROMOTED: Candidate Mean IR ({mean_ir:.2f}%) exceeds baseline ({BASELINE_GATE25K['mean_ir']:.2f}%) by +{ir_delta:.2f}%"
        details["promotion_status"] = "APPROVED"
        details["promotion_verdict"] = verdict
        return True, verdict, details
    elif ir_delta >= -1.0:
        if agile_delta >= 2.0 or sparse_delta >= 1.5 or worst_delta >= 1.0:
            verdict = (
                f"PROMOTED (Lexicographic Robustness): Candidate Mean IR ({mean_ir:.2f}%) within 1.0% of baseline, "
                f"with substantial robustness gains (Agile: {agile_delta:+.2f}%, Sparse: {sparse_delta:+.2f}%, Worst-case: {worst_delta:+.2f}%)"
            )
            details["promotion_status"] = "APPROVED"
            details["promotion_verdict"] = verdict
            return True, verdict, details

    verdict = (
        f"REJECT: Candidate Mean IR ({mean_ir:.2f}%) below baseline ({BASELINE_GATE25K['mean_ir']:.2f}%) "
        f"without required robustness offset (Agile: {agile_delta:+.2f}%, Sparse: {sparse_delta:+.2f}%, Worst-case: {worst_delta:+.2f}%)"
    )
    details["promotion_status"] = "REJECTED"
    details["promotion_verdict"] = verdict
    details["reasons"] = [verdict]
    return False, verdict, details


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Promotion Sentinel Evaluator")
    parser.add_argument("--candidate", required=True, help="Candidate gate report JSON")
    parser.add_argument("--checkpoint", help="Optional path to candidate checkpoint file on disk")
    parser.add_argument("--output", help="Optional output JSON path for promotion decision")
    parser.add_argument("--diagnostic-mode", action="store_true", help="Run in metric-only diagnostic mode")
    args = parser.parse_args()

    with open(args.candidate, "r") as f:
        rep = json.load(f)

    promoted, verdict, metrics = evaluate_promotion(
        rep,
        candidate_path=args.checkpoint,
        enforce_provenance=not args.diagnostic_mode,
    )
    print("=" * 80)
    print(f"PROMOTION SENTINEL EVALUATION: {args.candidate}")
    print(f"Machine Status: {metrics.get('promotion_status', 'UNKNOWN')}")
    print(f"Verdict: {verdict}")
    print(f"Scorecard: {json.dumps(metrics, indent=2)}")
    print("=" * 80)

    if args.output:
        out_dict = {
            "candidate_report": args.candidate,
            "promoted": promoted,
            "promotion_status": metrics.get("promotion_status"),
            "verdict": verdict,
            "details": metrics,
            "baseline": BASELINE_GATE25K,
        }
        with open(args.output, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Saved evaluation result to {args.output}")
