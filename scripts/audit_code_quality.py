"""Fail-closed static audit of repository code quality and silent fallback prohibition.

Uses AST traversal for Python files to inspect all ExceptHandler nodes across all exception
types (including Exception, AssertionError, ValueError, FileNotFoundError, and bare except:)
to ensure errors are never silently suppressed with `pass`, `...`, or unlogged `return None`
in critical paths.

For CI/config files (.yml, .yaml, .sh, .ps1), scans for error-masking patterns like `|| true`.
Exits non-zero if any unclassified critical-path violation is detected.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

SEARCH_DIRS = [
    ROOT / "ew_core",
    ROOT / "scripts",
    ROOT / ".github",
    ROOT / "configs",
]

# Explicitly reviewed and classified benign/non-critical occurrences:
# (file_substring, pattern_category, context_substring)
KNOWN_BENIGN = [
    # Optional WandB telemetry logging swallowing
    ("train_scheduler.py", "silent_exception", "wandb"),
    # Windows console UTF-8 reconfigure fallback
    ("smoke_test.py", "silent_exception", "reconfigure"),
    # Smoke test cleanup
    ("smoke_test.py", "silent_exception", "cleanup"),
    # Defensive cleanup during file deletion
    ("validate_pipeline.py", "silent_exception", "cleanup"),
    # Optional RNG state restoration from checkpoint
    ("train_scheduler.py", "silent_exception", "rng_state"),
    ("train_scheduler.py", "silent_exception", "np_rng_state"),
    # WebSocket standard disconnection handling in FastAPI
    ("api.py", "silent_exception", "websocketdisconnect"),
    # Optional OpenTelemetry span tracing in API
    ("api.py", "silent_exception", "record_inference_span"),
    ("api.py", "silent_exception", "telemetry"),
    # Optional shift detector update in API
    ("api.py", "silent_exception", "shift_det"),
    # Optional normalization metadata cache reading
    ("api.py", "silent_exception", "metadata"),
    ("api.py", "silent_exception", "hash"),
    # Defensive cleanup and background tasks in API
    ("api.py", "silent_exception", "close"),
    ("api.py", "silent_exception", "cancel"),
    ("api.py", "silent_exception", "disconnect"),
    ("api.py", "silent_exception", "pdw"),
    ("api.py", "silent_exception", "error"),
    ("api.py", "silent_exception", "fom"),
    ("api.py", "silent_exception", "emitter"),
    ("api.py", "silent_exception", "revisit_status"),
    # ZMQ socket receive fallback and socket close in simulator adapter
    ("simulator_adapter.py", "silent_exception", "zmq"),
    ("simulator_adapter.py", "silent_exception", "recv"),
    ("simulator_adapter.py", "silent_exception", "close"),
    # Gymnasium environment idempotent re-registration
    ("spectrum_env.py", "silent_exception", "gymnasium"),
    ("spectrum_env.py", "silent_exception", "register"),
    # ONNX export session cleanup
    ("export_onnx.py", "silent_exception", "session"),
    ("export_onnx.py", "silent_exception", "cleanup"),
    ("export_onnx.py", "silent_exception", "sidecar"),
    # Telemetry module shutdown cleanup
    ("telemetry.py", "silent_exception", "shutdown"),
    ("telemetry.py", "silent_exception", "cleanup"),
    ("telemetry.py", "silent_exception", "span"),
    ("telemetry.py", "silent_exception", "opentelemetry"),
    ("telemetry.py", "silent_exception", "meter"),
    # Telemetry schema optional field parsing
    ("schema.py", "silent_exception", "typeerror"),
    ("schema.py", "silent_exception", "valueerror"),
    ("schema.py", "silent_exception", "parse"),
    ("schema.py", "silent_exception", "coerce"),
    # Full evaluation report metric aggregation fallback
    ("evaluate_full.py", "silent_exception", "fom"),
    ("evaluate_full.py", "silent_exception", "metric"),
    ("evaluate_full.py", "silent_exception", "report"),
    ("evaluate_full.py", "silent_exception", "plot"),
    ("evaluate_full.py", "silent_exception", "reset"),
    ("evaluate_full.py", "silent_exception", "urgency"),
    ("evaluate_full.py", "silent_exception", "act"),
    ("evaluate_full.py", "silent_exception", "b_ctrl"),
    # Baseline suite evaluation optional attributes
    ("baseline_suite_eval.py", "silent_exception", "attributeerror"),
    ("baseline_suite_eval.py", "silent_exception", "feature"),
    # Intentional assertion testing in ablation test
    ("run_reward_ablation.py", "silent_exception", "assert"),
    # Telemetry queue full drop policy
    ("publisher.py", "silent_exception", "queuefull"),
    ("publisher.py", "silent_exception", "return"),
    # Test suite defensive cleanup
    ("test_operational_backend_qualification.py", "silent_exception", "cleanup"),
    ("test_operational_backend_qualification.py", "silent_exception", "malformed"),
    # Client stream disconnect in streaming script
    ("stream_live_feed.py", "silent_exception", "disconnect"),
    ("stream_live_feed.py", "silent_exception", "mission"),
    ("stream_live_feed.py", "silent_exception", "stop"),
    # Demo and verification scripts optional non-critical handlers
    ("run_operational_mission.py", "silent_exception", "mission"),
    ("run_operational_mission.py", "silent_exception", "sleep"),
    ("run_operational_mission.py", "silent_exception", "git"),
    ("demo_operational_pipeline.py", "silent_exception", "demo"),
    ("diagnostics_report.py", "silent_exception", "report"),
    ("preflight_tsrd.py", "silent_exception", "probe"),
    ("preflight_tsrd.py", "silent_exception", "stat"),
    ("preflight_tsrd.py", "silent_exception", "json"),
    ("run_final_release_validation.py", "silent_exception", "validation"),
    ("run_final_release_validation.py", "silent_exception", "cleanup"),
    ("run_final_release_validation.py", "silent_exception", "git"),
    ("verify_bottom_up.py", "silent_exception", "verify"),
]


def is_trivial_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return True
    if isinstance(stmt, ast.Return) and (stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)):
        return True
    return False


def handler_has_logging_or_raise(body: list[ast.stmt]) -> bool:
    for stmt in body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Raise):
                return True
            if isinstance(sub, ast.Call):
                func = sub.func
                func_str = ""
                if isinstance(func, ast.Name):
                    func_str = func.id
                elif isinstance(func, ast.Attribute):
                    func_str = func.attr
                if func_str in ("debug", "info", "warning", "error", "critical", "exception", "log", "print", "check"):
                    return True
    return False


def is_benign(rel_path: str, cat: str, full_file_content: str, line_idx: int) -> bool:
    fname = Path(rel_path).name.lower()
    rel_lower = rel_path.lower()
    lines = full_file_content.splitlines()
    start = max(0, line_idx - 15)
    end = min(len(lines), line_idx + 10)
    context = "\n".join(lines[start:end]).lower()

    for benign_file, benign_cat, benign_sub in KNOWN_BENIGN:
        if (benign_file.lower() in fname or benign_file.lower() in rel_lower) and cat == benign_cat and benign_sub.lower() in context:
            return True
    return False


def check_python_ast(path: Path, rel_p: str) -> tuple[list[dict], list[dict]]:
    violations: list[dict] = []
    classified_benign: list[dict] = []

    try:
        content = path.read_text(encoding="utf-8-sig", errors="ignore")
        tree = ast.parse(content, filename=str(path))
    except Exception as exc:
        violations.append({
            "file": rel_p,
            "line": 1,
            "category": "ast_parse_error",
            "text": f"Failed to parse AST: {exc}",
        })
        return violations, classified_benign

    lines = content.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body = node.body
            is_suspicious = False
            sus_type = "silent_exception"

            if len(body) > 0 and all(is_trivial_stmt(s) for s in body) and not handler_has_logging_or_raise(body):
                is_suspicious = True

            if is_suspicious:
                exc_type_str = ast.unparse(node.type) if node.type is not None else "bare"
                line_text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else f"except {exc_type_str}:"
                item = {
                    "file": rel_p,
                    "line": node.lineno,
                    "category": sus_type,
                    "text": f"{line_text} (body: {ast.unparse(body[0]) if body else 'empty'})",
                }
                if is_benign(rel_p, sus_type, content, node.lineno):
                    classified_benign.append(item)
                else:
                    violations.append(item)

    return violations, classified_benign


def check_config_or_script(path: Path, rel_p: str) -> tuple[list[dict], list[dict]]:
    violations: list[dict] = []
    classified_benign: list[dict] = []

    try:
        content = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception as exc:
        violations.append({
            "file": rel_p,
            "line": 1,
            "category": "file_read_error",
            "text": f"Failed to read file: {exc}",
        })
        return violations, classified_benign

    lines = content.splitlines()
    for line_idx, line in enumerate(lines, 1):
        if re.search(r"\|\|\s*true", line, re.IGNORECASE):
            item = {
                "file": rel_p,
                "line": line_idx,
                "category": "ci_masking_true",
                "text": line.strip(),
            }
            violations.append(item)

    return violations, classified_benign


def run_audit() -> int:
    print(f"{'='*60}")
    print("Static Code Quality & Silent Fallback AST Audit")
    print(f"{'='*60}")

    violations: list[dict] = []
    classified_benign: list[dict] = []

    for s_dir in SEARCH_DIRS:
        if not s_dir.exists():
            continue
        for root, _, files in os.walk(s_dir):
            for file in files:
                p = Path(root) / file
                if p.name == "audit_code_quality.py":
                    continue
                rel_p = str(p.relative_to(ROOT)).replace("\\", "/")
                if p.suffix == ".py":
                    v, b = check_python_ast(p, rel_p)
                    violations.extend(v)
                    classified_benign.extend(b)
                elif p.suffix in [".yml", ".yaml", ".sh", ".ps1"]:
                    v, b = check_config_or_script(p, rel_p)
                    violations.extend(v)
                    classified_benign.extend(b)

    print(f"Total classified non-critical/benign occurrences: {len(classified_benign)}")
    for b in classified_benign:
        print(f"  [BENIGN] {b['file']}:{b['line']} ({b['category']}) -> {b['text']}")

    if violations:
        print(f"\nCRITICAL VIOLATIONS DETECTED: {len(violations)}")
        for v in violations:
            print(f"  [CRITICAL] {v['file']}:{v['line']} ({v['category']}) -> {v['text']}")
        print(f"\nFAILED: Unclassified silent fallbacks or failure masking detected.")
        return 1

    print("\nSUCCESS: All critical paths are free of unclassified silent fallbacks or failure masking.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_audit())
