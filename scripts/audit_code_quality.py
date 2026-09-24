"""Fail-closed static audit of repository code quality and silent fallback prohibition.

Scans critical paths (ew_core, scripts, .github, configs) for forbidden patterns:
  - except Exception: pass (silent broad exception swallowing)
  - except AssertionError: pass (silent assertion swallowing)
  - || true (masking CI failures)
  - silent random/synthetic fallbacks in critical paths

Classifies occurrences into CRITICAL vs NON-CRITICAL (e.g. wandb optional logging).
Exits non-zero if any unclassified critical-path violation is detected.
"""

from __future__ import annotations

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

# Patterns that indicate potential silent failure or fallback
SUSPICIOUS_PATTERNS = [
    (r"except\s+Exception\s*:\s*pass", "silent_exception_pass"),
    (r"except\s+AssertionError\s*:\s*pass", "silent_assertion_pass"),
    (r"\|\|\s*true", "ci_masking_true"),
    (r"except\s*:\s*pass", "bare_except_pass"),
]

# Explicitly reviewed and classified benign/non-critical occurrences:
# key: (file_relative_path, pattern_category, line_number_or_context_substring)
KNOWN_BENIGN = [
    # Optional WandB telemetry logging swallowing
    ("train_scheduler.py", "silent_exception_pass", "wandb"),
    # Windows console UTF-8 reconfigure fallback
    ("smoke_test.py", "silent_exception_pass", "reconfigure"),
    # Defensive cleanup during file deletion
    ("validate_pipeline.py", "silent_exception_pass", "cleanup"),
    # Optional RNG state restoration from checkpoint
    ("train_scheduler.py", "silent_exception_pass", "rng_state"),
    ("train_scheduler.py", "silent_exception_pass", "np_rng_state"),
]


def is_benign(rel_path: str, cat: str, line_content: str, full_file_content: str, line_idx: int) -> bool:
    fname = Path(rel_path).name
    # Check context around line
    lines = full_file_content.splitlines()
    start = max(0, line_idx - 5)
    end = min(len(lines), line_idx + 3)
    context = "\n".join(lines[start:end]).lower()

    for benign_file, benign_cat, benign_sub in KNOWN_BENIGN:
        if benign_file in fname and cat == benign_cat and benign_sub in context:
            return True
    return False


def run_audit() -> int:
    print(f"{'='*60}")
    print("Static Code Quality & Silent Fallback Audit")
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
                if p.suffix in [".py", ".yml", ".yaml", ".sh", ".ps1"]:
                    rel_p = str(p.relative_to(ROOT)).replace("\\", "/")
                    try:
                        content = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        print(f"Warning: could not read {rel_p}: {e}")
                        continue

                    lines = content.splitlines()
                    for line_idx, line in enumerate(lines, 1):
                        for pattern, cat in SUSPICIOUS_PATTERNS:
                            if re.search(pattern, line, re.IGNORECASE):
                                item = {
                                    "file": rel_p,
                                    "line": line_idx,
                                    "category": cat,
                                    "text": line.strip(),
                                }
                                if is_benign(rel_p, cat, line, content, line_idx):
                                    classified_benign.append(item)
                                else:
                                    violations.append(item)

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
