"""Canonical Benchmark Contract for Cognitive EW SmartScan.

Defines BENCHMARK_VERSION = "2026.1-CANONICAL" and binds all 14 identity fields:
  1. BENCHMARK_VERSION
  2. GIT_COMMIT
  3. CHECKPOINT_SHA256 (frozen Gate-25k reference candidate)
  4. NORMALIZATION_SHA256 (canonical preprocessing statistics)
  5. MODEL_CONFIG_SHA256 (configs/model_config.yaml)
  6. TRAINING_CONFIG_SHA256 (configs/training_config.yaml)
  7. CANONICAL_SCENARIO_LIST (10 fixed held-out validation scenarios)
  8. SCENARIO_FILE_HASHES (per-file SHA256 of the 10 .h5 files)
  9. METRIC_ENGINE_VERSION (canonical FiguresOfMerit evaluation engine)
  10. SEED_POLICY (deterministic demo seed 42, robustness seeds 123, 999)
  11. EVALUATION_HORIZON (1000 dwells per scenario)
  12. ACTION_SPACE (180 joint actions: 36 bands x 5 dwell modes)
  13. OBSERVATION_DIM (360 continuous features: 36 bands x 10 features)
  14. BELIEF_CONFIGURATION (ema_alpha=0.30, ema_alpha_miss_confirmed=0.20)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = REPO_ROOT / "cognitive_ew_smart_scan"

BENCHMARK_VERSION = "2026.1-CANONICAL"
METRIC_ENGINE_VERSION = "v2.0-audited-confusion-matrix"

CANONICAL_SCENARIOS = [
    "config_117",
    "config_119",
    "config_143",
    "config_194",
    "config_195",
    "config_241",
    "config_29",
    "config_42",
    "config_64",
    "config_96",
]

SEED_POLICY = {
    "official_demo_seed": 42,
    "robustness_seeds": [123, 999],
}

ACTION_SPACE_CONTRACT = {
    "n_bands": 36,
    "n_modes": 5,
    "n_actions": 180,
    "action_selection_mode": "flat_argmax",
}

OBSERVATION_CONTRACT = {
    "n_bands": 36,
    "band_features": 10,
    "obs_dim": 360,
}

BELIEF_CONTRACT = {
    "ema_alpha": 0.30,
    "ema_alpha_miss_confirmed": 0.20,
    "staleness_weight": 0.35,
    "occupancy_weight": 0.25,
    "uncertainty_weight": 0.20,
    "periodic_weight": 0.10,
    "semantic_weight": 0.10,
}


def _file_sha256(path: Path) -> str:
    """Compute hex SHA256 of file contents."""
    if not path.exists():
        return f"MISSING:{path}"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_commit() -> str:
    """Resolve current git HEAD revision."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "UNKNOWN_REVISION"


def get_benchmark_contract(data_root: Path | str | None = None) -> Dict[str, Any]:
    """Assemble complete 14-point benchmark contract payload."""
    frozen_ckpt_path = (
        PKG_ROOT / "checkpoints" / "scheduler_v2_operational_candidate" / "checkpoint_gate_25000_frozen.pt"
    )
    norm_stats_path = PKG_ROOT / "checkpoints" / "deinterleaver" / "normalization_stats.json"
    model_cfg_path = PKG_ROOT / "configs" / "model_config.yaml"
    train_cfg_path = PKG_ROOT / "configs" / "training_config.yaml"

    # Resolve scenario files from TSRD root
    if data_root is None:
        data_root = Path("D:/TSRD")
    else:
        data_root = Path(data_root)

    scenario_hashes: Dict[str, str] = {}
    stare_val_dir = data_root / "stare" / "val_stare"
    for scen_id in CANONICAL_SCENARIOS:
        h5_path = stare_val_dir / f"{scen_id}.h5"
        scenario_hashes[scen_id] = _file_sha256(h5_path)

    contract: Dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "git_commit": _git_commit(),
        "checkpoint_sha256": _file_sha256(frozen_ckpt_path),
        "checkpoint_path": str(frozen_ckpt_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "normalization_sha256": _file_sha256(norm_stats_path),
        "normalization_path": str(norm_stats_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_config_sha256": _file_sha256(model_cfg_path),
        "training_config_sha256": _file_sha256(train_cfg_path),
        "canonical_scenarios": CANONICAL_SCENARIOS,
        "scenario_file_hashes": scenario_hashes,
        "metric_engine_version": METRIC_ENGINE_VERSION,
        "seed_policy": SEED_POLICY,
        "evaluation_horizon": 1000,
        "action_space": ACTION_SPACE_CONTRACT,
        "observation_dim": OBSERVATION_CONTRACT,
        "belief_configuration": BELIEF_CONTRACT,
    }
    return contract


def generate_benchmark_markdown(contract: Dict[str, Any]) -> str:
    """Format benchmark contract as human-readable GitHub Flavored Markdown."""
    lines = [
        f"# Canonical Benchmark Contract — Version {contract['benchmark_version']}",
        "",
        "> **Notice**: This document defines the single authoritative benchmark specification for the",
        "> Cognitive EW SmartScan project. All evaluation claims, scorecards, and checkpoint promotion gates",
        "> must conform to these exact parameters and reference hashes.",
        "",
        "## 1. Provenance & Artifact Identity",
        "",
        f"- **Benchmark Version**: `{contract['benchmark_version']}`",
        f"- **Git Commit**: `{contract['git_commit']}`",
        f"- **Reference Candidate Checkpoint**: `{contract['checkpoint_path']}`",
        f"  - SHA-256: `{contract['checkpoint_sha256']}`",
        f"- **Normalization Statistics**: `{contract['normalization_path']}`",
        f"  - SHA-256: `{contract['normalization_sha256']}`",
        f"- **Model Config**: `configs/model_config.yaml` (SHA-256: `{contract['model_config_sha256']}`)",
        f"- **Training Config**: `configs/training_config.yaml` (SHA-256: `{contract['training_config_sha256']}`)",
        f"- **Metric Engine Version**: `{contract['metric_engine_version']}`",
        "",
        "## 2. Action & Observation Space Contracts",
        "",
        f"- **Frequency Spectrum**: 36 Bands (0 to 18,000 MHz, 500 MHz IBW per band)",
        f"- **Dwell Modes**: 5 Modes (0: SHORT 125µs, 1: NORMAL 500µs, 2: LONG 1250µs, 3: REVISIT 500µs, 4: PREEMPTIVE 500-1500µs)",
        f"- **Joint Action Space**: `{contract['action_space']['n_actions']}` discrete actions (`band * 5 + mode`)",
        f"- **Action Selection Policy**: `{contract['action_space']['action_selection_mode']}`",
        f"- **Observation Dimension**: `{contract['observation_dim']['obs_dim']}` continuous floats (`[0.0, 1.0]`)",
        f"  - 10 features per band: occupancy, det_rate, miss_rate, uncertainty, revisit_age, emitter_count, deint_conf, pri_stability, agility, priority",
        "",
        "## 3. Evaluation Horizon & Seed Policy",
        "",
        f"- **Dwells per Scenario**: `{contract['evaluation_horizon']}` steps",
        f"- **Official Demonstration Seed**: `{contract['seed_policy']['official_demo_seed']}`",
        f"- **Robustness Verification Seeds**: `{contract['seed_policy']['robustness_seeds']}`",
        "",
        "## 4. Canonical 10 Validation Scenarios",
        "",
        "| Scenario ID | Category | Dataset Path | File SHA-256 |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for scen in contract["canonical_scenarios"]:
        sh = contract["scenario_file_hashes"].get(scen, "UNKNOWN")
        cat = "Agile Hopper" if scen in ["config_29", "config_195", "config_241", "config_119"] else "Stationary/Sparse"
        lines.append(f"| `{scen}` | {cat} | `stare/val_stare/{scen}.h5` | `{sh[:16]}...` |")

    lines.extend([
        "",
        "## 5. Belief State Smoothing Parameters",
        "",
        f"- EMA Detection Smoothing: `alpha = {contract['belief_configuration']['ema_alpha']}`",
        f"- Confirmed Miss Decay: `alpha_miss_confirmed = {contract['belief_configuration']['ema_alpha_miss_confirmed']}`",
        "",
    ])
    return "\n".join(lines)


def emit_contract_files(data_root: Path | str | None = None) -> tuple[Path, Path]:
    """Generate both JSON and Markdown contract artifacts."""
    contract = get_benchmark_contract(data_root)

    json_path = PKG_ROOT / "checkpoints" / "benchmark_contract.json"
    md_path = REPO_ROOT / "docs" / "BENCHMARK_V2.md"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    md_path.write_text(generate_benchmark_markdown(contract), encoding="utf-8")

    return json_path, md_path


if __name__ == "__main__":
    j_p, m_p = emit_contract_files()
    print(f"[+] Successfully emitted benchmark contract:")
    print(f"    JSON: {j_p}")
    print(f"    MD:   {m_p}")
