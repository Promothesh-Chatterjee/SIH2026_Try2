"""Generate controlled synthetic test fixtures for Phase 10 taxonomy coverage.

CRITICAL CONTRACT:
These fixtures are strictly test fixtures written to tests/fixtures/phase10_tsrd/.
They are explicitly tagged with provenance source_type="synthetic_fixture" and
MUST NEVER be written to the canonical data/ root or passed off as official TSRD.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np


def _create_h5_scenario(
    output_path: Path,
    pulses: np.ndarray,
    labels: np.ndarray,
    scenario_class: str,
    metadata_extra: dict | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort by ToA to ensure strict monotonicity
    sort_idx = np.argsort(pulses[:, 0], kind="stable")
    sorted_pulses = pulses[sort_idx].astype(np.float32)
    sorted_labels = labels[sort_idx].astype(np.int64).reshape(-1, 1)

    with h5py.File(str(output_path), "w") as handle:
        handle.create_dataset("data", data=sorted_pulses, compression="gzip", compression_opts=4)
        handle.create_dataset("labels", data=sorted_labels, compression="gzip", compression_opts=4)
        meta = handle.create_group("metadata")
        meta.attrs["source_type"] = "synthetic_fixture"
        meta.attrs["target_taxonomy_class"] = scenario_class
        meta.attrs["num_pulses"] = len(sorted_pulses)
        meta.attrs["num_emitters"] = len(np.unique(sorted_labels[sorted_labels != -1]))
        if metadata_extra:
            for k, v in metadata_extra.items():
                meta.attrs[k] = str(v)


def generate_fixed_scenario(seed: int, n_emitters: int = 2, n_pulses_per: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_pulses = []
    all_labels = []
    for e in range(n_emitters):
        base_cf = 3000.0 + e * 500.0
        pri = 500.0 + e * 100.0  # us
        toas = np.cumsum(rng.uniform(pri * 0.98, pri * 1.02, size=n_pulses_per))
        cfs = base_cf + rng.uniform(-2.0, 2.0, size=n_pulses_per)  # narrow span <= 4 MHz
        pws = rng.uniform(1.0, 2.0, size=n_pulses_per)
        aoas = np.full(n_pulses_per, 45.0 + e * 30.0)
        amps = rng.uniform(-40.0, -30.0, size=n_pulses_per)
        mat = np.column_stack([toas, cfs, pws, aoas, amps])
        all_pulses.append(mat)
        all_labels.append(np.full(n_pulses_per, e, dtype=np.int64))

    return np.vstack(all_pulses), np.concatenate(all_labels)


def generate_sparse_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pulses = 30
    # High jitter between 3,000 and 15,000 us prevents periodic_scan tag while maintaining density <= 0.2
    pris = rng.uniform(3000.0, 15000.0, size=n_pulses)
    toas = np.cumsum(pris)
    cfs = np.full(n_pulses, 5000.0) + rng.uniform(-1.0, 1.0, size=n_pulses)
    pws = rng.uniform(2.0, 3.0, size=n_pulses)
    aoas = np.full(n_pulses, 120.0)
    amps = rng.uniform(-50.0, -45.0, size=n_pulses)
    mat = np.column_stack([toas, cfs, pws, aoas, amps])
    labels = np.zeros(n_pulses, dtype=np.int64)
    return mat, labels


def generate_fast_agile_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pulses = 400
    pri = 400.0  # fast PRI <= 1000 us
    toas = np.cumsum(rng.uniform(pri * 0.95, pri * 1.05, size=n_pulses))
    # Wide agile frequency hopping >= 60 MHz span
    hop_cfs = np.array([4000.0, 4050.0, 4100.0, 4150.0, 4200.0])
    cfs = rng.choice(hop_cfs, size=n_pulses) + rng.uniform(-1.0, 1.0, size=n_pulses)
    pws = rng.uniform(1.0, 1.5, size=n_pulses)
    aoas = np.full(n_pulses, 90.0)
    amps = rng.uniform(-35.0, -25.0, size=n_pulses)
    mat = np.column_stack([toas, cfs, pws, aoas, amps])
    labels = np.zeros(n_pulses, dtype=np.int64)
    return mat, labels


def generate_slow_agile_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_pulses = 150
    pri = 3500.0  # slow PRI > 1000 us
    toas = np.cumsum(rng.uniform(pri * 0.95, pri * 1.05, size=n_pulses))
    hop_cfs = np.array([6000.0, 6080.0, 6150.0, 6220.0])
    cfs = rng.choice(hop_cfs, size=n_pulses) + rng.uniform(-1.0, 1.0, size=n_pulses)
    pws = rng.uniform(3.0, 5.0, size=n_pulses)
    aoas = np.full(n_pulses, 200.0)
    amps = rng.uniform(-40.0, -30.0, size=n_pulses)
    mat = np.column_stack([toas, cfs, pws, aoas, amps])
    labels = np.zeros(n_pulses, dtype=np.int64)
    return mat, labels


def generate_markov_agile_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_pulses = []
    all_labels = []
    # 2 distinct agile emitters hopping across discrete Markov states
    for e in range(2):
        n_pulses = 200
        pri = 1200.0 + e * 400.0
        toas = np.cumsum(rng.uniform(pri * 0.98, pri * 1.02, size=n_pulses))
        states = [7000.0 + e * 500.0, 7060.0 + e * 500.0, 7120.0 + e * 500.0, 7180.0 + e * 500.0]
        cfs = np.zeros(n_pulses)
        current = 0
        for i in range(n_pulses):
            cfs[i] = states[current] + rng.uniform(-0.5, 0.5)
            # Markov transition
            current = (current + rng.choice([0, 1, 2])) % len(states)
        pws = rng.uniform(2.0, 3.0, size=n_pulses)
        aoas = np.full(n_pulses, 60.0 + e * 80.0)
        amps = rng.uniform(-35.0, -25.0, size=n_pulses)
        mat = np.column_stack([toas, cfs, pws, aoas, amps])
        all_pulses.append(mat)
        all_labels.append(np.full(n_pulses, e, dtype=np.int64))

    return np.vstack(all_pulses), np.concatenate(all_labels)


def generate_periodic_scan_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Burst pulses followed by periodic gaps of 25,000 us (25 ms)
    n_bursts = 8
    burst_len = 15
    toas_list = []
    current_time = 1000.0
    scan_gap = 25000.0  # constant periodic scan illumination gap

    for _ in range(n_bursts):
        burst_toas = current_time + np.cumsum(rng.uniform(150.0, 200.0, size=burst_len))
        toas_list.append(burst_toas)
        current_time = burst_toas[-1] + scan_gap

    toas = np.concatenate(toas_list)
    n_pulses = len(toas)
    cfs = np.full(n_pulses, 9500.0) + rng.uniform(-1.0, 1.0, size=n_pulses)
    pws = rng.uniform(1.0, 1.5, size=n_pulses)
    aoas = np.full(n_pulses, 310.0)
    amps = rng.uniform(-40.0, -30.0, size=n_pulses)
    mat = np.column_stack([toas, cfs, pws, aoas, amps])
    labels = np.zeros(n_pulses, dtype=np.int64)
    return mat, labels


def generate_mixed_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Mix one fixed emitter and one agile emitter
    pulses_f, labels_f = generate_fixed_scenario(seed, n_emitters=1, n_pulses_per=150)
    pulses_a, labels_a = generate_fast_agile_scenario(seed + 100)
    labels_a = labels_a + 1  # emitter ID 1
    return np.vstack([pulses_f, pulses_a]), np.concatenate([labels_f, labels_a])


def generate_dense_scenario(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_emitters = 22  # >= 20 emitters satisfies dense
    all_pulses = []
    all_labels = []
    for e in range(n_emitters):
        n_p = 50
        pri = 200.0 + rng.uniform(0.0, 100.0)
        toas = np.cumsum(rng.uniform(pri * 0.95, pri * 1.05, size=n_p))
        base_cf = 2000.0 + e * 400.0
        cfs = base_cf + rng.uniform(-1.0, 1.0, size=n_p)  # fixed emitter (span <= 2 MHz)
        pws = rng.uniform(0.5, 2.5, size=n_p)
        aoas = np.full(n_p, (e * 360.0 / n_emitters) % 360.0)
        amps = rng.uniform(-60.0, -10.0, size=n_p)
        mat = np.column_stack([toas, cfs, pws, aoas, amps])
        all_pulses.append(mat)
        all_labels.append(np.full(n_p, e, dtype=np.int64))

    return np.vstack(all_pulses), np.concatenate(all_labels)


def generate_all_fixtures(output_dir: Path | str, scenarios_per_class: int = 3) -> dict[str, list[Path]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generators = {
        "fixed": generate_fixed_scenario,
        "sparse": generate_sparse_scenario,
        "fast_agile": generate_fast_agile_scenario,
        "slow_agile": generate_slow_agile_scenario,
        "markov_agile": generate_markov_agile_scenario,
        "periodic_scan": generate_periodic_scan_scenario,
        "mixed": generate_mixed_scenario,
        "dense": generate_dense_scenario,
    }

    generated_files: dict[str, list[Path]] = {}

    for cls_name, gen_func in generators.items():
        cls_dir = out / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        generated_files[cls_name] = []
        for i in range(scenarios_per_class):
            seed = 1000 + i * 73 + len(cls_name) * 11
            pulses, labels = gen_func(seed)
            target_file = cls_dir / f"scenario_{i + 1}.h5"
            _create_h5_scenario(
                target_file,
                pulses,
                labels,
                scenario_class=cls_name,
                metadata_extra={"generator_seed": seed, "index": i + 1},
            )
            generated_files[cls_name].append(target_file)

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate controlled synthetic Phase 10 test fixtures")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/fixtures/phase10_tsrd",
        help="Destination directory (must be under tests/fixtures/, never data/)",
    )
    parser.add_argument(
        "--count-per-class",
        type=int,
        default=3,
        help="Number of scenarios to generate per class (minimum 3 required for Gate 10.4B)",
    )
    args = parser.parse_args()

    target = Path(args.output_dir)
    if "tests" not in target.parts:
        raise ValueError(f"Prohibited output directory: {target}. Fixtures MUST reside under tests/fixtures/!")

    results = generate_all_fixtures(target, scenarios_per_class=args.count_per_class)
    total_files = sum(len(v) for v in results.values())
    print(f"Successfully generated {total_files} controlled test fixtures across {len(results)} classes in {target}")
