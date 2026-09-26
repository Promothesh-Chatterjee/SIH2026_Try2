# Checkpoint Provisioning Specification: Gate-25k Frozen Baseline

## 1. Overview & Immutability Contract
The Gate-25k baseline checkpoint represents the immutable scientific starting point for all Cognitive EW SmartScan continuation experiments.
To prevent repository bloat and ensure strict reproducibility, model weight checkpoints (`*.pt`) are excluded from Git tracking via `.gitignore`.

The checkpoint must exist in the local execution environment prior to running continuation training or readiness preflight checks.

---

## 2. Canonical Identity & Cryptographic Pinning
Any environment attempting to evaluate or retrain from the Gate-25k baseline must possess the exact bit-for-bit canonical artifact:

| Property | Value |
|---|---|
| **Canonical Relative Path** | `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt` |
| **Canonical SHA-256** | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` |
| **Training Step** | 25,000 |
| **Model Architecture** | 360-D Observation, 180 Actions (36 Bands × 5 Modes), Dueling DRQN with Aux Heads |
| **State Contract** | Frozen baseline; immutable reference |

---

## 3. Authoritative Cryptographic Manifests
The repository tracks the authoritative cryptographic manifests in Git:
- `experiments/checkpoints/production_baseline/BASELINE_MANIFEST.json`
- `experiments/checkpoints/production_baseline/SHA256SUMS`
- `experiments/checkpoints/production_baseline/baseline_metadata.json`

The file `SHA256SUMS` records the exact checksums of all baseline artifacts:
```
7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0  checkpoint_gate_25000_frozen.pt
77b7c703ea32e7d8da285b8ce88b31bc305452420ad3a61df28605ed056ff06a  baseline_metadata.json
adf02d70699a3a8225dd67c823cc6679586086e2d86b4f6d4a9e078e2a0b58bd  benchmark_v2_baseline_gate25k.json
a44e70df97a52ddaa22be23eb81a293be1b3fc120e2595ca5b739c1d0cae81b1  benchmark_v2_multiseed_summary.json
edcef07b020563aefeac99fa3b03c2c6a474f07afdac7660e61e336523b8fe0c  baseline_reservoir_5k.pkl
```

---

## 4. Local Provisioning Procedure
If provisioning the checkpoint into a new execution environment:
1. Obtain the artifact `checkpoint_gate_25000_frozen.pt` from secure artifact storage.
2. Place the file at:
   `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`
3. Verify its SHA-256 before running any code:
   ```bash
   python -c "import hashlib; h=hashlib.sha256(open('experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt', 'rb').read()).hexdigest(); assert h == '7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0', f'Mismatch: {h}'; print('Checkpoint verified bit-exact.')"
   ```

---

## 5. Fail-Closed Verification Behavior
Both `scripts/preflight_tsrd.py` and `scripts/check_retraining_readiness.py` strictly enforce checkpoint presence and checksum validity.
If the checkpoint is missing, corrupted, or altered by even a single bit, the preflight and readiness gates fail closed with exit code 1:
```
[FAIL] Gate-25k frozen checkpoint missing or SHA-256 mismatch!
Expected: 7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0
Retraining remains strictly blocked.
```
No silent fallback or fresh weight initialization is permitted during continuation runs.
