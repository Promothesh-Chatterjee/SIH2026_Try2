# Cognitive Electronic Warfare Smart Scan Scheduler

**SIH 2026 — DRDO Problem SIH26056** — ML-based ES receiver that deinterleaves interleaved radar PDWs and autonomously schedules narrow-band scanning without prior threat libraries. Sub-millisecond decisions on **NVIDIA Jetson AGX Orin**.

Dataset: [Turing Synthetic Radar Dataset (TSRD)](https://github.com/alan-turing-institute/turing-deinterleaving-challenge) · [HuggingFace](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)

## Architecture

```
PDW Stream (ToA,CF,PW,AoA,Amp)  5D
        │
        ▼
  normalise.py:10  → 6D (AoA→sin/cos, robust IQR, log1p)
        │
        ▼
 PDWTransformerEncoder (d_model=128, 4×8-head, ToA pos enc, L2) → (N,64)
        │
        ▼ HDBSCAN (min10, eom, euclidean)
  emitter labels (-1=noise) ──► SemanticMemory (SQLite) ──► get_band_priority_boost()
        │                                                    │
        ▼                                                    ▼
  PeriodicScanInterceptor (find_peaks PRI) ──► preemptive schedule
        │
        ▼
  SmartScanMoE ──┬─ EagerAgent (DRQN Dueling LSTM 256×2, Q=V+A-meanA)
                 └─ RevisitAgent (exp(decay·Δt), max_gap)
                        │ fused = 0.6·eager_norm + 0.4·revisit_norm
                        ▼ top-K bands
                 RFScanEnv (Gymnasium, Box 2·180, dwell 5×100µs)
                        │
                        ▼ FastAPI / ONNX (DRQN + Transformer)
```

Dueling: `Q(s,a)=V(s)+A(s,a)-mean(A)` · Thompson Beta(1,1) warmup 5000 steps · BPTT seq_len16

## Quick Start (safe default)

```bash
pip install -r requirements.txt
python scripts/download_data.py  # safe default: does not pull the huge TSRD archive
python -m src.training.train_deinterleaver --model-config configs/model_config.yaml --config configs/training_config.yaml
```

> The official TSRD download is intentionally opt-in. The project stays in a safe local-training mode unless you explicitly run with `--allow-download` after confirming the dataset size and your teammate’s repo path.

For a real TSRD run when the official repo is ready:

```bash
python scripts/download_data.py --allow-download --output-dir data --token "$HF_TOKEN"
python -m src.evaluation.evaluate_full --config configs/model_config.yaml --test-dir data/test --output-dir results --mode scan
```

## Detailed Setup

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # torch CUDA 12.1 extra-index included
cp .env.example .env  # set HF_TOKEN, DEVICE=cuda
# Verify imports
python -c "from src.preprocessing.normalise import normalise_pdws; print('OK')"
python -c "from src.environment.rf_scan_env import RFScanEnv; print('OK')"
python -c "from src.models.deinterleaver import PDWTransformerEncoder; print('OK')"
python -c "from src.models.drqn_scheduler import DRQNScheduler; print('OK')"
python -c "from src.models.smartscan_moe import SmartScanMoE; print('OK')"
```

Hardware: **Dev** — 16GB RAM, CUDA 12.1, 50GB disk for TSRD (stare 3.86B pulses, scan 282M). **Edge** — Jetson AGX Orin 32GB, JetPack 5+, ONNX Runtime GPU.

## Training

```bash
# Deinterleaver (Transformer Triplet, file-local batches, HDBSCAN V-measure, early stopping)
bash scripts/train_deinterleaver.sh
# or: python -m src.training.train_deinterleaver --model-config configs/model_config.yaml --config configs/training_config.yaml

# Scheduler (DRQN Dueling + Thompson warmup + MoE eval every 5000 steps)
bash scripts/train_scheduler.sh

# Both
bash scripts/train_all.sh
```

Configs: `configs/model_config.yaml` (d_model, nhead, triplet_margin, gamma, eager_weight) and `configs/training_config.yaml` (seed 42, max_pulses 2048, epochs 50, seq_len 16). All hyperparams config-driven, no magic numbers. Seed all RNGs from `seed:42`.

## Evaluation

```bash
bash scripts/evaluate_full.sh
# Produces: results/results.csv, aggregate_metrics.json, roc_curve.pdf, deinterleaving_performance.pdf
# Prints table: Achieved vs Baseline vs Targets
python -m src.evaluation.evaluate_full --deinterleaver-ckpt checkpoints/deinterleaver/best.pt --scheduler-ckpt checkpoints/scheduler/best.pt --config configs/model_config.yaml --test-dir data/test --output-dir results --mode scan
```

Metrics: **V-measure / AMI / ARI** (deinterleaving), **Pd / Pfa / Sensitivity / Avg Intercept Rate / Avg Intercept Time Error / Avg Reward**, ROC. Evaluation iterates 250 test pulse trains in stare/scan.

## API

```bash
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000  # DEVICE=cuda/cpu via env
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict_bands -H "Content-Type: application/json" -d '{"obs": [0.0, 0.1, 0.5, "...360 floats..."], "k": 1}'
# Response: {"bands":[42], "attribution":{"eager_pct":0.6,"revisit_pct":0.4}, "latency_ms":0.8}

curl -X POST http://localhost:8000/deinterleave -H "Content-Type: application/json" -d '{"pdws": [[100.0, 9000.0, 1.2, 30.0, 10.0], ...], "min_cluster_size": 10}'
curl -X POST http://localhost:8000/update_memory -H "Content-Type: application/json" -d '{"emitter_id":"E1","freq_min_mhz":9000,"freq_max_mhz":9010,"priority_score":0.9}'
curl http://localhost:8000/memory/emitters
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/reset
```

ONNX export for Jetson:
```bash
python -m src.deployment.export_onnx --model-config configs/model_config.yaml --deinterleaver-ckpt checkpoints/deinterleaver/best.pt --scheduler-ckpt checkpoints/scheduler/best.pt --output-dir checkpoints/onnx
```

## Docker

```bash
docker build -t cognitive-ew-smartscan .
docker run --gpus all -p 8000:8000 -e DEVICE=cuda -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/data:/app/data cognitive-ew-smartscan
# Jetson: docker run --runtime nvidia -p 8000:8000 cognitive-ew-smartscan
```

## Achieved vs Baseline vs Targets

| Metric | Baseline (HDBSCAN raw) | Target | Achieved (scan) | Notes |
|---|---|---|---|---|
| V-measure (scan) | 0.62 | 0.85 | *train & evaluate to fill* | Transformer+Triplet+HDBSCAN |
| AMI (scan) | — | — | *see aggregate_metrics.json* | |
| Pd | 0.65 | 0.90 | *see results.csv* | Thompson+DRQN+MoE |
| Pfa | 0.12 | 0.05 | *see roc_curve.pdf* | |
| Avg Intercept Rate | — | — | *metrics* | |
| Scheduler Latency | — | <1 ms | ONNX pre-export tested | Jetson AGX Orin |

## References

- TSRD & Challenge: https://github.com/alan-turing-institute/turing-deinterleaving-challenge, JC Wise, Radar Emitter Database 2024
- Transformer deinterleaving arXiv:2503.13476
- Electronics 2025 SmartScan (DRQN + Thompson)
- Dueling DQN (Wang et al. 2016), HDBSCAN (McInnes 2017)

## Project Docs

- `PRD.md` — requirements, users, features
- `Architecture.md` — structure, stack, flow
- `Rules.md` — file-local labels, error handling, config-driven
- `Design.md` — API & viz style
- `Memory.md` — progress tracker (update per phase)

## License

SIH 2026 — for evaluation. TSRD under ATI license.
