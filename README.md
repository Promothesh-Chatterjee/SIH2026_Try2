# Cognitive Electronic Warfare Smart Scan Strategy

[![CI](https://github.com/Promothesh-Chatterjee/SIH2026_Try2/actions/workflows/ci.yml/badge.svg)](https://github.com/Promothesh-Chatterjee/SIH2026_Try2/actions/workflows/ci.yml)
[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Vercel-0070F3?logo=vercel)](https://sih-2026-try2.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python)](https://python.org)
[![Coverage](https://img.shields.io/badge/Coverage-81%25-brightgreen)](https://github.com/Promothesh-Chatterjee/SIH2026_Try2)

Autonomous cognitive radar scanning strategy for Electronic Warfare (EW) Electronic Support (ES) receivers operating across 36 frequency bands under non-cooperative conditions. Designed for **DRDO Problem Statement SIH26056 (Smart India Hackathon 2026)**, the system intercepts, deinterleaves, and tracks non-cooperative radar emissions—including agile frequency-hopping emitters, periodic scanning search radars, and fixed emitters—without prior threat libraries. By combining high-purity windowed signal deinterleaving with a Dueling Deep Recurrent Q-Network (DRQN) scheduler, the receiver achieves sub-millisecond dwell scheduling decisions, outperforming classical search strategies by more than **11.5× in intercept rate**.

---

## End-to-End Architecture

```
                                  RF EMISSION ENVIRONMENT
   ┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
   │    Static Radar Emitter   │   Agile Hopping Emitter   │   Periodic Scan Radar     │
   │   (Fixed PRI, Duty 50%)   │ (Random/Markov Hop Sets)  │  (Sector Scan, Swept PRI) │
   └─────────────┬─────────────┴─────────────┬─────────────┴─────────────┬─────────────┘
                 │                           │                           │
                 └───────────────────────────┼───────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │    SpectrumEnvironment    │
                               │     (SmartScanEW-v0)      │
                               │  36 Bands · 180 Actions   │
                               │  360-D Observation Space  │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
          SIGNAL DEINTERLEAVER SUBSYSTEM               COGNITIVE SCHEDULER (DRQN)
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │ Pulse Descriptor Words (PDWs) │           │ Observation Vector (360-D)    │
         │  (ToA, Freq, PW, Amplitude)   │           │ 36 bands × 10 belief features │
         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                         ▼                                           ▼
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │  PDWFeatureExtractor & Scale  │           │   2-Layer LSTM Recurrent Core │
         │   (Robust IQR & Time Diffs)   │           │     (2 × 256 units, BPTT)     │
         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                         ▼                                           ▼
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │ WindowedDeinterleaver & Match │           │     Dueling Q-Value Heads     │
         │  (HDBSCAN / DBSCAN Fallback)  │           │ V(s) + [A_band(s,a) - mean A] │
         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                         ▼                                           ▼
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │    CrossWindowReconciler      │           │ Joint Action: (Band, Mode)    │
         │ (Purity ≥ 0.965, Identity St.)│           │ [SHORT/NORMAL/LONG/REV/PRE]   │
         └───────────────────────────────┘           └───────────────┬───────────────┘
                                                                     │
                                             ┌───────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │     EW Metrics Engine     │
                               │    (7 Figures of Merit)   │
                               │  Pd, Pfa, S_min, IR, etc. │
                               └─────────────┬─────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
         ┌──────────────────────────────┐          ┌──────────────────────────────┐
         │        FastAPI Server        │          │   Interactive Web Dashboard  │
         │  /api/v1/metrics             │ ───────► │   https://sih-2026-try2      │
         │  /api/v1/spectrum            │          │   .vercel.app                │
         └──────────────────────────────┘          └──────────────────────────────┘
```

---

## Figures of Merit (Validation Results)

Evaluation conducted on the 36-band RF environment ($T_{\text{steps}} = 1000$) with a multi-threat mix (frequency-hopping agile emitter + periodic scanning emitter) comparing the trained DRQN scheduler against the classical Round-Robin baseline:

| # | Figure of Merit (FoM) | Formula / Definition | Baseline (Round-Robin) | Achieved (DRQN Agent) | Target / Spec |
|:---:|:---|:---|:---:|:---:|:---:|
| **1** | **Probability of Detection ($P_d$)** | $TP / (TP + FN)$ on tuned active dwells | $1.000$ ($100\%$) | **$1.000$ ($100\%$)** | $\ge 0.90$ |
| **2** | **Probability of False Alarm ($P_{fa}$)** | $FP / (FP + TN)$ on tuned empty dwells | $0.000$ ($0.0\%$) | **$0.000$ ($0.0\%$)** | $\le 0.05$ |
| **3** | **Receiver Sensitivity ($S_{\min}$)** | Minimum detectable signal threshold | $-140.0\,\text{dBm}$ | **$-140.0\,\text{dBm}$** | $-140.0\,\text{dBm}$ |
| **4** | **Average Intercept Rate** | $n_{\text{intercepts}} / T_{\text{steps}}$ | $0.0650$ ($6.5\%$) | **$0.7480$ ($74.8\%$)** | **$> 11.5\times$ Gain** |
| **5** | **Average Reward per Dwell** | Mean environment reinforcement return | $+0.0650$ | **$+0.7480$** | Positive & maximal |
| **6** | **Prediction Accuracy** | Dwells landing on active emitter bands | $6.50\%$ | **$74.80\%$** | Outperform sweep |
| **7** | **Avg Intercept Time Error** | Temporal ToA deviation on intercept | $0.00\,\mu\text{s}$ | **$0.00\,\mu\text{s}$** | $< 5.00\,\mu\text{s}$ |

---

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Promothesh-Chatterjee/SIH2026_Try2.git
cd SIH2026_Try2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install package in editable mode with all dependencies
pip install -e .
```

### 2. Run End-to-End Pipeline Validation
Execute the gate validation script to verify the RF environment, DRQN scheduler, deinterleaving pipeline, and EW Figures of Merit:
```bash
python scripts/validate_pipeline.py
```
*Expected output: exits with code `0`, reporting all 7 FoMs and confirming $>11.5\times$ intercept improvement over Round-Robin.*

### 3. Run Test Suite with Coverage
```bash
pytest ew_core/tests/ --cov=ew_core --cov-fail-under=70
```
*Executes all 774 unit, integration, and regression tests with strict coverage enforcement ($\ge 81\%$ achieved).*

> [!NOTE]
> Baseline-immutability and reservoir tests skip in CI environments where checkpoint files are not hosted locally — Sprint 4 closes this gap with hosted artifact downloads.

### 4. Launch Backend API Server
```bash
uvicorn ew_core.deployment.api:app --host 0.0.0.0 --port 8000 --reload
```
Endpoints available:
- `GET /health` — Service health and GPU/CPU device metadata
- `GET /api/v1/metrics` — Latest episode EW figures of merit ($P_d, P_{fa}$, intercept rate, etc.)
- `GET /api/v1/spectrum` — Time-frequency truth matrix and receiver dwell positions
- `POST /predict_bands` — Inference endpoint for scheduler action selection
- `POST /deinterleave` — High-speed PDW clustering and emitter separation

---

## Cloud Deployment & Operations (Azure AKS)

The cognitive EW microservice is containerized and deployed to **Azure Kubernetes Service (AKS)** with zero out-of-pocket costs ($0.00 model via Azure for Students & GitHub Container Registry):

* **Backend Live Endpoint**: [http://172.198.227.59](http://172.198.227.59) *(start AKS first)*
* **Container Registry**: `ghcr.io/promothesh-chatterjee/sih2026_try2:latest` (GHCR)
* **Storage**: Azure Blob Storage (`smartscanstore4301`) for TSRD dataset & neural models

### Single-Command Cluster Lifecycle Management
To preserve compute credits when not conducting demonstrations:

```powershell
# 1. Start / Resume AKS cluster (takes ~2 minutes)
az aks start --name smartscan-aks --resource-group smartscan-rg

# 2. Run operational smoke test against live endpoint
python scripts/smoke_test.py --api_url http://172.198.227.59 --api_key smartscan-sih2026-demo-key

# 3. Stop / Pause AKS cluster (freezes compute billing at $0.00)
az aks stop --name smartscan-aks --resource-group smartscan-rg
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full architectural specifications, storage mounting, and CI/CD automation.
