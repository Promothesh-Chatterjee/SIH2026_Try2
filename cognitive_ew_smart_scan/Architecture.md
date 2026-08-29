# Architecture Document

## 1. File and Folder Structure
```
cognitive_ew_smart_scan/
├── README.md
├── requirements.txt
├── .env                          # HF_TOKEN
├── Dockerfile
├── data/                         # TSRD data (gitignored)
│   ├── stare/
│   └── scan/
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis
│   ├── 02_baseline_hdbscan.ipynb # reproduce ATI baseline
│   └── 03_evaluation.ipynb       # final figures of merit
├── src/
│   ├── environment/
│   │   ├── state_matrix.py       # binary transmission matrix builder
│   │   └── rf_scan_env.py        # OpenAI Gymnasium environment
│   ├── preprocessing/
│   │   └── normalise.py          # PDW feature normalisation
│   ├── models/
│   │   ├── deinterleaver.py      # Transformer encoder
│   │   ├── drqn_scheduler.py     # Deep Recurrent Q-Network
│   │   └── smartscan_moe.py      # Eager + Revisit + MoE fusion
│   ├── training/
│   │   ├── train_deinterleaver.py # triplet loss training loop
│   │   ├── train_scheduler.py    # DRQN training with TSRD env
│   │   ├── reward.py             # reward function
│   │   └── thompson_sampling.py  # MAB exploration warmup
│   ├── cognitive/
│   │   ├── memory.py             # episodic + semantic memory
│   │   └── periodic_interceptor.py # phase-locked periodic scan intercept
│   ├── evaluation/
│   │   ├── metrics.py            # FiguresOfMerit class
│   │   └── evaluate_full.py      # full test-set evaluation pipeline
│   └── deployment/
│       ├── export_onnx.py        # model export
│       └── api.py                # FastAPI REST microservice
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
└── scripts/
    ├── download_data.py
    ├── train_all.sh
    └── evaluate.sh
```

## 2. Tech Stack
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch >= 2.3 (CUDA 12.1)
- **Reinforcement Learning:** Gymnasium, custom PyTorch training loops
- **Data & Processing:** NumPy, SciPy, Pandas, h5py, pyarrow
- **Clustering:** HDBSCAN, Scikit-Learn
- **API & Deployment:** FastAPI, Uvicorn, ONNX, ONNXRuntime-GPU
- **Logging & Viz:** Weights & Biases (wandb), Matplotlib, Seaborn, Plotly

## 3. App Flow
1. **Data Ingestion:** Interleaved RF pulses (PDWs) enter the system.
2. **Preprocessing:** PDWs are normalized (robust Z-score, log1p, min-max). Angle of Arrival is converted to sine/cosine coordinates (6D total).
3. **Deinterleaving:** Transformer encoder processes pulse sequences to embed them; HDBSCAN clusters embeddings into unique emitter IDs.
4. **Cognitive Memory Analysis:** The Periodic Interceptor analyzes PRI/ToA histograms to predict future occurrences. Known patterns are stored in SQLite Semantic Memory.
5. **Scheduler (MoE):** 
   - *Eager Agent (DRQN)* predicts the value of scanning specific frequency bands based on environmental state memory (LSTM).
   - *Revisit Agent* generates urgency scores for bands not scanned recently.
   - MoE fuses scores, selecting the optimal band to tune the narrow-band receiver.
6. **Execution/Update:** The receiver scans the band, updates observations, logs metrics, and updates the RL environment's state.
