# Development Rules & Constraints

## 1. Code Completeness
- **NO STUBS.** Every function body must be fully implemented.
- Do not use `pass`, `raise NotImplementedError`, or `# TODO` comments.

## 2. Typing & Documentation
- **Type Annotations:** All functions require complete parameter and return type hints using Python 3.10+ syntax (e.g., `list[str] | None`).
- **Docstrings:** All classes and public functions must use Google-style docstrings (Args, Returns, Raises, Description).

## 3. Data Integrity & Validation (CRITICAL)
- **File-Local Labels:** Emitter labels in the TSRD dataset are unique *only per file*. Label `1` in `file_A.h5` is entirely unrelated to label `1` in `file_B.h5`.
- **Constraint Enforcement:** In any batching, collation, or triplet mining logic, explicitly assert and comment that pulses from different `.h5` files are never mixed for label comparison.

## 4. Error Handling & Robustness
- **Graceful Failures:** Catch missing files, empty pulse trains, and corrupt `.h5` files.
- **Hardware Fallback:** Catch CUDA OOM errors and dynamically fallback to CPU or smaller batch sizes where applicable.
- **Safe Inference:** Use `@torch.inference_mode()` for all evaluation and prediction steps.

## 5. Configuration & Reproducibility
- **No Magic Numbers:** All hyperparameters (model dimensions, reward weights, RL epsilon values) must be loaded from `configs/model_config.yaml` and `configs/training_config.yaml`.
- **Seeding:** Seed all RNGs (PyTorch, NumPy, Python Random) deterministically using the configuration seed.
- **Data Leakage Prevention:** Do not use `fit_stats` calculated on the test/evaluation datasets; use only pre-computed train statistics.

## 6. Execution Quality
- Use `logging` instead of `print()`. Categorize via `INFO`, `DEBUG`, `WARNING`, `ERROR`.
- Ensure modularity so scripts can run independently.
