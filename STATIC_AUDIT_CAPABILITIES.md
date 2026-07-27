# Static Capability Audit

| Capability | Exist? | File / Function | Missing / Concern |
|---|---|---|---|
| Dataset loading | Yes | `mri_dl/dataset.py` - `MRIUndersampledDataset.__init__`, `__getitem__` | None |
| Train/validation/test separation | Yes | `mri_dl/experiment_config.py` - `train_data_root`, `validation_data_root`, `test_data_root`; also notebook split loaders | None |
| Model creation | Yes | `mri_dl/model.py` - `ResidualUNet` | None |
| Training loop | Yes | `mri_dl/train_utils.py` - `train_model` (epoch loop), `run_epoch` (optimizer step) | None |
| Validation loop | Yes | `mri_dl/train_utils.py` - `run_epoch` called without optimizer for validation | None |
| Best-checkpoint saving | Yes | `mri_dl/train_utils.py` - `save_checkpoint`; `train_model` best-val tracking and save | None |
| Checkpoint loading | Yes | `mri_dl/train_utils.py` - `load_checkpoint` | None |
| Inference | Yes | `mri_dl/inference.py` - `predict_tensor`, `predict_numpy`, `predict_file` | None |
| Residual reconstruction | Yes | `mri_dl/model.py` residual target design; `mri_dl/inference.py` adds predicted delta back to input | None |
| Data-consistency post-processing | Yes | `src/k_space_utils.py` - `enforce_kspace_data_consistency` | None |
| PSNR and SSIM calculation | Yes | `src/metrices.py` - `calculate_psnr`, `calculate_ssim` | File name typo (`metrices.py`) is non-blocking but confusing |
| Per-image CSV logging | Yes | `mri_dl/evaluation.py` - `evaluate_and_save_results` writes per-image rows to CSV | None |
| Configuration selection | Yes | `mri_dl/experiment_config.py` - `ExperimentConfig` validation + selected plane/retain ratio + training params | Smoke mode intentionally removed by user preference |
| Result-directory creation | Yes | `mri_dl/experiment_config.py` - `result_dir`; `mri_dl/evaluation.py` uses `mkdir(parents=True, exist_ok=True)` | None |
| Visualization | Yes | `mri_dl/model.py` - `visualize_model`; notebooks include sample/prediction plotting cells | Notebook cells can break if assumptions on tensor shapes/index uniqueness are changed |
| **US dataset creation from original database (Notebook 05)** | **Yes** | `notebooks/05_create_dl_small_data_set.ipynb` calls `create_dataset_split`; `src/create_mri_dataset.py` - `create_dataset_split`, `_create_split_from_rows`, `create_unique_row_mask`, `apply_row_mask` | None |

## Notes
- Scope: static code audit only (no refactor, no new runtime logic).
- "US" row is now interpreted as **undersampled dataset creation from original data** (Notebook 05 pipeline), not baseline metric columns.
- User preference recorded: keep implementation minimalistic; no extra mode/framework unless needed.
