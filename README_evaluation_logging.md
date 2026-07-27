# Per-image test evaluation logging

## Minimal experiment configuration

`mri_dl.ModelConfig` is the single experiment-level configuration. The
notebook `06_a_simple_cnn_for_nb_05.ipynb` creates it near the beginning; change
only `plane`, `retain_ratio`, and `run_mode` there to select a run.

Supported conditions are:

- planes: `axial`, `coronal`, `sagittal`
- retained k-space ratios: `0.20`, `0.30`, `0.50`
- run modes: `full` and `smoke`

Full runs use the configured dataset and epoch count. Smoke runs use two epochs
and the small train/validation/test limits defined on the same config. Result
directories are deterministic, for example `results/coronal_retain_30/` and
`results/smoke_coronal_retain_30/`. When passed to `train_model`, the effective
configuration is saved as `config_used.json` in that directory.

The configuration also exposes `dataset_filter_kwargs`, `model_kwargs`,
`make_train_config()`, and split-path properties so existing project APIs can
consume it without duplicating settings.

This project now includes persistent per-image test logging via:

- `mri_dl.evaluation.evaluate_and_save_results`

## What it saves

For each evaluated test image, one CSV row is written with:

- identification columns: `sample_id`, `volume_id`, `plane`, `slice_index`, `retain_ratio`, `mask_id`, `masked_rows`
- metrics: `psnr_undersampled`, `psnr_resunet`, `psnr_resunet_dc`, `ssim_undersampled`, `ssim_resunet`, `ssim_resunet_dc`
- gain columns and boolean improvement flags
- optional metadata such as checkpoint name, epoch, split, and inference time

## Metric policy

All three compared methods (undersampled, ResUNet, ResUNet+DC) are measured
against the same target image with a shared metric policy:

- PSNR: ROI policy used by `calculate_psnr` (`reference > 0` mask)
- SSIM: full-frame policy used by `calculate_ssim`
- Data range: computed from the reference image in both metrics
- No independent per-image rescaling before metric evaluation

## Output path

`<output_root>/<plane_lower>/retain_XX/test_per_image.csv`

Example:

`results/coronal/retain_30/test_per_image.csv`

## Minimal usage

```python
from pathlib import Path
from torch.utils.data import DataLoader
from mri_dl import MRIUndersampledDataset, ResidualUNet, load_checkpoint, evaluate_and_save_results

test_dataset = MRIUndersampledDataset(
    Path("../undersampled_dataset_split") / "test",
    csv_name="samples.csv",
    plane="Coronal",
    retain_ratio=0.30,
    load_mask=True,
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
model, checkpoint = load_checkpoint("checkpoints/coronal_r30_delta_best.pt", ResidualUNet)

results_df = evaluate_and_save_results(
    model=model,
    test_loader=test_loader,
    output_root=Path("results"),
    plane="Coronal",
    retain_ratio=0.30,
    checkpoint_name="coronal_r30_delta_best.pt",
    epoch=checkpoint.get("epoch"),
)

print(results_df.head())
print(results_df.shape)
```

## Quick verifier

A small non-destructive smoke test is available:

```python
python verify_evaluation_logging.py
```

