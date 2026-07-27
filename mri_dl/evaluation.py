from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.k_space_utils import enforce_kspace_data_consistency
from src.metrices import calculate_psnr, calculate_ssim
from .inference import predict_tensor
from .train_utils import choose_device


def _batch_item(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        item = value[index]
        if item.ndim == 0:
            return item.item()
        return item
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _ensure_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def evaluate_and_save_results(
    model: nn.Module,
    test_loader: DataLoader,
    output_root: str | Path,
    plane: str,
    retain_ratio: float,
    checkpoint_name: str | None = None,
    epoch: int | None = None,
    clamp_to_unit_range: bool = True,
) -> pd.DataFrame:
    """Evaluate test samples and persist one row per image to CSV."""
    dataset = test_loader.dataset
    split_root = getattr(dataset, "split_root", None)
    if split_root is None:
        raise ValueError("test_loader.dataset must expose split_root to resolve mask/k-space files.")

    resolved_device = choose_device()
    model.to(resolved_device)
    model.eval()

    records: list[dict[str, Any]] = []
    evaluated_samples = 0

    with torch.inference_mode():
        for batch in test_loader:
            inputs = batch["input"]
            targets = batch["target"]
            batch_predictions = predict_tensor(
                model=model,
                input_tensor=inputs,
                device=resolved_device,
                clamp_to_unit_range=clamp_to_unit_range,
            )

            batch_size = int(inputs.shape[0])
            for i in range(batch_size):
                evaluated_samples += 1

                undersampled = inputs[i, 0].detach().cpu().numpy()
                target = targets[i, 0].detach().cpu().numpy()
                resunet = batch_predictions[i, 0].detach().cpu().numpy()
                if undersampled.shape != target.shape or resunet.shape != target.shape:
                    raise ValueError(
                        f"Shape mismatch at batch item {i}: "
                        f"undersampled={undersampled.shape}, resunet={resunet.shape}, target={target.shape}"
                    )
                if not np.all(np.isfinite(undersampled)) or not np.all(np.isfinite(target)) or not np.all(np.isfinite(resunet)):
                    raise ValueError(f"NaN/Inf detected in evaluation tensors at batch item {i}.")

                k_space_file = _ensure_str(_batch_item(batch.get("k_space_file"), i), default="")
                mask_file = _ensure_str(_batch_item(batch.get("mask_file"), i), default="")
                if not k_space_file:
                    raise KeyError("Batch item is missing 'k_space_file'.")
                if not mask_file:
                    raise KeyError("Batch item is missing 'mask_file'.")

                acquired_k_space = np.load(Path(split_root) / k_space_file, allow_pickle=False)
                if "mask" in batch:
                    row_mask = _batch_item(batch["mask"], i).detach().cpu().numpy()
                else:
                    row_mask = np.load(Path(split_root) / mask_file, allow_pickle=False)
                row_mask = np.asarray(row_mask)
                if row_mask.ndim != 1:
                    raise ValueError(f"row_mask must be 1D; got {row_mask.shape}")
                if row_mask.shape[0] != acquired_k_space.shape[0]:
                    raise ValueError(
                        f"row_mask length ({row_mask.shape[0]}) does not match k-space rows ({acquired_k_space.shape[0]})."
                    )
                if not np.all(np.isin(row_mask, (0, 1))):
                    raise ValueError("row_mask must contain only 0/1 values.")
                if not np.iscomplexobj(acquired_k_space):
                    raise ValueError("acquired_k_space must be complex-valued.")
                if not np.all(np.isfinite(acquired_k_space)):
                    raise ValueError("acquired_k_space contains NaN/Inf values.")

                retain_ratio_value = float(_batch_item(batch.get("retain_ratio"), i))

                observed_removed_rows = int(np.count_nonzero(row_mask == 0))
                expected_removed_rows = int(acquired_k_space.shape[0] - np.count_nonzero(row_mask))
                if observed_removed_rows != expected_removed_rows:
                    raise ValueError(
                        f"Removed-row count mismatch: observed={observed_removed_rows}, expected={expected_removed_rows}."
                    )

                expected_retain_rows = int(round(acquired_k_space.shape[0] * retain_ratio_value))
                observed_retain_rows = int(np.count_nonzero(row_mask))
                if abs(expected_retain_rows - observed_retain_rows) > 1:
                    raise ValueError(
                        f"Retain-row count mismatch for sample {i}: observed={observed_retain_rows}, "
                        f"expected~={expected_retain_rows} (ratio={retain_ratio_value})."
                    )

                zero_line_mask = row_mask.astype(bool, copy=False) == 0

                t0 = perf_counter()
                final_image, _, _ = enforce_kspace_data_consistency(
                    reconstructed_image=resunet,
                    acquired_k_space=acquired_k_space,
                    zero_line_mask=zero_line_mask,
                )
                if clamp_to_unit_range:
                    final_image = final_image.clip(0.0, 1.0)
                if not np.all(np.isfinite(final_image)):
                    raise ValueError("final_image contains NaN/Inf values after post-processing.")
                inference_time_ms = (perf_counter() - t0) * 1000.0

                psnr_undersampled = float(calculate_psnr(target, undersampled))
                psnr_resunet = float(calculate_psnr(target, resunet))
                psnr_resunet_dc = float(calculate_psnr(target, final_image))
                ssim_undersampled = float(calculate_ssim(target, undersampled))
                ssim_resunet = float(calculate_ssim(target, resunet))
                ssim_resunet_dc = float(calculate_ssim(target, final_image))

                sample_id = _ensure_str(_batch_item(batch.get("sample_id"), i), default="")
                plane_value = _ensure_str(_batch_item(batch.get("plane"), i), default=plane)
                slice_index_value = _batch_item(batch.get("slice_index"), i)
                slice_index = int(slice_index_value) if slice_index_value is not None else -1
                mask_id = _ensure_str(_batch_item(batch.get("mask_id"), i), default=Path(mask_file).stem)
                resolved_volume_path = _ensure_str(_batch_item(batch.get("resolved_volume_path"), i), default="")
                subject_id = _ensure_str(_batch_item(batch.get("subject_id"), i), default="")
                volume_id = subject_id or (Path(resolved_volume_path).stem if resolved_volume_path else "unknown")

                if not sample_id:
                    sample_id = f"{volume_id}_{plane_value}_s{slice_index:04d}_{mask_id}"

                masked_rows = json.dumps(np.where(row_mask == 0)[0].astype(int).tolist())

                record = {
                    "sample_id": sample_id,
                    "volume_id": volume_id,
                    "plane": plane_value,
                    "slice_index": int(slice_index),
                    "retain_ratio": float(retain_ratio_value),
                    "mask_id": mask_id,
                    "masked_rows": masked_rows,
                    "psnr_undersampled": psnr_undersampled,
                    "psnr_resunet": psnr_resunet,
                    "psnr_resunet_dc": psnr_resunet_dc,
                    "ssim_undersampled": ssim_undersampled,
                    "ssim_resunet": ssim_resunet,
                    "ssim_resunet_dc": ssim_resunet_dc,
                    "psnr_gain_resunet_vs_us": psnr_resunet - psnr_undersampled,
                    "psnr_gain_dc_vs_resunet": psnr_resunet_dc - psnr_resunet,
                    "psnr_gain_dc_vs_us": psnr_resunet_dc - psnr_undersampled,
                    "ssim_gain_resunet_vs_us": ssim_resunet - ssim_undersampled,
                    "ssim_gain_dc_vs_resunet": ssim_resunet_dc - ssim_resunet,
                    "ssim_gain_dc_vs_us": ssim_resunet_dc - ssim_undersampled,
                    "resunet_improved_psnr": bool(psnr_resunet > psnr_undersampled),
                    "dc_improved_psnr": bool(psnr_resunet_dc > psnr_resunet),
                    "final_improved_psnr_vs_us": bool(psnr_resunet_dc > psnr_undersampled),
                    "resunet_improved_ssim": bool(ssim_resunet > ssim_undersampled),
                    "dc_improved_ssim": bool(ssim_resunet_dc > ssim_resunet),
                    "final_improved_ssim_vs_us": bool(ssim_resunet_dc > ssim_undersampled),
                    "checkpoint_name": checkpoint_name or "",
                    "epoch": int(epoch) if epoch is not None else np.nan,
                    "split": "test",
                    "normalization_method": "clamp_to_unit_range" if clamp_to_unit_range else "none",
                    "data_range": "reference_min_max (PSNR/SSIM), no per-image rescaling before metrics",
                    "metric_policy": "PSNR_ROI(reference>0), SSIM_full_frame",
                    "inference_time_ms": float(inference_time_ms),
                }
                records.append(record)

    results_df = pd.DataFrame(records)
    if len(results_df) != evaluated_samples:
        raise ValueError(
            f"Row count mismatch: built {len(results_df)} rows for {evaluated_samples} evaluated samples."
        )

    duplicate_ids = results_df[results_df.duplicated("sample_id", keep=False)]["sample_id"].unique()
    if len(duplicate_ids) > 0:
        raise ValueError(f"Duplicate sample_id values detected: {duplicate_ids.tolist()}")

    results_df = results_df.sort_values(
        by=["volume_id", "plane", "slice_index", "mask_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    ratio_label = f"retain_{int(round(float(retain_ratio) * 100)):02d}"
    output_dir = Path(output_root) / str(plane).lower() / ratio_label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_per_image.csv"
    results_df.to_csv(output_path, index=False)

    unique_volumes = int(results_df["volume_id"].nunique()) if len(results_df) else 0
    print(f"Saved rows: {len(results_df)}")
    print(f"Output path: {output_path}")
    print(f"Unique volumes: {unique_volumes}")
    print(f"Plane: {plane}")
    print(f"Retain ratio: {retain_ratio}")

    return results_df


