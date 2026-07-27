from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from mri_dl import MRIUndersampledDataset, evaluate_and_save_results
from src.k_space_utils import image_to_kspace


class ZeroDeltaModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _write_sample(split_root: Path, sample_id: str, original: np.ndarray, row_mask: np.ndarray, retain_ratio: float, slice_index: int) -> dict[str, object]:
    ratio_label = f"retain_{int(round(retain_ratio * 100)):02d}"
    original_path = Path("originals") / f"{sample_id}.npy"
    undersampled_path = Path("undersampled") / ratio_label / f"{sample_id}.npy"
    kspace_path = Path("k_spaces") / ratio_label / f"{sample_id}.npy"
    mask_id = f"mask_{sample_id}"
    mask_path = Path("masks") / ratio_label / f"{mask_id}.npy"

    k_space = image_to_kspace(original)
    harmed = k_space.copy()
    harmed[row_mask == 0, :] = 0
    undersampled = np.abs(np.fft.ifft2(np.fft.ifftshift(harmed))).astype(np.float32)

    np.save(split_root / original_path, original.astype(np.float32), allow_pickle=False)
    np.save(split_root / undersampled_path, undersampled, allow_pickle=False)
    np.save(split_root / kspace_path, harmed, allow_pickle=False)
    np.save(split_root / mask_path, row_mask.astype(np.uint8), allow_pickle=False)

    return {
        "sample_id": sample_id,
        "subject_id": "subject_a",
        "original_volume_path": "subject_a.npy",
        "resolved_volume_path": "subject_a.npy",
        "plane": "Coronal",
        "slice_index": int(slice_index),
        "original_image_file": str(original_path.as_posix()),
        "retain_ratio": float(retain_ratio),
        "mask_id": mask_id,
        "mask_file": str(mask_path.as_posix()),
        "k_space_file": str(kspace_path.as_posix()),
        "undersampled_image_file": str(undersampled_path.as_posix()),
        "source_metadata_split": "test",
        "source_shape": "[8, 8]",
    }


def _prepare_split(split_root: Path) -> None:
    (split_root / "originals").mkdir(parents=True, exist_ok=True)
    for base in ("undersampled", "masks", "k_spaces"):
        (split_root / base / "retain_30").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        split_root = root / "test"
        _prepare_split(split_root)

        rng = np.random.default_rng(7)
        rows = []
        for i in range(2):
            original = rng.random((8, 8), dtype=np.float32)
            row_mask = np.zeros(8, dtype=np.uint8)
            row_mask[[2, 5]] = 1
            rows.append(_write_sample(split_root, f"sample_{i:03d}", original, row_mask, 0.30, i))

        pd.DataFrame(rows).to_csv(split_root / "samples.csv", index=False)

        dataset = MRIUndersampledDataset(split_root, planes=("Coronal",), retain_ratios=(0.30,), load_mask=True)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        results_df = evaluate_and_save_results(
            model=ZeroDeltaModel(),
            test_loader=loader,
            output_root=root / "results",
            plane="Coronal",
            retain_ratio=0.30,
            checkpoint_name="dummy.pt",
            epoch=0,
            device="cpu",
        )

        expected_csv = root / "results" / "coronal" / "retain_30" / "test_per_image.csv"
        if not expected_csv.exists():
            raise FileNotFoundError(f"Expected output CSV was not created: {expected_csv}")

        loaded = pd.read_csv(expected_csv)
        assert len(loaded) == len(dataset)
        assert loaded["sample_id"].is_unique
        required_columns = {
            "sample_id", "volume_id", "plane", "slice_index", "retain_ratio", "mask_id", "masked_rows",
            "psnr_undersampled", "psnr_resunet", "psnr_resunet_dc",
            "ssim_undersampled", "ssim_resunet", "ssim_resunet_dc",
            "psnr_gain_resunet_vs_us", "psnr_gain_dc_vs_resunet", "psnr_gain_dc_vs_us",
            "ssim_gain_resunet_vs_us", "ssim_gain_dc_vs_resunet", "ssim_gain_dc_vs_us",
            "resunet_improved_psnr", "dc_improved_psnr", "final_improved_psnr_vs_us",
            "resunet_improved_ssim", "dc_improved_ssim", "final_improved_ssim_vs_us",
        }
        missing = required_columns - set(loaded.columns)
        if missing:
            raise AssertionError(f"Missing required result columns: {sorted(missing)}")

        print("Evaluation logging verification passed.")
        print(results_df.head())
        print(results_df.shape)

