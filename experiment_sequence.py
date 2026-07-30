from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from matplotlib.lines import Line2D

from mri_dl import (
	ModelConfig,
	MRIUndersampledDataset,
	RETAIN_RATIOS,
	ResidualUNet,
	evaluate_and_save_results,
	load_checkpoint,
	predict_numpy,
	predict_numpy_with_data_consistency,
	train_model, choose_device,
)
from src.create_mri_dataset import DatasetCreationPlan, SplitMultiplicityConfig, create_dataset_split
from src.general_utils import BRAIN_PLANES, SCV_FILES
from src.k_space_utils import image_to_kspace, kspace_log_magnitude
from src.metrices import calculate_psnr, calculate_ssim
from src.evaluation.report_outputs import METHOD_ORDER, generate_all_evaluation_outputs


def _format_duration(seconds: float) -> str:
	seconds = max(0.0, float(seconds))
	hours, remainder = divmod(seconds, 3600.0)
	minutes, secs = divmod(remainder, 60.0)
	if hours >= 1:
		return f"{int(hours)}h {int(minutes)}m {secs:05.2f}s"
	if minutes >= 1:
		return f"{int(minutes)}m {secs:05.2f}s"
	return f"{secs:.2f}s"


def _read_csv_frame(path: Path) -> pd.DataFrame:
	return cast(pd.DataFrame, cast(Any, pd.read_csv)(str(path)))


def _combine_group_mean_std(group: pd.DataFrame, metric_name: str) -> tuple[float, float]:
	counts = pd.to_numeric(group["count"], errors="coerce").to_numpy(dtype=float)
	means = pd.to_numeric(group[f"{metric_name}_mean"], errors="coerce").to_numpy(dtype=float)
	stds = pd.to_numeric(group[f"{metric_name}_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
	valid = np.isfinite(counts) & np.isfinite(means) & (counts > 0)
	counts = counts[valid]
	means = means[valid]
	stds = stds[valid]
	if counts.size == 0:
		return float("nan"), float("nan")
	total_count = float(np.sum(counts))
	pooled_mean = float(np.sum(counts * means) / total_count)
	if total_count <= 1:
		return pooled_mean, float("nan")
	within = np.sum(np.maximum(counts - 1.0, 0.0) * (stds ** 2))
	between = np.sum(counts * ((means - pooled_mean) ** 2))
	pooled_var = float((within + between) / (total_count - 1.0))
	return pooled_mean, float(np.sqrt(max(pooled_var, 0.0)))


def _build_expected_aggregate_csv_paths(results_root: Path, selected_planes: list[str], retain_ratios: list[float]) -> list[Path]:
	paths: list[Path] = []
	for plane in selected_planes:
		for retain_ratio in retain_ratios:
			retain_token = int(round(float(retain_ratio) * 100))
			paths.append(
				results_root / plane / f"retain_{retain_token}" / "report_outputs" / f"aggregate_metrics_{plane}_{retain_token}.csv"
			)
	return paths


def _aggregate_all_planes_metrics(results_root: Path, selected_planes: list[str], retain_ratios: list[float]) -> Path | None:
	expected_paths = _build_expected_aggregate_csv_paths(results_root, selected_planes, retain_ratios)
	existing_paths = [path for path in expected_paths if path.exists()]
	missing_paths = [path for path in expected_paths if not path.exists()]
	for path in missing_paths:
		print(f"Warning: aggregate CSV not found, skipping: {path}")
	if not existing_paths:
		print("Warning: no per-run aggregate CSV files found; skipping all-planes aggregation.")
		return None
	frames = [_read_csv_frame(path) for path in existing_paths]
	combined = pd.concat(frames, ignore_index=True)
	required_cols = {"method", "sampling_ratio", "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"}
	missing_cols = required_cols - set(combined.columns)
	if missing_cols:
		raise ValueError(f"Missing required aggregate columns in input CSVs: {sorted(missing_cols)}")
	rows: list[dict[str, float | str]] = []
	for (method, sampling_ratio), group in combined.groupby(["method", "sampling_ratio"], dropna=False, sort=False):
		psnr_mean, psnr_std = _combine_group_mean_std(group, "psnr")
		ssim_mean, ssim_std = _combine_group_mean_std(group, "ssim")
		rows.append(
			{
				"method": str(method),
				"sampling_ratio": float(sampling_ratio),
				"count": int(pd.to_numeric(group["count"], errors="coerce").fillna(0).sum()),
				"psnr_mean": psnr_mean,
				"psnr_std": psnr_std,
				"ssim_mean": ssim_mean,
				"ssim_std": ssim_std,
			}
		)
	result = pd.DataFrame(rows)
	method_rank = {name: idx for idx, name in enumerate(METHOD_ORDER)}
	result["_method_rank"] = result["method"].map(lambda m: method_rank.get(str(m), len(method_rank)))
	result = result.sort_values(["sampling_ratio", "_method_rank"], ascending=[False, True], kind="mergesort").drop(columns=["_method_rank"])
	result = result[["method", "sampling_ratio", "count", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]].reset_index(drop=True)
	output_path = results_root / "aggregated_all_planes_metrics.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(output_path, index=False)
	print(f"Created all-planes aggregate CSV: {output_path}")
	return output_path


def _create_root_ratio_plots(results_root: Path, aggregated_csv_path: Path | None) -> list[Path]:
	if aggregated_csv_path is None or not aggregated_csv_path.exists():
		print("Warning: no aggregated all-planes CSV found; skipping ratio plots.")
		return []

	aggregated = _read_csv_frame(aggregated_csv_path)
	required_cols = {"method", "sampling_ratio", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"}
	missing_cols = required_cols - set(aggregated.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in aggregated CSV: {sorted(missing_cols)}")

	method_labels = {
		"zero_filled": "Zero-filled",
		"resunet": "ResUNet",
		"resunet_data_consistency": "ResUNet + DC",
	}
	method_colors = {
		"zero_filled": "tab:blue",
		"resunet": "tab:orange",
		"resunet_data_consistency": "tab:green",
	}
	method_markers = {
		"zero_filled": "o",
		"resunet": "o",
		"resunet_data_consistency": "o",
	}
	ratio_order = [0.20, 0.30, 0.50]
	ratio_xticks = np.arange(len(ratio_order))
	ratio_xticklabels = [f"{int(round(r * 100))}%" for r in ratio_order]

	output_paths: list[Path] = []
	for metric, ylabel, output_base in (
		("psnr", "PSNR (dB)", results_root / "psnr_vs_ratio_all_planes"),
		("ssim", "SSIM", results_root / "ssim_vs_ratio_all_planes"),
	):
		fig, ax = plt.subplots(figsize=(13.0, 7.6))
		for method in ("zero_filled", "resunet", "resunet_data_consistency"):
			subset = aggregated[aggregated["method"].astype(str) == method].copy()
			if subset.empty:
				continue
			subset = subset.sort_values("sampling_ratio", kind="mergesort")
			x = np.arange(len(subset))
			y = subset[f"{metric}_mean"].astype(float).to_numpy(copy=True)
			yerr = subset[f"{metric}_std"].astype(float).fillna(0.0).to_numpy(copy=True)
			ax.errorbar(
				x,
				y,
				yerr=yerr,
				marker=method_markers[method],
				linewidth=4.0,
				markersize=12,
				capsize=8,
				color=method_colors[method],
				label=method_labels[method],
			)
		ax.set_xticks(ratio_xticks)
		ax.set_xticklabels(ratio_xticklabels)
		ax.set_xlabel("Retained k-space rows", fontsize=24)
		ax.set_ylabel(ylabel, fontsize=24)
		title = "PSNR (dB) vs. sampling ratio — all planes" if metric == "psnr" else "SSIM vs. sampling ratio — all planes"
		ax.set_title(title, fontsize=24, pad=18)
		ax.grid(True, alpha=0.35, linewidth=1.0)
		ax.tick_params(axis="both", labelsize=20, width=1.5, length=6)
		ax.legend(loc="upper left", fontsize=22, frameon=True)
		fig.tight_layout()
		for fmt in ("png", "pdf"):
			fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
		output_paths.append(output_base.with_suffix(".png"))
		output_paths.append(output_base.with_suffix(".pdf"))
		plt.close(fig)
	print(f"Created root-level ratio plots at: {results_root}")
	return output_paths


def _build_expected_test_csv_paths(results_root: Path, selected_planes: list[str], retain_ratios: list[float]) -> list[Path]:
	paths: list[Path] = []
	for plane in selected_planes:
		for retain_ratio in retain_ratios:
			retain_token = int(round(float(retain_ratio) * 100))
			paths.append(results_root / plane / f"retain_{retain_token}" / "test_per_image.csv")
	return paths


def _create_root_psnr_scatter(results_root: Path, selected_planes: list[str], retain_ratios: list[float]) -> Path | None:
	expected_paths = _build_expected_test_csv_paths(results_root, selected_planes, retain_ratios)
	existing_paths = [path for path in expected_paths if path.exists()]
	missing_paths = [path for path in expected_paths if not path.exists()]
	for path in missing_paths:
		print(f"Warning: test CSV not found, skipping: {path}")
	if not existing_paths:
		print("Warning: no per-image test CSV files found; skipping root scatter plot.")
		return None

	frames = [_read_csv_frame(path) for path in existing_paths]
	combined = pd.concat(frames, ignore_index=True)
	required_cols = {"retain_ratio", "psnr_undersampled", "psnr_resunet_dc"}
	missing_cols = required_cols - set(combined.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in test CSVs: {sorted(missing_cols)}")

	combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=["retain_ratio", "psnr_undersampled", "psnr_resunet_dc"])
	if combined.empty:
		print("Warning: combined test metrics are empty; skipping root scatter plot.")
		return None

	x = combined["psnr_undersampled"].astype(float).to_numpy(copy=True)
	y = combined["psnr_resunet_dc"].astype(float).to_numpy(copy=True)
	ratio_order = sorted(float(ratio) for ratio in combined["retain_ratio"].dropna().astype(float).unique())
	ratio_colors = {
		0.20: "tab:blue",
		0.30: "tab:orange",
		0.50: "tab:green",
	}
	fig, ax = plt.subplots(figsize=(10.6, 8.6))
	for ratio in ratio_order:
		subset = combined[np.isclose(combined["retain_ratio"].astype(float), ratio)]
		if subset.empty:
			continue
		color = ratio_colors.get(round(ratio, 2), "tab:blue")
		label = f"{int(round(ratio * 100))}% retained"
		ax.scatter(
			subset["psnr_undersampled"],
			subset["psnr_resunet_dc"],
			s=42,
			alpha=0.35,
			color=color,
			edgecolors="none",
			label=label,
		)

	lo = float(np.nanmin(np.concatenate([x, y])))
	hi = float(np.nanmax(np.concatenate([x, y])))
	pad = 0.03 * max(1.0, hi - lo)
	lo -= pad
	hi += pad
	ax.plot([lo, hi], [lo, hi], linestyle="--", color="tab:blue", linewidth=2.2, label="Equal performance")
	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_xlabel("Zero-filled PSNR (dB)")
	ax.set_ylabel("ResUNet + DC PSNR (dB)")
	ax.set_title("Sample-wise PSNR comparison — all planes and ratios")
	ax.grid(True, alpha=0.25)

	pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
	improved_rate = float(np.mean(y > x) * 100.0)
	combined_count = int(combined.shape[0])
	info_text = f"Pearson r = {pearson_r:.3f}\nImproved samples = {improved_rate:.1f}%\nn = {combined_count:,}"
	ax.text(
		0.03,
		0.97,
		info_text,
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=12,
		bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.35", alpha=0.95),
	)

	handles = [
		Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=ratio_colors.get(ratio, "tab:blue"), markeredgecolor="none", markersize=7, alpha=0.35, label=f"{int(round(ratio * 100))}% retained")
		for ratio in ratio_order
	]
	handles.append(Line2D([0], [0], linestyle="--", color="tab:blue", linewidth=2.2, label="Equal performance"))
	ax.legend(handles=handles, loc="lower right", frameon=True)
	fig.tight_layout()

	output_base = results_root / "sample_wise_psnr_comparison_all_planes_and_ratios"
	for fmt in ("png", "pdf"):
		fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Created root-level PSNR comparison scatter: {output_base.with_suffix('.png')} and {output_base.with_suffix('.pdf')}")
	return output_base.with_suffix(".png")


def _create_root_ssim_scatter(results_root: Path, selected_planes: list[str], retain_ratios: list[float]) -> Path | None:
	expected_paths = _build_expected_test_csv_paths(results_root, selected_planes, retain_ratios)
	existing_paths = [path for path in expected_paths if path.exists()]
	missing_paths = [path for path in expected_paths if not path.exists()]
	for path in missing_paths:
		print(f"Warning: test CSV not found, skipping: {path}")
	if not existing_paths:
		print("Warning: no per-image test CSV files found; skipping root scatter plot.")
		return None

	frames = [_read_csv_frame(path) for path in existing_paths]
	combined = pd.concat(frames, ignore_index=True)
	required_cols = {"retain_ratio", "ssim_undersampled", "ssim_resunet_dc"}
	missing_cols = required_cols - set(combined.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in test CSVs: {sorted(missing_cols)}")

	combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=["retain_ratio", "ssim_undersampled", "ssim_resunet_dc"])
	if combined.empty:
		print("Warning: combined test metrics are empty; skipping root scatter plot.")
		return None

	x = combined["ssim_undersampled"].astype(float).to_numpy(copy=True)
	y = combined["ssim_resunet_dc"].astype(float).to_numpy(copy=True)
	ratio_order = sorted(float(ratio) for ratio in combined["retain_ratio"].dropna().astype(float).unique())
	ratio_colors = {
		0.20: "tab:blue",
		0.30: "tab:orange",
		0.50: "tab:green",
	}
	fig, ax = plt.subplots(figsize=(10.6, 8.6))
	for ratio in ratio_order:
		subset = combined[np.isclose(combined["retain_ratio"].astype(float), ratio)]
		if subset.empty:
			continue
		color = ratio_colors.get(round(ratio, 2), "tab:blue")
		label = f"{int(round(ratio * 100))}% retained"
		ax.scatter(
			subset["ssim_undersampled"],
			subset["ssim_resunet_dc"],
			s=42,
			alpha=0.35,
			color=color,
			edgecolors="none",
			label=label,
		)

	lo = float(np.nanmin(np.concatenate([x, y])))
	hi = float(np.nanmax(np.concatenate([x, y])))
	pad = 0.03 * max(1.0, hi - lo)
	lo -= pad
	hi += pad
	ax.plot([lo, hi], [lo, hi], linestyle="--", color="tab:blue", linewidth=2.2, label="Equal performance")
	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_xlabel("Zero-filled SSIM")
	ax.set_ylabel("ResUNet + DC SSIM")
	ax.set_title("Sample-wise SSIM comparison — all planes and ratios")
	ax.grid(True, alpha=0.25)

	pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
	improved_rate = float(np.mean(y > x) * 100.0)
	combined_count = int(combined.shape[0])
	info_text = f"Pearson r = {pearson_r:.3f}\nImproved samples = {improved_rate:.1f}%\nn = {combined_count:,}"
	ax.text(
		0.03,
		0.97,
		info_text,
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=12,
		bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.35", alpha=0.95),
	)

	handles = [
		Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=ratio_colors.get(ratio, "tab:blue"), markeredgecolor="none", markersize=7, alpha=0.35, label=f"{int(round(ratio * 100))}% retained")
		for ratio in ratio_order
	]
	handles.append(Line2D([0], [0], linestyle="--", color="tab:blue", linewidth=2.2, label="Equal performance"))
	ax.legend(handles=handles, loc="lower right", frameon=True)
	fig.tight_layout()

	output_base = results_root / "sample_wise_ssim_comparison_all_planes_and_ratios"
	for fmt in ("png", "pdf"):
		fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Created root-level SSIM comparison scatter: {output_base.with_suffix('.png')} and {output_base.with_suffix('.pdf')}")
	return output_base.with_suffix(".png")

def _save_training_history(history: dict[str, list[float]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _clear_previous_results(result_dir: Path, retain_model) -> None:
	if result_dir.exists():
		if retain_model:
			for path in sorted(result_dir.rglob("*"), reverse=True):
				if path.is_file():
					if any(suffix.lower().startswith(".pt") for suffix in path.suffixes):
						continue
					path.unlink()
			print(f"Preserving model files and clearing older results in: {result_dir}")
		else:
			print(f"Removing older results directory: {result_dir}")
			shutil.rmtree(result_dir)

def _verify_required_files(required_paths: dict[str, Path]) -> list[str]:
	missing: list[str] = []
	for label, path in required_paths.items():
		if not path.exists():
			missing.append(f"{label} ({path})")
	return missing


def _is_relative_to(path: Path, parent: Path) -> bool:
	"""Compatibility helper for Path.is_relative_to across Python versions."""
	try:
		path.relative_to(parent)
		return True
	except ValueError:
		return False


def _validate_io_roots(source_root: Path, destination_root: Path) -> None:
	"""Ensure source exists and outputs do not write under the source dataset path."""
	if not source_root.exists() or not source_root.is_dir():
		raise FileNotFoundError(f"Source dataset root does not exist or is not a directory: {source_root}")

	# Any outputs are created under destination_root; prevent writing into source tree.
	if destination_root == source_root or _is_relative_to(destination_root, source_root):
		raise ValueError(
			"Destination root must be outside the source dataset root to avoid writing into source data. "
			f"source={source_root}, destination={destination_root}"
		)

def _resolve_evaluation_checkpoint_path(model_config: ModelConfig, model_path_root: str | Path | None) -> Path:
	"""Resolve the checkpoint to load for evaluation.

	If model_path_root is provided, expect the tree:
	<model_path_root>/<plane>/retain_<xx>/best_model.pt
	Otherwise use the experiment's own checkpoint.
	"""
	if model_path_root is None:
		return model_config.checkpoint_path

	model_path_text = str(model_path_root).strip()
	if model_path_text == "":
		return model_config.checkpoint_path

	root = Path(model_path_text).expanduser().resolve()
	checkpoint_path = root / model_config.experiment_relative_path / "best_model.pt"
	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"External evaluation checkpoint not found: {checkpoint_path}"
		)
	return checkpoint_path


def _resolve_volume_key(row: pd.Series, fallback_index: int) -> str:
	priority_columns = ("subject_id", "resolved_volume_path", "original_volume_path", "sample_id")
	for column in priority_columns:
		if column not in row.index:
			continue
		value = row[column]
		if value is None:
			continue
		if isinstance(value, float) and np.isnan(value):
			continue
		text = str(value).strip()
		if text == "" or text.lower() == "nan":
			continue
		return f"{column}:{text}"
	return f"sample_{fallback_index:06d}"


def _resolve_slice_key(row: pd.Series, fallback_index: int) -> str:
	if "slice_index" in row.index and not pd.isna(row["slice_index"]):
		return f"slice_{int(row['slice_index'])}"
	return f"unknown_slice_{fallback_index:06d}"


def _select_comparison_indices(
	dataset: MRIUndersampledDataset,
	volumes_to_sample: int,
	slices_per_volume: int,
	samplings_per_slice: int,
	random_seed: int,
) -> list[int]:
	"""Randomly sample indices by volume->slice hierarchy for figure generation."""
	if volumes_to_sample <= 0 or slices_per_volume <= 0 or samplings_per_slice <= 0:
		return []
	df = dataset.samples.reset_index(drop=True).copy()
	df["_dataset_index"] = df.index.astype(int)
	df["_volume_key"] = [
		_resolve_volume_key(row, fallback_index=int(idx))
		for idx, row in df.iterrows()
	]
	df["_slice_key"] = [
		_resolve_slice_key(row, fallback_index=int(idx))
		for idx, row in df.iterrows()
	]
	rng = np.random.default_rng(random_seed)
	volume_keys = df["_volume_key"].dropna().unique().tolist()
	if not volume_keys:
		return list(range(min(volumes_to_sample * slices_per_volume * samplings_per_slice, len(dataset))))
	rng.shuffle(volume_keys)
	selected_volume_keys = volume_keys[: min(volumes_to_sample, len(volume_keys))]
	selected_indices: list[int] = []
	for volume_key in selected_volume_keys:
		volume_df = df[df["_volume_key"] == volume_key]
		slice_keys = volume_df["_slice_key"].dropna().unique().tolist()
		rng.shuffle(slice_keys)
		for slice_key in slice_keys[: min(slices_per_volume, len(slice_keys))]:
			slice_df = volume_df[volume_df["_slice_key"] == slice_key]
			candidate_indices = slice_df["_dataset_index"].astype(int).to_numpy(copy=True)
			rng.shuffle(candidate_indices)
			for dataset_index in candidate_indices[: min(samplings_per_slice, len(candidate_indices))]:
				selected_indices.append(int(dataset_index))
	target = volumes_to_sample * slices_per_volume * samplings_per_slice
	if len(selected_indices) < target:
		print(
			"Warning: requested "
			f"{target} comparison figures but only {len(selected_indices)} could be sampled "
			"from available volume/slice combinations."
		)
	return selected_indices


def _save_comparison_figures(
	dataset: MRIUndersampledDataset,
	model: ResidualUNet,
	output_dir: Path,
	volumes_to_sample: int = 20,
	slices_per_volume: int = 2,
	samplings_per_slice: int = 1,
	random_seed: int = 42,
) -> list[Path]:
	"""Save comparison figures with random volume/slice sampling diversity."""
	output_dir.mkdir(parents=True, exist_ok=True)
	device = choose_device()
	selected_indices = _select_comparison_indices(
		dataset,
		volumes_to_sample=volumes_to_sample,
		slices_per_volume=slices_per_volume,
		samplings_per_slice=samplings_per_slice,
		random_seed=random_seed,
	)
	saved_paths: list[Path] = []

	for figure_index, sample_index in enumerate(selected_indices, start=1):
		sample = dataset[sample_index]
		undersampled = cast(Tensor, sample["input"])[0].numpy()
		target = cast(Tensor, sample["target"])[0].numpy()
		k_space_path = dataset.split_root / str(sample["k_space_file"])
		mask_path = dataset.split_root / str(sample["mask_file"])
		acquired_k_space = np.load(k_space_path, allow_pickle=False)
		row_mask = np.load(mask_path, allow_pickle=False)

		resunet = predict_numpy(model, undersampled, device=device)
		final_dc = predict_numpy_with_data_consistency(
			model=model,
			image=undersampled,
			acquired_k_space=acquired_k_space,
			row_mask=row_mask,
			device=device,
		)

		panels = [
			("Target", target, kspace_log_magnitude(image_to_kspace(target))),
			("Undersampled", undersampled, kspace_log_magnitude(acquired_k_space)),
			("ResUNet", resunet, kspace_log_magnitude(image_to_kspace(resunet))),
			("ResUNet + DC", final_dc, kspace_log_magnitude(image_to_kspace(final_dc))),
		]
		fig, axes = plt.subplots(2, 4, figsize=(14, 7))
		for panel_index, (title, image, k_space_image) in enumerate(panels):
			if np.allclose(target, image):
				psnr, ssim = float("inf"), 1.0
			else:
				psnr = calculate_psnr(target, image)
				ssim = calculate_ssim(target, image)
			axes[0, panel_index].imshow(image, cmap="gray")
			axes[0, panel_index].set_title(f"{title}\nPSNR={psnr:.2f} SSIM={ssim:.3f}")
			axes[0, panel_index].axis("off")
			axes[1, panel_index].imshow(k_space_image, cmap="magma")
			axes[1, panel_index].set_title(f"{title} k-space")
			axes[1, panel_index].axis("off")

		subject = str(sample.get("subject_id", "unknown"))
		plane = str(sample.get("plane", "unknown"))
		slice_index = sample.get("slice_index", "unknown")
		retain_ratio = float(cast(Tensor, sample["retain_ratio"]).item())
		fig.suptitle(
			f"volume={subject} | plane={plane} | slice={slice_index} | retain_ratio={retain_ratio:.2f}",
			fontsize=12,
		)
		fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

		sample_id = str(sample.get("sample_id", f"sample_{sample_index:04d}"))
		safe_sample_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in sample_id)
		output_path = output_dir / f"comparison_{figure_index:03d}_{safe_sample_id}.png"
		fig.savefig(output_path, dpi=150)
		plt.close(fig)
		saved_paths.append(output_path)

	return saved_paths



def _evaluate(model_config: ModelConfig, dataset_plan: DatasetCreationPlan, model_path_root: str | Path | None = None) -> int:
	"""Evaluate test split; return number of test samples."""
	phase_start = time.perf_counter()
	phase_started_at = datetime.now()
	print(f"[{phase_started_at:%Y-%m-%d %H:%M:%S}] Evaluation phase started")
	print("Evaluate sequence start")
	print("loading test data sequence")
	test_dataset = MRIUndersampledDataset(
		model_config.test_data_root,
		csv_name=dataset_plan.csv_file_name,
		plane=model_config.plane.capitalize(),
		retain_ratio=model_config.retain_ratio,
		load_mask=True)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
	checkpoint_path = _resolve_evaluation_checkpoint_path(model_config, model_path_root)
	print(f"Loading evaluation model from: {checkpoint_path}")
	print("Evaluating model...")
	evaluation_start = time.perf_counter()
	try:
		reloaded_model, checkpoint = load_checkpoint(checkpoint_path, ResidualUNet)
	except Exception as exc:
		raise RuntimeError(f"Checkpoint loading failed for {checkpoint_path}: {exc}") from exc
	try:
		_ = evaluate_and_save_results(
			model=reloaded_model,
			test_loader=test_loader,
			output_root=model_config.result_root,
			plane=model_config.plane,
			retain_ratio=model_config.retain_ratio,
			checkpoint_name=model_config.checkpoint_path.name,
			epoch=int(checkpoint["epoch"]),
		)
	except ValueError as exc:
		if "Shape mismatch" in str(exc):
			raise ValueError(f"Tensor shapes do not match during evaluation: {exc}") from exc
		raise
	evaluation_finished_at = datetime.now()
	print(
		f"[{evaluation_finished_at:%Y-%m-%d %H:%M:%S}] Evaluation inference and per-image metrics finished in "
		f"{_format_duration(time.perf_counter() - evaluation_start)}"
	)

	csv_path = model_config.test_results_path
	figures_dir = model_config.result_dir / "comparison_figures"
	saved_figure_paths = _save_comparison_figures(
		test_dataset,
		reloaded_model,
		figures_dir,
		volumes_to_sample=20,
		slices_per_volume=2,
		samplings_per_slice=1,
		random_seed=int(model_config.random_seed),
	)
	if not saved_figure_paths:
		raise FileNotFoundError(f"No comparison figures were created in: {figures_dir}")

	required = {
		"checkpoint": checkpoint_path,
		"training_history": model_config.history_path,
		"config_used": model_config.result_dir / "config_used.json",
		"test_per_image_csv": csv_path,
		"comparison_figures_dir": figures_dir,
	}
	missing = _verify_required_files(required)
	if missing:
		raise FileNotFoundError(f"Result files were not created: {missing}")

	report_start = time.perf_counter()
	report_started_at = datetime.now()
	print(f"[{report_started_at:%Y-%m-%d %H:%M:%S}] Evaluation results generation started")
	generate_all_evaluation_outputs(
		sample_metrics_csv=csv_path,
		output_dir=model_config.result_dir / "report_outputs",
		baseline_method="resunet",
		proposed_method="resunet_data_consistency",
	)
	report_finished_at = datetime.now()
	print(
		f"[{report_finished_at:%Y-%m-%d %H:%M:%S}] Evaluation results generation finished in "
		f"{_format_duration(time.perf_counter() - report_start)}"
	)
	phase_finished_at = datetime.now()
	print(
		f"[{phase_finished_at:%Y-%m-%d %H:%M:%S}] Evaluation phase finished in "
		f"{_format_duration(time.perf_counter() - phase_start)}"
	)

	return len(test_dataset)

def _create_and_train_model(model_config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader) -> None:
	phase_start = time.perf_counter()
	phase_started_at = datetime.now()
	print(f"[{phase_started_at:%Y-%m-%d %H:%M:%S}] Training phase started")
	model_config.save()  # persist config_used.json before training starts
	model_kwargs = model_config.model_kwargs
	model = ResidualUNet(**model_kwargs)
	train_config = model_config.make_train_config()
	history = train_model(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		train_config=train_config,
		model_kwargs=model_kwargs,
		experiment_config=model_config,
	)
	if not model_config.checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint saving failed, file not found: {model_config.checkpoint_path}")
	_save_training_history(history, model_config.history_path)
	phase_finished_at = datetime.now()
	print(
		f"[{phase_finished_at:%Y-%m-%d %H:%M:%S}] Training phase finished in "
		f"{_format_duration(time.perf_counter() - phase_start)}"
	)

def _create_train_data_loaders(params: dict, model_config: ModelConfig, dataset_plan: DatasetCreationPlan) -> tuple[DataLoader, DataLoader, int, int]:
	"""Return (train_loader, val_loader, train_count, val_count)."""
	data_sets_root = model_config.data_root
	train_dataset = MRIUndersampledDataset(
		data_sets_root / "train",
		csv_name=dataset_plan.csv_file_name,
		plane=model_config.plane.capitalize(),
		retain_ratio=model_config.retain_ratio,
		load_mask=False)
	val_dataset = MRIUndersampledDataset(
		data_sets_root / "val",
		csv_name=dataset_plan.csv_file_name,
		plane=model_config.plane.capitalize(),
		retain_ratio=model_config.retain_ratio,
		load_mask=False)

	train_loader = DataLoader(
		train_dataset,
		batch_size=model_config.batch_size,
		shuffle=True,
		num_workers=model_config.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=model_config.batch_size,
		shuffle=False,
		num_workers=model_config.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	return train_loader, val_loader, len(train_dataset), len(val_dataset)

def _create_model_config(params, plane, retain_ratio) -> ModelConfig:
	result = ModelConfig(
		plane=plane,
		retain_ratio=float(retain_ratio),
		epochs=params["epochs"],
		batch_size=32,
		learning_rate=5e-5,
		weight_decay=1e-4,
		random_seed=42,
		num_workers=0 if os.name == "nt" else 4,
		data_consistency_enabled=True,
		per_image_csv_logging=True,
		data_root=Path(str(params["dataset_split_root"])).resolve(),
		result_root=Path(str(params["results_root"])).resolve(),
	)
	print("Model config created:")
	print("===============================")
	pprint(asdict(result))
	print("===============================")
	return result

def _create_undersampled_data_set(params:dict) -> DatasetCreationPlan:
	phase_start = time.perf_counter()
	phase_started_at = datetime.now()
	print(f"[{phase_started_at:%Y-%m-%d %H:%M:%S}] Dataset split phase started")
	data_creation_plan = DatasetCreationPlan(
		source_dataset_root=Path(str(params["source_dataset_root"])).resolve(),
		source_csv_names=SCV_FILES,
		output_dataset_root=params["dataset_split_root"],
		split_multiplicity={
			"train": params["train_set_size"],
			"val": params["val_set_size"],
			"test": params["test_set_size"],
		},
		slice_percentile_range=(35.0, 65.0),
		planes=params["selected_planes"],
		retain_ratios=params["retain_ratios"],
		first_seed=1234,
		sigma_fraction=1 / 6,
	)
	created_split = False
	if params["skip_split_creation"]:
		print("Skip splitting creation")
	else:
		print("Creating undersampled data set:")
		print("===============================")
		pprint(asdict(data_creation_plan))
		print("===============================")
		create_dataset_split(data_creation_plan)
		created_split = True
		print(f"Data set created at: {data_creation_plan.output_dataset_root}")
	phase_finished_at = datetime.now()
	phase_duration = _format_duration(time.perf_counter() - phase_start)
	if created_split:
		print(f"[{phase_finished_at:%Y-%m-%d %H:%M:%S}] Dataset split phase finished in {phase_duration}")
	else:
		print(f"[{phase_finished_at:%Y-%m-%d %H:%M:%S}] Dataset split phase skipped in {phase_duration}")
	return data_creation_plan


def _run_sequence(params: dict) -> int:
	sequence_start = time.perf_counter()
	sequence_started_at = datetime.now()
	print(f"[{sequence_started_at:%Y-%m-%d %H:%M:%S}] Experiment sequence started")
	dataset_plan = _create_undersampled_data_set(params)
	all_passed = True
	for plane in params["selected_planes"]:
		for retain_ratio in params["retain_ratios"]:
			print(f"\nExecuting undersampled experiment: plane={plane}, retain_ratio={retain_ratio}")
			model_config = _create_model_config(params, plane, retain_ratio)
			_clear_previous_results(model_config.result_dir, params["skip_train"])
			train_loader, val_loader, train_count, val_count = _create_train_data_loaders(params, model_config, dataset_plan)
			status = "PASS"
			test_count = 0
			try:
				if not params["skip_train"]:
					_create_and_train_model(model_config, train_loader, val_loader)
				if not params["skip_evaluation"]:
						test_count = _evaluate(model_config, dataset_plan, model_path_root=params.get("model_path"))
			except Exception as exc:
				status = f"FAIL: {exc}"
				all_passed = False

			print("\n" + "=" * 60)
			print(f"Status summary  plane={plane}  retain_ratio={retain_ratio}")
			print(f"  train samples : {train_count}")
			print(f"  val samples   : {val_count}")
			print(f"  test samples  : {test_count}")
			print(f"  checkpoint    : {model_config.checkpoint_path}")
			print(f"  csv           : {model_config.test_results_path}")
			print(f"  figures dir   : {model_config.result_dir / 'comparison_figures'}")
			print(f"  config        : {model_config.result_dir / 'config_used.json'}")
			print(f"  result        : {status}")
			print("=" * 60)

	results_root = Path(str(params["results_root"])).resolve()
	selected_planes = [str(plane).lower() for plane in params["selected_planes"]]
	retain_ratios = [float(ratio) for ratio in params["retain_ratios"]]
	aggregated_csv_path = _aggregate_all_planes_metrics(results_root, selected_planes, retain_ratios)
	_create_root_ratio_plots(results_root, aggregated_csv_path)
	_create_root_psnr_scatter(results_root, selected_planes, retain_ratios)
	_create_root_ssim_scatter(results_root, selected_planes, retain_ratios)
	sequence_finished_at = datetime.now()
	print(
		f"[{sequence_finished_at:%Y-%m-%d %H:%M:%S}] Experiment sequence finished in "
		f"{_format_duration(time.perf_counter() - sequence_start)}"
	)

	return 0 if all_passed else 1

def _create_parameters_for_mode(args) -> dict[str, object]:
	source_root = Path(str(args.source_data_root)).expanduser().resolve()
	destination_root = Path(str(args.destination_root)).expanduser().resolve()
	_validate_io_roots(source_root, destination_root)
	destination_root.mkdir(parents=True, exist_ok=True)

	if args.mode == "main_experiment":
		result = {
			"mode": args.mode,
			"source_dataset_root": source_root,
			"destination_root": destination_root,
			"selected_planes": BRAIN_PLANES,
			"retain_ratios": RETAIN_RATIOS,
			"dataset_split_root": (destination_root / "undersampled_dataset_split").resolve(),
			"results_root": (destination_root / "undersampled_results").resolve(),
			"train_set_size": SplitMultiplicityConfig(
				number_of_volumes=800,
				slices_per_volume_per_plane=4,
				undersampling_per_slice = 3,
			),
			"val_set_size": SplitMultiplicityConfig(
				number_of_volumes=80,
				slices_per_volume_per_plane=2,
				undersampling_per_slice = 2
			),
			"test_set_size": SplitMultiplicityConfig(
				number_of_volumes=300,
				slices_per_volume_per_plane=2,
				undersampling_per_slice = 2
			),
			"epochs": 100,
		}
	elif args.mode == "smoke":
		result = {
			"mode": args.mode,
			"source_dataset_root": source_root,
			"destination_root": destination_root,
			"selected_planes": BRAIN_PLANES,
			"retain_ratios": RETAIN_RATIOS,
			"dataset_split_root": (destination_root / "smoke_dataset_split").resolve(),
			"results_root": (destination_root / "smoke_results").resolve(),
			"train_set_size": SplitMultiplicityConfig(
				number_of_volumes=4,
				slices_per_volume_per_plane=2,
				undersampling_per_slice=2,
			),
			"val_set_size": SplitMultiplicityConfig(
				number_of_volumes=4,
				slices_per_volume_per_plane=2,
				undersampling_per_slice=2
			),
			"test_set_size": SplitMultiplicityConfig(
				number_of_volumes=8,
				slices_per_volume_per_plane=2,
				undersampling_per_slice=2
			),
			"epochs": 4,
		}
	else:
		raise ValueError(f"Unknown mode: {args.mode}")
	result["skip_split_creation"] = args.skip_split_creation
	result["skip_train"] = args.skip_train
	result["skip_evaluation"] = args.skip_evaluation
	result["model_path"] = args.model_path
	return result

def main() -> int:
	app_start = time.perf_counter()
	app_started_at = datetime.now()
	print(f"[{app_started_at:%Y-%m-%d %H:%M:%S}] Experiment app started")
	parser = argparse.ArgumentParser(description="MRI reconstruction command line runner")
	parser.add_argument("--mode", choices=("smoke", "main_experiment"), default="smoke", help="Run smoke test or the main experiment run.")
	parser.add_argument("--source_data_root", required=True, help="Existing source dataset root. Used only for reading.")
	parser.add_argument("--destination_root", required=True, help="Destination parent root for generated dataset splits and results.")
	parser.add_argument("--skip_split_creation", action="store_true", help="Skip the creation of dataset splits.")
	parser.add_argument("--skip_train", action="store_true", help="Skip the the train.")
	parser.add_argument("--skip_evaluation", action="store_true", help="Skip the the evaluation on the test set.")
	parser.add_argument("--model_path", help="Path to the model checkpoint. If none - use from default location")

	args = parser.parse_args()

	params = _create_parameters_for_mode(args)
	print("Experiment Parameters:")
	print("===============================")
	pprint(params)
	print("===============================")

	try:
		return _run_sequence(params)
	finally:
		app_finished_at = datetime.now()
		print(
			f"[{app_finished_at:%Y-%m-%d %H:%M:%S}] Experiment app finished in "
			f"{_format_duration(time.perf_counter() - app_start)}"
		)


if __name__ == "__main__":
	raise SystemExit(main())
