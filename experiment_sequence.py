from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

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
from src.evaluation.report_outputs import generate_all_evaluation_outputs

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


def _select_distinct_volume_indices(dataset: MRIUndersampledDataset, max_items: int = 10) -> list[int]:
	"""Pick up to max_items indices, preferring one sample per distinct volume."""
	priority_columns = ("subject_id", "resolved_volume_path", "original_volume_path", "sample_id")
	available_columns = [column for column in priority_columns if column in dataset.samples.columns]
	selected: list[int] = []
	seen_keys: set[str] = set()

	for idx, row in dataset.samples.iterrows():
		key_parts: list[str] = []
		for column in available_columns:
			value = row[column]
			if value is None:
				continue
			if isinstance(value, float) and np.isnan(value):
				continue
			text = str(value).strip()
			if text == "" or text.lower() == "nan":
				continue
			key_parts.append(text)
		volume_key = "|".join(key_parts) if key_parts else str(row.get("sample_id", idx))
		if volume_key in seen_keys:
			continue
		seen_keys.add(volume_key)
		selected.append(int(idx))
		if len(selected) >= max_items:
			break

	if selected:
		return selected
	return list(range(min(max_items, len(dataset))))


def _save_comparison_figures(dataset: MRIUndersampledDataset, model: ResidualUNet, output_dir: Path, max_images: int = 10) -> list[Path]:
	"""Save comparison figures for up to max_images from different volumes."""
	output_dir.mkdir(parents=True, exist_ok=True)
	device = choose_device()
	selected_indices = _select_distinct_volume_indices(dataset, max_items=max_images)
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
		output_path = output_dir / f"comparison_{figure_index:02d}_{safe_sample_id}.png"
		fig.savefig(output_path, dpi=150)
		plt.close(fig)
		saved_paths.append(output_path)

	return saved_paths



def _evaluate(model_config: ModelConfig, dataset_plan: DatasetCreationPlan, model_path_root: str | Path | None = None) -> int:
	"""Evaluate test split; return number of test samples."""
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

	csv_path = model_config.test_results_path
	figures_dir = model_config.result_dir / "comparison_figures"
	saved_figure_paths = _save_comparison_figures(test_dataset, reloaded_model, figures_dir, max_images=10)
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

	generate_all_evaluation_outputs(
		sample_metrics_csv=csv_path,
		output_dir=model_config.result_dir / "report_outputs",
		baseline_method="resunet",
		proposed_method="resunet_data_consistency",
	)

	return len(test_dataset)

def _create_and_train_model(model_config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader) -> None:
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
	data_creation_plan = DatasetCreationPlan(
		source_dataset_root=Path(os.getcwd()),
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
	if params["skip_split_creation"]:
		print("Skip splitting creation")
		return data_creation_plan
	print("Creating undersampled data set:")
	print("===============================")
	pprint(asdict(data_creation_plan))
	print("===============================")
	create_dataset_split(data_creation_plan)
	print(f"Data set created at: {data_creation_plan.output_dataset_root}")
	return data_creation_plan


def _run_sequence(params: dict) -> int:
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

	return 0 if all_passed else 1

def _create_parameters_for_mode(args) -> dict[str, object]:
	cwd = Path(os.getcwd())
	if args.mode == "main_experiment":
		result = {
			"mode": args.mode,
			"selected_planes": BRAIN_PLANES,
			"retain_ratios": RETAIN_RATIOS,
			"dataset_split_root": (cwd / ".." / "undersampled_dataset_split").resolve(),
			"results_root": (cwd / ".." / "undersampled_results").resolve(),
			"train_set_size": SplitMultiplicityConfig(
				number_of_volumes=700,
				slices_per_volume_per_plane=4,
				undersampling_per_slice = 3,
			),
			"val_set_size": SplitMultiplicityConfig(
				number_of_volumes=70,
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
			"selected_planes": BRAIN_PLANES,
			"retain_ratios": RETAIN_RATIOS,
			"dataset_split_root": (cwd / ".." / "smoke_dataset_split").resolve(),
			"results_root": (cwd / ".." / "smoke_results").resolve(),
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
	parser = argparse.ArgumentParser(description="MRI reconstruction command line runner")
	parser.add_argument("--mode", choices=("smoke", "main_experiment"), default="smoke", help="Run smoke test or the main experiment run.")
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

	return _run_sequence(params)


if __name__ == "__main__":
	raise SystemExit(main())
