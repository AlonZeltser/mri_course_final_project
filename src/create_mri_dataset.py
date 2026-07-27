from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data_utils import (
    csv_path_to_local_path,
    extract_slice,
    load_metadata,
    load_volume,
    min_max_normalize,
)
from src.general_utils import BRAIN_PLANES, DL_SPLITS, SCV_FILES
from src.k_space_utils import image_to_kspace, kspace_to_image

VALID_METADATA_FILE_NAME = "valid_metadata.csv"
SPLIT_VOLUME_ASSIGNMENT = "split_volume_assignment.csv"

@dataclass(frozen=True)
class SplitMultiplicityConfig:
    number_of_volumes: int
    slices_per_volume_per_plane: int
    undersampling_per_slice: int

@dataclass(frozen=True)
class DatasetCreationPlan:
    """Single-call plan that generates train/val/test from one valid pool."""
    source_dataset_root: Path
    source_csv_names: Mapping[str, str]
    output_dataset_root: Path
    split_multiplicity: Mapping[str, SplitMultiplicityConfig]

    slice_percentile_range: tuple[float, float]
    planes: tuple[str, ...]
    retain_ratios: tuple[float, ...]
    first_seed: int
    sigma_fraction: float
    path_column: str = "filePath"
    subject_column: str = "Subject"
    output_dtype: str = "float32"
    overwrite: bool = False
    csv_file_name: str = "samples.csv"


def validate_creation_plan(plan: DatasetCreationPlan) -> None:
    if not plan.output_dataset_root:
        raise ValueError("output_dataset_root must be set.")

    low, high = plan.slice_percentile_range
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(
            "slice_percentile_range must satisfy 0 <= low < high <= 100."
        )

    if plan.sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be positive.")

    unknown_planes = set(plan.planes) - set(BRAIN_PLANES)
    if unknown_planes:
        raise ValueError(
            f"Unknown planes: {sorted(unknown_planes)}. "
            f"Supported planes: {sorted(BRAIN_PLANES)}."
        )

    for ratio in plan.retain_ratios:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"Each retain ratio must be in (0, 1], received {ratio}."
            )

    missing_split_configs = set(DL_SPLITS) - set(plan.split_multiplicity)
    if missing_split_configs:
        raise ValueError(
            "split_multiplicity is missing: "
            f"{sorted(missing_split_configs)}"
        )

    missing_csv_names = set(SCV_FILES) - set(plan.source_csv_names)
    if missing_csv_names:
        raise ValueError(
            "source_csv_names is missing: "
            f"{sorted(missing_csv_names)}"
        )

    for split_name in DL_SPLITS:
        split_config = plan.split_multiplicity[split_name]
        if split_config.number_of_volumes <= 0:
            raise ValueError(
                f"number_of_volumes for '{split_name}' must be positive."
            )
        if split_config.undersampling_per_slice <= 0:
            raise ValueError(
                f"undersampling_per_slice for '{split_name}' must be positive."
            )

        if split_config.slices_per_volume_per_plane <= 0:
            raise ValueError(f"slices per volume per plane must be positive")

def candidate_slice_indices(
    axis_length: int,
    percentile_range: tuple[float, float],
) -> np.ndarray:
    """
    Return indices inside a percentile interval along an axis.

    Example:
        (25, 75) selects indices from the central 50% of the volume.
    """
    low_percentile, high_percentile = percentile_range

    low = int(np.ceil((low_percentile / 100.0) * axis_length))
    high_exclusive = int(
        np.floor((high_percentile / 100.0) * axis_length)
    )

    low = max(0, min(low, axis_length - 1))
    high_exclusive = max(low + 1, min(high_exclusive, axis_length))

    return np.arange(low, high_exclusive, dtype=int)


def create_unique_row_mask(
    number_of_rows: int,
    retain_ratio: float,
    seed: int,
    sigma_fraction: float,
) -> np.ndarray:
    """
    Create a unique 1D binary k-space row mask.

    A value of 1 means that the row is retained.
    A value of 0 means that the row is removed.

    The removal distribution matches zero_k_space_rows_random_dist():
    outer rows are more likely to be removed than central rows.
    """
    if number_of_rows <= 0:
        raise ValueError("number_of_rows must be positive.")

    rows_to_retain = int(round(number_of_rows * retain_ratio))
    rows_to_retain = min(number_of_rows, max(1, rows_to_retain))
    rows_to_remove = number_of_rows - rows_to_retain

    mask = np.ones(number_of_rows, dtype=np.uint8)
    if rows_to_remove == 0:
        return mask

    row_indices = np.arange(number_of_rows)
    center = (number_of_rows - 1) / 2.0
    sigma = number_of_rows * sigma_fraction

    preservation_probability = np.exp(
        -0.5 * ((row_indices - center) / sigma) ** 2
    )
    removal_probability = 1.0 - preservation_probability
    removal_probability += np.finfo(float).eps
    removal_probability /= removal_probability.sum()

    rng = np.random.default_rng(seed)
    removed_rows = rng.choice(
        row_indices,
        size=rows_to_remove,
        replace=False,
        p=removal_probability,
    )
    mask[removed_rows] = 0

    if int(mask.sum()) != rows_to_retain:
        raise AssertionError("Generated mask has the wrong retained-row count.")

    return mask


def apply_row_mask(
    k_space: np.ndarray,
    row_mask: np.ndarray,
) -> np.ndarray:
    if k_space.ndim != 2:
        raise ValueError("k_space must be 2D.")
    if row_mask.ndim != 1:
        raise ValueError("row_mask must be 1D.")
    if row_mask.shape[0] != k_space.shape[0]:
        raise ValueError(
            "row_mask length must equal the number of k-space rows."
        )
    if not np.all(np.isin(row_mask, (0, 1))):
        raise ValueError("row_mask must contain only 0 and 1.")

    return k_space * row_mask[:, np.newaxis]


def ratio_folder_name(retain_ratio: float) -> str:
    percentage = int(round(retain_ratio * 100))
    return f"retain_{percentage:02d}"


def safe_token(value: object) -> str:
    text = str(value).strip()
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in text
    )
    return safe or "unknown"


def prepare_output_directories(
    split_root: Path,
    retain_ratios: Sequence[float]
) -> None:
    print("preparing output directories...")
    # clear previous
    if split_root.exists():
        print(f"{split_root} already exists. Removing...")
        try:
            shutil.rmtree(split_root)
        except Exception as e:
            print("Failed to delete existing output directories.")
            raise
        print(f"{split_root} removed")
        assert not split_root.exists()

    print("creating output directories...")
    split_root.mkdir(parents=False, exist_ok=False)
    print(f"{split_root} created")

    originals = (split_root / "originals")
    originals.mkdir(parents=False, exist_ok=False)
    print(f"{originals} created")

    us = (split_root / "undersampled")
    us.mkdir(parents=False, exist_ok=False)
    print(f"{us} created")
    k_spaces = (split_root / "k_spaces")
    k_spaces.mkdir(parents=False, exist_ok=False)
    print(f"{k_spaces} created")
    masks = (split_root / "masks")
    masks.mkdir(parents=False, exist_ok=False)
    print(f"{masks} created")

    for retain_ratio in retain_ratios:
        ratio_folder = ratio_folder_name(retain_ratio)

        us_ratio = (us / ratio_folder)
        us_ratio.mkdir(parents=False, exist_ok=False)
        print(f"{us_ratio} created")

        k_spaces_ratio = (k_spaces / ratio_folder)
        k_spaces_ratio.mkdir(parents=False, exist_ok=False)
        print(f"{k_spaces_ratio} created")

        masks_ratio = (masks / ratio_folder)
        masks_ratio.mkdir(parents=False, exist_ok=False)
        print(f"{masks_ratio} created")



def collect_valid_metadata_pool(plan: DatasetCreationPlan) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    repeating_paths = 0
    skipped_not_found = 0
    skipped_unreadable = 0

    for source_split in DL_SPLITS:
        if source_split in plan.source_csv_names.keys():
            source_csv_path = (
                plan.source_dataset_root / plan.source_csv_names[source_split]
            )
            print(f"collecting metadata for {source_split}")
            metadata = load_metadata(source_csv_path)
            if plan.path_column not in metadata.columns:
                raise KeyError(
                    f"CSV {source_csv_path} does not contain path column "
                    f"'{plan.path_column}'."
                )

            for csv_row_index, row in metadata.iterrows():
                resolved_volume_path = Path(csv_path_to_local_path(row[plan.path_column]))
                if not resolved_volume_path.is_absolute():
                    resolved_volume_path = plan.source_dataset_root / resolved_volume_path
                resolved_volume_path = resolved_volume_path.resolve()
                if resolved_volume_path in seen_paths:
                    repeating_paths += 1
                    continue

                if not resolved_volume_path.exists():
                    skipped_not_found += 1
                    continue

                volume = load_volume(resolved_volume_path)
                if volume is None:
                    skipped_unreadable += 1
                    continue

                seen_paths.add(resolved_volume_path)
                record = row.to_dict()
                record["source_metadata_split"] = source_split
                record["source_metadata_row"] = int(str(csv_row_index))
                record["resolved_volume_path"] = str(resolved_volume_path)
                records.append(record)

    if not records:
        raise RuntimeError(
            "Could not find any existing volume files from the provided metadata CSVs."
        )

    valid_pool = pd.DataFrame.from_records(records)
    if len(valid_pool) > len(seen_paths):
        valid_pool = valid_pool.drop_duplicates(
            subset=["resolved_volume_path"]
        ).reset_index(drop=True)

    print(
        f"Collected valid metadata pool: {len(valid_pool)} volumes. \n"
        f"Skipped not-found={skipped_not_found}, unreadable={skipped_unreadable}.\n"
        f"Repeating paths={repeating_paths}\n"
        f"unreadable={skipped_unreadable}\n"
    )
    return valid_pool


def sample_split_volume_rows(
    valid_pool: pd.DataFrame,
    plan: DatasetCreationPlan,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    counts = {
        split_name: int(plan.split_multiplicity[split_name].number_of_volumes)
        for split_name in DL_SPLITS
    }
    requested_total = sum(counts.values())
    if requested_total > len(valid_pool):
        raise RuntimeError(
            f"Requested {requested_total} volumes across splits, but only "
            f"{len(valid_pool)} valid unique volumes are available."
        )
    # random list of indices
    shuffled_indices = rng.permutation(len(valid_pool))
    sampled_rows: dict[str, pd.DataFrame] = {}
    cursor = 0
    for split_name in DL_SPLITS:
        print(f"sampling items for {split_name}")
        count = counts[split_name]
        # next sub list of reandome indices
        split_indices = shuffled_indices[cursor:cursor + count]
        #take the item: copy, re index
        sampled_rows[split_name] = valid_pool.iloc[split_indices].copy().reset_index(drop=True)
        cursor += count

    return sampled_rows


def _create_split_from_rows(
    split_name: str,
    selected_rows: pd.DataFrame,
    plan: DatasetCreationPlan,
    selection_rng: np.random.Generator,
    next_mask_seed: int,
    csv_file_name: str,
) -> tuple[pd.DataFrame, int]:
    split_root = plan.output_dataset_root / split_name
    split_config = plan.split_multiplicity[split_name]
    output_dtype = np.dtype(plan.output_dtype)
    csv_records: list[dict[str, object]] = []
    sample_counter = 0
    original_counter = 0

    for selected_number, (_, row) in enumerate(selected_rows.iterrows(), start=1):
        volume_path = Path(str(row["resolved_volume_path"]))
        volume = load_volume(volume_path)
        if volume is None:
            # Path validity is checked during pool creation, but keep runtime robust.
            continue

        subject_value = (
            row[plan.subject_column]
            if plan.subject_column in selected_rows.columns
            else volume_path.stem
        )
        subject_id = safe_token(subject_value)
        source_row = int(row.get("source_metadata_row", selected_number))
        volume_token = f"{subject_id}_row{source_row:06d}"

        print(
            f"[{split_name} {selected_number}/{len(selected_rows)}] "
            f"{volume_path}, shape={volume.shape}"
        )

        for plane in plan.planes:
            axis = BRAIN_PLANES[plane]
            candidates = candidate_slice_indices(
                volume.shape[axis],
                plan.slice_percentile_range,
            )
            slice_count = split_config.slices_per_volume_per_plane
            if slice_count > len(candidates):
                raise ValueError(
                    f"Requested {slice_count} {plane} slices from "
                    f"{volume_path.name}, but the configured percentile "
                    f"range contains only {len(candidates)} indices."
                )

            selected_slice_indices = selection_rng.choice(
                candidates,
                size=slice_count,
                replace=False,
            )
            selected_slice_indices.sort()

            for slice_index in selected_slice_indices:
                # Use shared volume slicing logic to keep orientation/axis handling consistent.
                _, raw_slice = extract_slice(
                    volume=volume,
                    axis=axis,
                    slice_index=int(slice_index),
                )
                normalized_slice = min_max_normalize(raw_slice).astype(
                    output_dtype,
                    copy=False,
                )

                plane_token = plane.lower()
                original_id = (
                    f"{volume_token}_{plane_token}_"
                    f"s{int(slice_index):04d}"
                )
                original_relative_path = Path("originals") / f"{original_id}.npy"
                np.save(
                    split_root / original_relative_path,
                    normalized_slice,
                    allow_pickle=False,
                )
                original_counter += 1

                original_k_space = image_to_kspace(normalized_slice)
                for retain_ratio in plan.retain_ratios:
                    ratio_folder = ratio_folder_name(retain_ratio)
                    retain_percentage = int(round(retain_ratio * 100))
                    for repetition_index in range(
                        split_config.undersampling_per_slice
                    ):
                        mask_seed = next_mask_seed
                        next_mask_seed += 1
                        row_mask = create_unique_row_mask(
                            number_of_rows=original_k_space.shape[0],
                            retain_ratio=retain_ratio,
                            seed=mask_seed,
                            sigma_fraction=plan.sigma_fraction,
                        )
                        undersampled_k_space = apply_row_mask(
                            original_k_space,
                            row_mask,
                        )
                        sample_id = (
                            f"{original_id}_r{retain_percentage:02d}_"
                            f"u{repetition_index:03d}"
                        )
                        k_space_relative_path = (
                            Path("k_spaces")
                            / ratio_folder
                            / f"{sample_id}.npy"
                        )
                        np.save(
                            split_root / k_space_relative_path,
                            undersampled_k_space,
                            allow_pickle=False,
                        )
                        undersampled_image = kspace_to_image(
                            undersampled_k_space
                        ).astype(output_dtype, copy=False)

                        mask_id = f"mask_{sample_id}"
                        mask_relative_path = (
                            Path("masks") / ratio_folder / f"{mask_id}.npy"
                        )
                        undersampled_relative_path = (
                            Path("undersampled")
                            / ratio_folder
                            / f"{sample_id}.npy"
                        )

                        np.save(
                            split_root / mask_relative_path,
                            row_mask,
                            allow_pickle=False,
                        )
                        np.save(
                            split_root / undersampled_relative_path,
                            undersampled_image,
                            allow_pickle=False,
                        )

                        csv_records.append(
                            {
                                "sample_id": sample_id,
                                "subject_id": str(subject_value),
                                "original_volume_path": str(row[plan.path_column]),
                                "resolved_volume_path": str(volume_path),
                                "plane": plane,
                                "slice_index": int(slice_index),
                                "original_image_file": str(
                                    original_relative_path.as_posix()
                                ),
                                "retain_ratio": float(retain_ratio),
                                "mask_id": mask_id,
                                "mask_file": str(mask_relative_path.as_posix()),
                                "k_space_file": str(k_space_relative_path.as_posix()),
                                "undersampled_image_file": str(
                                    undersampled_relative_path.as_posix()
                                ),
                                "source_metadata_split": str(
                                    row.get("source_metadata_split", "unknown")
                                ),
                                "source_shape": json.dumps(
                                    list(normalized_slice.shape)
                                ),
                            }
                        )
                        sample_counter += 1

    samples = pd.DataFrame.from_records(csv_records)
    csv_output_path = split_root / csv_file_name
    samples.to_csv(csv_output_path, index=False)
    print(
        f"Created split '{split_name}' at {split_root}\n"
        f"Volumes: {len(selected_rows)}\n"
        f"Original slices: {original_counter}\n"
        f"Undersampled samples: {sample_counter}\n"
        f"CSV: {csv_output_path}"
    )
    return samples, next_mask_seed


def create_dataset_split(
    config: DatasetCreationPlan,
) -> dict[str, pd.DataFrame]:

    validate_creation_plan(config)
    config.output_dataset_root.mkdir(parents=True, exist_ok=True)

    valid_pool: pd.DataFrame = collect_valid_metadata_pool(config)
    valid_pool_csv_path = config.output_dataset_root / VALID_METADATA_FILE_NAME
    valid_pool.to_csv(valid_pool_csv_path, index=False)
    print(f"Saved unified valid metadata CSV: {valid_pool_csv_path}")

    selection_rng = np.random.default_rng(config.first_seed)
    # split:rows-list
    sampled_rows_by_split = sample_split_volume_rows(
        valid_pool,
        config,
        selection_rng,
    )
    split_assignment_records: list[dict[str, object]] = []
    next_mask_seed = int(config.first_seed)
    outputs: dict[str, pd.DataFrame] = {}
    for split_name in DL_SPLITS:
        split_root = config.output_dataset_root / split_name
        prepare_output_directories(split_root, config.retain_ratios)

        sampled_rows = sampled_rows_by_split[split_name]
        for _, row in sampled_rows.iterrows():
            split_assignment_records.append(
                {
                    "split": split_name,
                    "original_volume_path": str(row[config.path_column]),
                    "resolved_volume_path": str(row["resolved_volume_path"]),
                    "source_metadata_split": str(
                        row.get("source_metadata_split", "unknown")
                    ),
                }
            )

        samples, next_mask_seed = _create_split_from_rows(
            split_name=split_name,
            selected_rows=sampled_rows,
            plan=config,
            selection_rng=selection_rng,
            next_mask_seed=next_mask_seed,
            csv_file_name=config.csv_file_name,
        )
        outputs[split_name] = samples

    assignment_csv_path = config.output_dataset_root / SPLIT_VOLUME_ASSIGNMENT
    pd.DataFrame.from_records(split_assignment_records).to_csv(
        assignment_csv_path,
        index=False,
    )
    print(f"Saved split assignment CSV: {assignment_csv_path}")

    return outputs
