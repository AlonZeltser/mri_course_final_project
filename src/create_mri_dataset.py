from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from typing import Mapping, Sequence
from pprint import pprint

import numpy as np
import pandas as pd

from src.general_utils import csvs
from src.data_utils import (
    csv_path_to_local_path,
    load_metadata,
    load_volume,
    min_max_normalize,
)
from src.general_utils import brain_planes
from src.k_space_utils import image_to_kspace, kspace_to_image

DEFAULT_RETAIN_RATIOS = (0.20, 0.30, 0.50)

@dataclass(frozen=True)
class DatasetCreationConfig:
    """Configuration for one train/validation/test dataset split."""

    split_name: str
    source_dataset_root: Path
    source_csv_name: str
    output_dataset_root: Path
    number_of_volumes: int
    slices_per_volume_per_plane: int | Mapping[str, int]
    slice_percentile_range: tuple[float, float]
    undersampling_per_slice: int

    planes: tuple[str, ...]
    retain_ratios: tuple[float, ...]
    first_seed: int
    sigma_fraction: float
    path_column: str = "filePath"
    subject_column: str = "Subject"
    output_dtype: str = "float32"
    overwrite: bool = False

def create_config(split_name: str,
                  number_of_volumes:int,
                  slices_per_volume_per_plane: int,
                  slice_percentile_range: tuple[float, float],
                  undersampling_per_slice:int,
                  first_seed: int,
                  sigma_fraction: float,
                  output_dataset_root:str) -> DatasetCreationConfig:
    return DatasetCreationConfig(
    split_name=split_name,
    source_dataset_root=Path("."),
    source_csv_name=csvs[split_name],
    output_dataset_root=Path(output_dataset_root),
    number_of_volumes=number_of_volumes,
    slices_per_volume_per_plane=slices_per_volume_per_plane,
    slice_percentile_range=slice_percentile_range,
    undersampling_per_slice=undersampling_per_slice,
    planes=tuple(brain_planes.keys()),
    retain_ratios=DEFAULT_RETAIN_RATIOS,
    first_seed=first_seed,
    sigma_fraction=sigma_fraction
)

def validate_config(config: DatasetCreationConfig) -> None:
    if config.split_name not in {"train", "val", "test"}:
        raise ValueError("split_name must be 'train', 'val', or 'test'.")

    if config.number_of_volumes <= 0:
        raise ValueError("number_of_volumes must be positive.")

    if config.undersampling_per_slice <= 0:
        raise ValueError("undersampling_per_slice must be positive.")

    low, high = config.slice_percentile_range
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(
            "slice_percentile_range must satisfy 0 <= low < high <= 100."
        )

    if config.sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be positive.")

    unknown_planes = set(config.planes) - set(brain_planes)
    if unknown_planes:
        raise ValueError(
            f"Unknown planes: {sorted(unknown_planes)}. "
            f"Supported planes: {sorted(brain_planes)}."
        )

    for ratio in config.retain_ratios:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"Each retain ratio must be in (0, 1], received {ratio}."
            )

    if isinstance(config.slices_per_volume_per_plane, Mapping):
        missing = set(config.planes) - set(config.slices_per_volume_per_plane)
        if missing:
            raise ValueError(
                "Missing slice count for planes: "
                f"{sorted(missing)}."
            )
        counts = [
            config.slices_per_volume_per_plane[plane]
            for plane in config.planes
        ]
    else:
        counts = [config.slices_per_volume_per_plane]

    if any(count <= 0 for count in counts):
        raise ValueError("All slice counts must be positive.")

def get_slice_count(
    slices_per_volume_per_plane: int | Mapping[str, int],
    plane: str,
) -> int:
    if isinstance(slices_per_volume_per_plane, Mapping):
        return int(slices_per_volume_per_plane[plane])
    return int(slices_per_volume_per_plane)


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


def extract_slice(
    volume: np.ndarray,
    plane: str,
    slice_index: int,
) -> np.ndarray:
    """Extract a 2D slice using the orientation convention in data_utils.py."""
    axis = brain_planes[plane]

    if slice_index < 0 or slice_index >= volume.shape[axis]:
        raise IndexError(
            f"Slice index {slice_index} is invalid for plane {plane} "
            f"with axis length {volume.shape[axis]}."
        )

    if axis == 0:
        image = volume[slice_index, :, :]
    elif axis == 1:
        image = volume[:, slice_index, :]
    elif axis == 2:
        image = volume[:, :, slice_index]
    else:
        raise AssertionError(f"Unexpected axis: {axis}")

    return np.rot90(image)


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
    if split_root.exists():
        print(f"{split_root} already exists. Removing...")
        shutil.rmtree(split_root)
        print(f"{split_root} removed")
        assert not split_root.exists()

    print("creating output directories...")
    split_root.mkdir(parents=True, exist_ok=True)
    print(f"{split_root} created")

    originals = (split_root / "originals")
    originals.mkdir(parents=False, exist_ok=False)
    print(f"{originals} created")

    us = (split_root / "undersampled")
    us.mkdir(parents=False, exist_ok=False)
    print(f"{us} created")
    masks = (split_root / "masks")
    masks.mkdir(parents=False, exist_ok=False)
    print(f"{masks} created")

    for retain_ratio in retain_ratios:
        ratio_folder = ratio_folder_name(retain_ratio)

        us_ratio = (us / ratio_folder)
        us_ratio.mkdir(parents=False, exist_ok=False)
        print(f"{us_ratio} created")

        masks_ratio = (masks / ratio_folder)
        masks_ratio.mkdir(parents=False, exist_ok=False)
        print(f"{masks_ratio} created")


def choose_existing_volumes(
    metadata: pd.DataFrame,
    config: DatasetCreationConfig,
    rng: np.random.Generator,
) -> list[tuple[int, pd.Series, Path]]:
    if config.path_column not in metadata.columns:
        raise KeyError(
            f"CSV does not contain path column '{config.path_column}'. "
            f"Available columns: {list(metadata.columns)}"
        )

    row_order = rng.permutation(len(metadata))
    selected: list[tuple[int, pd.Series, Path]] = []
    seen_paths: set[Path] = set()
    skipped_missing = 0
    skipped_invalid = 0
    for positional_index in row_order:
        row = metadata.iloc[int(positional_index)]
        resolved_path = csv_path_to_local_path(str(row[config.path_column]))
        if resolved_path is None:
            skipped_missing += 1
            print("None")
            continue
        if resolved_path in seen_paths:
            print("in seen, skipped")
            continue
        abs_path = os.path.abspath(resolved_path)
        print(f"examining {abs_path}")
        try:
            print("loading volume...")
            volume = load_volume(abs_path)
            print("loaded")
        except (OSError, ValueError):
            skipped_invalid += 1
            continue

        if volume is None:
            print("value is None")
            skipped_missing += 1
            continue

        seen_paths.add(resolved_path)
        selected.append(
            (int(positional_index), row.copy(), abs_path)
        )

        if len(selected) == config.number_of_volumes:
            print(f"found {len(selected)} volumes")
            break

    if len(selected) < config.number_of_volumes:
        raise RuntimeError(
            f"Requested {config.number_of_volumes} existing volumes, "
            f"but found only {len(selected)} after checking "
            f"{len(metadata)} CSV rows. Missing: {skipped_missing}; "
            f"invalid: {skipped_invalid}."
        )

    print(
        f"Selected {len(selected)} volumes. "
        f"Skipped missing={skipped_missing}, invalid={skipped_invalid}."
    )
    return selected


def create_dataset_split(
    config: DatasetCreationConfig,
) -> pd.DataFrame:
    validate_config(config)
    print("config validated")
    source_csv_path = (config.source_dataset_root / config.source_csv_name)
    print(f"loading source_csv_path: {source_csv_path}")
    metadata = load_metadata(source_csv_path)
    print(f"metadata loaded. length: {metadata.shape[0]}")
    split_root = config.output_dataset_root / config.split_name
    print(f"saving split_root: {split_root}")
    prepare_output_directories(split_root, config.retain_ratios)

    print(f"current working directory: {os.getcwd()}")

    selection_rng = np.random.default_rng(config.first_seed)
    selected_volumes = choose_existing_volumes(
        metadata,
        config,
        selection_rng,
    )

    output_dtype = np.dtype(config.output_dtype)
    csv_records: list[dict[str, object]] = []

    # Kept independent of volume/slice selection RNG consumption.
    next_mask_seed = int(config.first_seed)
    sample_counter = 0
    original_counter = 0


    for selected_number, (csv_row_index, row, volume_path) in enumerate(
        selected_volumes,
        start=1,
    ):
        volume = load_volume(volume_path)
        if volume is None:
            # The path was checked above, but this keeps the loop robust.
            continue

        subject_value = (
            row[config.subject_column]
            if config.subject_column in metadata.columns
            else volume_path.stem
        )
        subject_id = safe_token(subject_value)
        volume_token = f"{subject_id}_row{csv_row_index:06d}"

        print(
            f"[{selected_number}/{len(selected_volumes)}] "
            f"{volume_path}, shape={volume.shape}"
        )

        for plane in config.planes:
            axis = brain_planes[plane]
            candidates = candidate_slice_indices(
                volume.shape[axis],
                config.slice_percentile_range,
            )
            slice_count = get_slice_count(
                config.slices_per_volume_per_plane,
                plane,
            )

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
                raw_slice = extract_slice(
                    volume,
                    plane,
                    int(slice_index),
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
                original_relative_path = (
                    Path("originals") / f"{original_id}.npy"
                )
                original_absolute_path = (
                    split_root / original_relative_path
                )
                np.save(
                    original_absolute_path,
                    normalized_slice,
                    allow_pickle=False,
                )
                original_counter += 1

                original_k_space = image_to_kspace(normalized_slice)

                for retain_ratio in config.retain_ratios:
                    ratio_folder = ratio_folder_name(retain_ratio)
                    retain_percentage = int(
                        round(retain_ratio * 100)
                    )

                    for repetition_index in range(
                        config.undersampling_per_slice
                    ):
                        mask_seed = next_mask_seed
                        next_mask_seed += 1

                        row_mask = create_unique_row_mask(
                            number_of_rows=original_k_space.shape[0],
                            retain_ratio=retain_ratio,
                            seed=mask_seed,
                            sigma_fraction=config.sigma_fraction,
                        )
                        undersampled_k_space = apply_row_mask(
                            original_k_space,
                            row_mask,
                        )
                        undersampled_image = kspace_to_image(
                            undersampled_k_space
                        ).astype(output_dtype, copy=False)

                        sample_id = (
                            f"{original_id}_r{retain_percentage:02d}_"
                            f"u{repetition_index:03d}"
                        )
                        mask_id = f"mask_{sample_id}"

                        mask_relative_path = (
                            Path("masks")
                            / ratio_folder
                            / f"{mask_id}.npy"
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
                                "original_volume_path": str(
                                    row[config.path_column]
                                ),
                                "plane": plane,
                                "slice_index": int(slice_index),
                                "original_image_file": str(
                                    original_relative_path.as_posix()
                                ),
                                "retain_ratio": float(retain_ratio),
                                "mask_id": mask_id,
                                "mask_file": str(
                                    mask_relative_path.as_posix()
                                ),
                                "undersampled_image_file": str(
                                    undersampled_relative_path.as_posix()
                                ),
                                "source_shape": json.dumps(
                                    list(normalized_slice.shape)
                                ),
                            }
                        )
                        sample_counter += 1

    samples = pd.DataFrame.from_records(csv_records)
    csv_output_path = split_root / "samples.csv"
    samples.to_csv(csv_output_path, index=False)

    print(
        f"Created split '{config.split_name}' at {split_root}\n"
        f"Volumes: {len(selected_volumes)}\n"
        f"Original slices: {original_counter}\n"
        f"Undersampled samples: {sample_counter}\n"
        f"CSV: {csv_output_path}"
    )

    return samples


def parse_plane_slice_counts(
    value: str,
) -> int | dict[str, int]:
    """
    Parse either:
        "20"
    or:
        "Axial=20,Coronal=15,Sagittal=15"
    """
    value = value.strip()
    if "=" not in value:
        return int(value)

    result: dict[str, int] = {}
    for item in value.split(","):
        plane, count = item.split("=", maxsplit=1)
        result[plane.strip()] = int(count)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a precomputed MRI undersampling dataset split."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "val", "test"),
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-csv",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--number-of-volumes",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--slices-per-plane",
        required=True,
        type=parse_plane_slice_counts,
        help=(
            "One count for every plane, e.g. '20', or plane-specific "
            "counts, e.g. 'Axial=20,Coronal=15,Sagittal=15'."
        ),
    )
    parser.add_argument(
        "--slice-percentiles",
        nargs=2,
        type=float,
        default=(25.0, 75.0),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--undersampling-per-slice",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--planes",
        nargs="+",
        default=("Axial",),
        choices=tuple(brain_planes.keys()),
    )
    parser.add_argument(
        "--retain-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_RETAIN_RATIOS,
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sigma-fraction",
        type=float,
        default=1 / 6,
    )
    parser.add_argument(
        "--path-column",
        default="filePath",
    )
    parser.add_argument(
        "--subject-column",
        default="Subject",
    )
    parser.add_argument(
        "--output-dtype",
        default="float32",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    config = DatasetCreationConfig(
        split_name=args.split,
        source_dataset_root=args.source_root,
        source_csv_name=args.source_csv,
        output_dataset_root=args.output_root,
        number_of_volumes=args.number_of_volumes,
        slices_per_volume_per_plane=args.slices_per_plane,
        slice_percentile_range=tuple(args.slice_percentiles),
        undersampling_per_slice=args.undersampling_per_slice,
        planes=tuple(args.planes),
        retain_ratios=tuple(args.retain_ratios),
        first_seed=args.first_seed,
        sigma_fraction=args.sigma_fraction,
        path_column=args.path_column,
        subject_column=args.subject_column,
        output_dtype=args.output_dtype,
        overwrite=args.overwrite,
    )
    create_dataset_split(config)


if __name__ == "__main__":
    main()
