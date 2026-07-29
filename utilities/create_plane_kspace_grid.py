from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.create_mri_dataset import apply_row_mask, create_unique_row_mask
from src.data_utils import min_max_normalize
from src.k_space_utils import image_to_kspace, kspace_log_magnitude


DEFAULT_RETAIN_RATIOS = (0.20, 0.30, 0.50)
DEFAULT_SIGMA_FRACTION = 1 / 6
KSPACE_CMAP = "magma"


@dataclass(frozen=True)
class SliceSelection:
    plane: str
    sample_id: str
    original_path: Path


def _normalize_plane_name(raw_plane: str) -> str:
    text = str(raw_plane).strip().lower()
    if text in {"axial", "coronal", "sagittal"}:
        return text.capitalize()
    return text.capitalize() if text else "Unknown"


def _infer_plane_from_text(text: str) -> str:
    lowered = text.lower()
    if "axial" in lowered:
        return "Axial"
    if "coronal" in lowered:
        return "Coronal"
    if "sagittal" in lowered:
        return "Sagittal"
    return "Unknown"


def _validate_split_layout(split_root: Path) -> Path:
    required = ("originals", "masks", "undersampled")
    missing = [name for name in required if not (split_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Split root is missing required folders: {missing}. Expected under: {split_root}"
        )

    # Support both spellings to match user request and existing project layout.
    kspace_candidates = (split_root / "k_spaces", split_root / "k_sample")
    kspace_dir = next((p for p in kspace_candidates if p.exists()), None)
    if kspace_dir is None:
        raise FileNotFoundError(
            "Expected one of these folders to exist: 'k_spaces' or 'k_sample'."
        )
    return kspace_dir


def _load_samples_csv(split_root: Path) -> pd.DataFrame | None:
    csv_path = split_root / "samples.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _build_selections_from_csv(
    split_root: Path,
    samples: pd.DataFrame,
    rng: np.random.Generator,
) -> list[SliceSelection]:
    if "original_image_file" not in samples.columns:
        raise ValueError("'samples.csv' is missing required column: original_image_file")

    work = samples.copy()
    if "plane" in work.columns:
        work["_plane"] = work["plane"].apply(_normalize_plane_name)
    else:
        work["_plane"] = work["original_image_file"].apply(_infer_plane_from_text)

    plane_order = ["Axial", "Coronal", "Sagittal"]
    selected: list[SliceSelection] = []

    for plane in plane_order:
        plane_rows = work[work["_plane"] == plane]
        if len(plane_rows) == 0:
            continue
        row = plane_rows.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        original_path = split_root / str(row["original_image_file"])
        selected.append(
            SliceSelection(
                plane=plane,
                sample_id=str(row.get("sample_id", original_path.stem)),
                original_path=original_path,
            )
        )

    if len(selected) >= 3:
        return selected[:3]

    # Fill missing rows from any remaining available planes/files to always create 3 pairs.
    existing_paths = {sel.original_path.resolve() for sel in selected if sel.original_path.exists()}
    fallback_rows = work
    for _, row in fallback_rows.iterrows():
        if len(selected) >= 3:
            break
        original_path = split_root / str(row["original_image_file"])
        if not original_path.exists():
            continue
        resolved = original_path.resolve()
        if resolved in existing_paths:
            continue
        existing_paths.add(resolved)
        selected.append(
            SliceSelection(
                plane=_normalize_plane_name(row.get("_plane", "Unknown")),
                sample_id=str(row.get("sample_id", original_path.stem)),
                original_path=original_path,
            )
        )

    return selected


def _build_selections_from_originals(
    originals_dir: Path,
    rng: np.random.Generator,
) -> list[SliceSelection]:
    files = sorted(originals_dir.glob("*.npy"))
    if len(files) < 3:
        raise ValueError(
            f"Need at least 3 files in {originals_dir} to create a 6-line figure. Found: {len(files)}"
        )

    rng.shuffle(files)
    selections: list[SliceSelection] = []
    for file_path in files:
        if len(selections) >= 3:
            break
        selections.append(
            SliceSelection(
                plane=_infer_plane_from_text(file_path.name),
                sample_id=file_path.stem,
                original_path=file_path,
            )
        )
    return selections


def _load_original_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Original image file not found: {path}")
    image = np.load(path, allow_pickle=False)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image at {path}, got shape {image.shape}")
    return image


def _plot_image(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len - 3]}..."


def _build_grid_figure(
    selections: list[SliceSelection],
    output_path: Path,
    seed: int,
    sigma_fraction: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
) -> None:
    if len(selections) < 3:
        raise ValueError("Need 3 slice selections to create 6 rows (3 line-pairs).")

    # Keep title rows short and image rows tall so each subplot remains readable.
    nrows_spec = 3 * 3  # 3 pairs x (1 title row + 2 image rows)
    row_heights = [0.09, 1.0, 1.0] * 3
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=nrows_spec,
        ncols=3,
        height_ratios=row_heights,
        hspace=0.10,
        wspace=0.08,
    )
    fig.subplots_adjust(top=0.975, bottom=0.02, left=0.03, right=0.97)

    rng = np.random.default_rng(seed)

    for pair_index, selection in enumerate(selections[:3]):
        # Title row (doesn't hold axes, just space for text)
        title_row = pair_index * 3
        img_row_top = title_row + 1
        img_row_bottom = title_row + 2

        original = _load_original_image(selection.original_path)
        normalized = min_max_normalize(original)
        kspace = image_to_kspace(normalized)
        kspace_vis = kspace_log_magnitude(kspace)

        # First image row: original, normalized, and k-space visualization.
        ax_orig = fig.add_subplot(gs[img_row_top, 0])
        ax_norm = fig.add_subplot(gs[img_row_top, 1])
        ax_ksp = fig.add_subplot(gs[img_row_top, 2])

        _plot_image(ax_orig, original, "Original")
        _plot_image(ax_norm, normalized, "Normalized")
        _plot_image(ax_ksp, kspace_vis, "K-space log magnitude", cmap=KSPACE_CMAP)

        # Build masked k-space visualizations once, then render all with shared contrast.
        masked_kspace_views: list[tuple[int, np.ndarray]] = []
        for retain_ratio in DEFAULT_RETAIN_RATIOS:
            row_mask = create_unique_row_mask(
                number_of_rows=kspace.shape[0],
                retain_ratio=retain_ratio,
                seed=int(rng.integers(0, 2**31 - 1)),
                sigma_fraction=sigma_fraction,
            )
            undersampled_kspace = apply_row_mask(kspace, row_mask)
            undersampled_kspace_vis = kspace_log_magnitude(undersampled_kspace)
            retain_pct = int(round(retain_ratio * 100))
            masked_kspace_views.append((retain_pct, undersampled_kspace_vis))

        # Use one value range so all k-space panels are visually comparable.
        kspace_stack = np.stack([kspace_vis] + [view for _, view in masked_kspace_views], axis=0)
        kspace_vmin = float(np.percentile(kspace_stack, 5))
        kspace_vmax = float(np.percentile(kspace_stack, 99.5))
        _plot_image(
            ax_ksp,
            kspace_vis,
            "K-space log magnitude",
            cmap=KSPACE_CMAP,
            vmin=kspace_vmin,
            vmax=kspace_vmax,
        )

        for col, (retain_pct, undersampled_kspace_vis) in enumerate(masked_kspace_views):
            ax_us = fig.add_subplot(gs[img_row_bottom, col])
            _plot_image(
                ax_us,
                undersampled_kspace_vis,
                f"K-space retain {retain_pct}%",
                cmap=KSPACE_CMAP,
                vmin=kspace_vmin,
                vmax=kspace_vmax,
            )

        # Add pair title above the image rows in the title row area.
        pair_title = (
            f"{selection.plane} slice | sample: {_truncate_text(selection.sample_id, 42)}\n"
            f"file: {_truncate_text(selection.original_path.name, 56)}"
        )
        ax_title = fig.add_subplot(gs[title_row, :])
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.02, pair_title,
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold",
            transform=ax_title.transAxes,
            wrap=True,
        )

    fig.suptitle("MRI Slice and K-space Retention Grid", fontsize=16, y=0.988)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_plane_kspace_grid(
    split_root: str | Path,
    output_path: str | Path | None = None,
    seed: int = 42,
    sigma_fraction: float = DEFAULT_SIGMA_FRACTION,
    fig_width: float = 18.0,
    fig_height: float = 24.0,
    dpi: int = 200,
) -> Path:
    split_root = Path(split_root).resolve()
    _validate_split_layout(split_root)

    rng = np.random.default_rng(seed)
    samples = _load_samples_csv(split_root)

    if samples is not None and len(samples) > 0:
        selections = _build_selections_from_csv(split_root, samples, rng)
    else:
        selections = _build_selections_from_originals(split_root / "originals", rng)

    if len(selections) < 3:
        raise ValueError(
            "Could not select 3 slices for the 6-line grid. "
            "Ensure data includes at least 3 suitable originals."
        )

    if output_path is None:
        output_path = split_root / "plane_kspace_grid.png"
    else:
        output_path = Path(output_path).resolve()

    _build_grid_figure(
        selections=selections,
        output_path=output_path,
        seed=seed,
        sigma_fraction=sigma_fraction,
        fig_width=float(fig_width),
        fig_height=float(fig_height),
        dpi=int(dpi),
    )
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 6-line MRI visualization grid from a split folder containing "
            "originals/masks/undersampled/k-space data."
        )
    )
    parser.add_argument(
        "split_root",
        type=str,
        help="Path to split folder (e.g., train) containing originals, masks, undersampled.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="PNG output path. Defaults to <split_root>/plane_kspace_grid.png",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for slice and mask selection.",
    )
    parser.add_argument(
        "--sigma-fraction",
        type=float,
        default=DEFAULT_SIGMA_FRACTION,
        help="Gaussian sigma fraction used in row mask generation.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=18.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=24.0,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG export DPI.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_path = create_plane_kspace_grid(
        split_root=args.split_root,
        output_path=args.output,
        seed=args.seed,
        sigma_fraction=float(args.sigma_fraction),
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
    )
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()

