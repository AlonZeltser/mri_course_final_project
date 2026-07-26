from pathlib import Path
import os
import numpy as np
import pandas as pd


def csv_path_to_local_path(csv_path: str) -> Path:
    return os.path.join("selected_npy", os.path.basename(csv_path))

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_volume(volume_path: str | Path) -> np.ndarray | None:
    volume_path = Path(volume_path)
    if not volume_path.exists():
        return None
    try:
        volume = np.load(volume_path)
    except (EOFError, OSError, ValueError):
        return None
    if volume.ndim != 3:
        return None
    return volume


def extract_slice(
    volume: np.ndarray,
    axis: int,
    slice_index: int | None = None,
    central: bool = False,
) -> tuple[int, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    if 0 > axis > 3:
        raise ValueError("Axis must be 0, 1, or 2")
    if central:
        index = volume.shape[axis] // 2
    elif slice_index is not None:
        index = slice_index
    else:
        raise ValueError("Either central must be True or slice_index must be provided")

    slice_image = None
    if axis == 0:
        slice_image = volume[index, :, :]
    elif axis == 1:
        slice_image = volume[:, index, :]
    elif axis == 2:
        slice_image = volume[:, :, index]
    else:
        raise AssertionError(f"Unexpected axis: {axis}")
    return index, np.rot90(slice_image)


def min_max_normalize(
    image: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    image = image.astype(np.float64)
    image = image - np.min(image)
    minimum = image.min()
    maximum = image.max()
    if maximum - minimum < eps:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


