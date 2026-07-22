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
    # allow partial data set for local runs
    if not volume_path.exists():
        return None
    volume = np.load(volume_path)
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, received shape {volume.shape}"
        )
    return volume


def select_central_slice(
    volume: np.ndarray,
    axis: int
) -> tuple[int, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    if 0 > axis > 3:
        raise ValueError("Axis must be 0, 1, or 2")
    index = volume.shape[axis] // 2
    slice_image = None
    if axis == 0:
        slice_image = np.rot90(volume[index, :, :])
    elif axis == 1:
        slice_image = np.rot90(volume[:, index, :])
    elif axis == 2:
        slice_image = np.rot90(volume[:, :, index])
    else:
        assert False
    return index, slice_image


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


