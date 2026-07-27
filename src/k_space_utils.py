from typing import Callable

import numpy as np
from src.data_utils import extract_slice, min_max_normalize


def image_to_kspace(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected a 2D image")

    return np.fft.fftshift(np.fft.fft2(image))


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    if kspace.ndim != 2:
        raise ValueError("Expected a 2D k-space array")

    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))


def merge_kspace_with_data_consistency(
    acquired_k_space: np.ndarray,
    cnn_k_space: np.ndarray,
    zero_line_mask: np.ndarray,
) -> np.ndarray:
    """Merge k-space by keeping measured rows and replacing only zeroed rows."""
    if acquired_k_space.ndim != 2 or cnn_k_space.ndim != 2:
        raise ValueError("Both k-space inputs must be 2D arrays")
    if acquired_k_space.shape != cnn_k_space.shape:
        raise ValueError(
            f"Shape mismatch: acquired={acquired_k_space.shape}, cnn={cnn_k_space.shape}"
        )
    if zero_line_mask.ndim != 1 or zero_line_mask.shape[0] != acquired_k_space.shape[0]:
        raise ValueError("zero_line_mask must be a 1D array with one value per k-space row")
    if not np.all(np.isin(zero_line_mask, (False, True, 0, 1))):
        raise ValueError("zero_line_mask must be boolean (or 0/1-convertible).")
    if not np.iscomplexobj(acquired_k_space) or not np.iscomplexobj(cnn_k_space):
        raise ValueError("k-space inputs must remain complex for data consistency merging.")
    if not np.all(np.isfinite(acquired_k_space)) or not np.all(np.isfinite(cnn_k_space)):
        raise ValueError("k-space inputs contain NaN or Inf values.")

    zero_line_mask = zero_line_mask.astype(bool, copy=False)
    merged = acquired_k_space.copy()
    merged[zero_line_mask, :] = cnn_k_space[zero_line_mask, :]
    if not np.iscomplexobj(merged):
        raise ValueError("Combined k-space unexpectedly lost complex dtype.")
    if not np.all(np.isfinite(merged)):
        raise ValueError("Combined k-space contains NaN or Inf values.")
    return merged


def enforce_kspace_data_consistency(
    reconstructed_image: np.ndarray,
    acquired_k_space: np.ndarray,
    zero_line_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply FFT-based data consistency and return final image + debug k-spaces."""
    if reconstructed_image.ndim != 2:
        raise ValueError("Expected reconstructed_image to be 2D")
    if acquired_k_space.ndim != 2:
        raise ValueError("Expected acquired_k_space to be 2D")
    if reconstructed_image.shape != acquired_k_space.shape:
        raise ValueError(
            f"Shape mismatch: reconstructed_image={reconstructed_image.shape}, "
            f"acquired_k_space={acquired_k_space.shape}"
        )
    if not np.all(np.isfinite(reconstructed_image)):
        raise ValueError("reconstructed_image contains NaN or Inf values.")

    cnn_k_space = image_to_kspace(reconstructed_image)
    if not np.iscomplexobj(cnn_k_space):
        raise ValueError("CNN-derived k-space must be complex.")
    merged_k_space = merge_kspace_with_data_consistency(
        acquired_k_space=acquired_k_space,
        cnn_k_space=cnn_k_space,
        zero_line_mask=zero_line_mask,
    )
    final_image = kspace_to_image(merged_k_space)
    if not np.all(np.isfinite(final_image)):
        raise ValueError("Final reconstructed image contains NaN or Inf values.")
    return final_image, cnn_k_space, merged_k_space


def kspace_log_magnitude(
    kspace: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    return np.log(np.abs(kspace) + eps)


def zero_central_rows(
    k_space: np.ndarray,
    num_rows: int,
    eps: float = 1e-8,
) -> np.ndarray:
    if k_space.ndim != 2:
        raise ValueError("Expected a 2D k-space array")
    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")
    height = k_space.shape[0]
    if num_rows > height:
        raise ValueError(f"num_rows cannot exceed number of rows ({height})")

    start = (height - num_rows) // 2
    end = start + num_rows

    k_space_zeroed = np.array(k_space, copy=True)
    k_space_zeroed[start:end, :] = 0
    return k_space_zeroed

def zero_every_n_rows(
    k_space: np.ndarray,
    num_rows_to_zero: int,
    n: int,
) -> np.ndarray:
    if k_space.ndim != 2:
        raise ValueError("Expected a 2D k-space array")
    if num_rows_to_zero < 0:
        raise ValueError("num_rows_to_zero must be non-negative")
    if n <= 0:
        raise ValueError("n must be positive")

    height = k_space.shape[0]
    if num_rows_to_zero > height:
        raise ValueError(f"num_rows_to_zero cannot exceed number of rows ({height})")

    k_space_zeroed = np.array(k_space, copy=True)
    for i in range(0, height, n):
        end = min(i + num_rows_to_zero, height)
        k_space_zeroed[i:end, :] = 0
    return k_space_zeroed


def zero_k_space_rows_normal_random_dist(
    k_space: np.ndarray,
    n: int,
    seed: int | None = None,
    sigma_fraction: float = 1 / 6,
) -> tuple[np.ndarray, np.ndarray]:

    if not isinstance(k_space, np.ndarray):
        raise TypeError("k_space must be a NumPy array.")

    if k_space.ndim != 2:
        raise ValueError(
            f"k_space must be a 2D array, got shape {k_space.shape}."
        )

    if not np.iscomplexobj(k_space):
        raise ValueError("k_space must be complex-valued.")

    number_of_rows = k_space.shape[0]

    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")

    if n < 0 or n > number_of_rows:
        raise ValueError(
            f"n must be between 0 and {number_of_rows}, got {n}."
        )

    if sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be positive.")

    if n == 0:
        return k_space.copy(), np.array([], dtype=int)

    rng = np.random.default_rng(seed)

    row_indices = np.arange(number_of_rows)
    center = (number_of_rows - 1) / 2
    sigma = number_of_rows * sigma_fraction

    # High near the center, low near the outer rows.
    preservation_probability = np.exp(
        -0.5 * ((row_indices - center) / sigma) ** 2
    )

    # Convert preservation likelihood into removal likelihood.
    zero_probability = 1.0 - preservation_probability

    # Avoid exact zero probability at the central row.
    zero_probability += np.finfo(float).eps
    zero_probability /= zero_probability.sum()

    zeroed_rows = rng.choice(
        row_indices,
        size=n,
        replace=False,
        p=zero_probability,
    )

    zeroed_k_space = k_space.copy()
    zeroed_k_space[zeroed_rows, :] = 0

    return zeroed_k_space, np.sort(zeroed_rows)

def create_images_from_central_slice_in_volume_with_operation(
        volume: np.ndarray,
        axis: int,
        k_space_op: Callable[[np.ndarray], np.ndarray] | None,
        **k_space_op_kwargs
) -> dict:
    slice_index, original_slice = extract_slice(volume, axis, central=True)
    normalized_slice = min_max_normalize(original_slice)
    k_space = image_to_kspace(normalized_slice)
    original_k_space = k_space.copy()
    zero_lines_indices = np.array([], dtype=int)
    if k_space_op is not None:
        k_space_result = k_space_op(k_space, **k_space_op_kwargs)
        if isinstance(k_space_result, tuple):
            if len(k_space_result) != 2:
                raise ValueError("k_space_op must return either ndarray or (ndarray, zero_lines_indices)")
            k_space, zero_lines_indices = k_space_result
            zero_lines_indices = np.asarray(zero_lines_indices, dtype=int)
        else:
            k_space = k_space_result
    image_from_kspace = kspace_to_image(k_space)
    k_space_vis = kspace_log_magnitude(k_space)
    max_error = np.max(np.abs(image_from_kspace - normalized_slice))

    return {
        'slice_index': slice_index,
        'original': original_slice,
        'normalized': normalized_slice,
        'k_space_vis': k_space_vis,
        'original_k_space': original_k_space,
        'k_space': k_space,
        'zero_lines_indices': zero_lines_indices,
        'image_from_kspace': image_from_kspace,
        'max_error': max_error,
    }

def verify_non_zero_lines(
    complex_image1: np.ndarray,
    complex_image2: np.ndarray,
) -> tuple[bool, int]:
    """Verify that retained rows in ``complex_image2`` match ``complex_image1``.

    A zero row has zero real and imaginary values in every column. Such rows in
    ``complex_image2`` are ignored; every other row must be exactly equal to
    its corresponding row in ``complex_image1``.
    """
    if not isinstance(complex_image1, np.ndarray) or not isinstance(complex_image2, np.ndarray):
        raise TypeError("Both complex images must be NumPy arrays.")
    if complex_image1.ndim != 2 or complex_image2.ndim != 2:
        raise ValueError("Both complex images must be 2D arrays.")
    if complex_image1.shape != complex_image2.shape:
        raise ValueError("Both complex images must have the same shape.")
    if not np.iscomplexobj(complex_image1) or not np.iscomplexobj(complex_image2):
        raise ValueError("Both complex images must be complex-valued.")

    non_zero_lines = ~np.all(complex_image2 == 0, axis=1)
    number_of_non_zero_lines = int(np.count_nonzero(non_zero_lines))
    lines_are_equal = np.array_equal(
        complex_image1[non_zero_lines],
        complex_image2[non_zero_lines],
    )
    return lines_are_equal, number_of_non_zero_lines
