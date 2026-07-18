import numpy as np
from src.data_utils import select_central_slice, min_max_normalize


def image_to_kspace(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected a 2D image")

    return np.fft.fftshift(np.fft.fft2(image))


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    if kspace.ndim != 2:
        raise ValueError("Expected a 2D k-space array")

    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace)))


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


def zero_k_space_rows_random_dist(
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

    return zeroed_k_space # np.sort(zeroed_rows)

def create_missing_central_slice_from_volume(volume: np.ndarray, axis: int, k_space_op=None, **k_space_op_kwargs) -> dict:
    slice_index, original_slice = select_central_slice(volume, axis)
    normalized_slice = min_max_normalize(original_slice)
    k_space = image_to_kspace(normalized_slice)
    original_k_space = k_space.copy()
    if k_space_op is not None:
        k_space = k_space_op(k_space, **k_space_op_kwargs)
    missing_central = kspace_to_image(k_space)
    k_space_vis = kspace_log_magnitude(k_space)
    max_error = np.max(np.abs(missing_central - normalized_slice))

    return {
        'slice_index': slice_index,
        'original': original_slice,
        'normalized': normalized_slice,
        'k_space_vis': k_space_vis,
        'original_k_space': original_k_space,
        'k_space': k_space,
        'missing_central': missing_central,
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
