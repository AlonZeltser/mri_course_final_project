from src.k_space_utils import *
from src.data_utils import *
from scipy.interpolate import griddata

def k_space_interpolation(
    k_space: np.ndarray,
    method: str = 'linear_1d',
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    supported_methods = {'linear_1d', 'bilinear', 'bicubic', 'nearest_row'}
    if k_space.ndim != 2:
        raise ValueError('k_space must be 2D')
    if tolerance < 0:
        raise ValueError('tolerance must be non-negative')
    if method not in supported_methods:
        raise ValueError(
            "method must be one of: 'linear_1d', 'nearest_row', "
            "'bilinear', 'bicubic'"
        )


    magnitude_image = np.abs(k_space)
    sum_lines = np.sum(magnitude_image, axis=1)
    zero_mask = sum_lines <= tolerance
    zero_lines_indices = np.where(zero_mask)[0]

    corrected_k_space = k_space.copy()
    if zero_lines_indices.size == 0:
        return corrected_k_space, zero_lines_indices

    row_indices = np.arange(k_space.shape[0])
    known_rows = row_indices[~zero_mask]

    # If all rows are missing, there is nothing reliable to interpolate from.
    if known_rows.size == 0:
        return corrected_k_space, zero_lines_indices

    def interpolate_1d(values: np.ndarray) -> np.ndarray:
        known_values = values[known_rows]
        if known_rows.size == 1:
            return np.full(zero_lines_indices.shape, known_values[0], dtype=values.dtype)
        return np.interp(zero_lines_indices, known_rows, known_values)

    if method == 'nearest_row':
        center_row = (k_space.shape[0] - 1) / 2
        for missing_row in zero_lines_indices:
            distances = np.abs(known_rows - missing_row)
            closest_rows = known_rows[distances == distances.min()]
            inward_distances = np.abs(closest_rows - center_row)
            source_row = closest_rows[np.argmin(inward_distances)]

            corrected_k_space[missing_row, :] = k_space[source_row, :]

        return corrected_k_space, zero_lines_indices

    is_complex = np.iscomplexobj(k_space)
    if (
        method == 'linear_1d'
        or known_rows.size < 2
        or (method == 'bicubic' and known_rows.size < 4)
        or k_space.shape[1] < 2
    ):
        for col_idx in range(k_space.shape[1]):
            real_values = interpolate_1d(np.real(k_space[:, col_idx]))
            if is_complex:
                imag_values = interpolate_1d(np.imag(k_space[:, col_idx]))
                corrected_k_space[zero_lines_indices, col_idx] = real_values + 1j * imag_values
            else:
                corrected_k_space[zero_lines_indices, col_idx] = real_values
        return corrected_k_space, zero_lines_indices

    all_rows, all_cols = np.indices(k_space.shape)
    known_mask = ~np.broadcast_to(zero_mask[:, np.newaxis], k_space.shape)
    missing_mask = ~known_mask
    known_points = np.column_stack((all_rows[known_mask], all_cols[known_mask]))
    missing_points = np.column_stack((all_rows[missing_mask], all_cols[missing_mask]))
    scipy_method = 'linear' if method == 'bilinear' else 'cubic'

    def interpolate_2d(values: np.ndarray) -> np.ndarray:
        interpolated = griddata(
            known_points,
            values[known_mask],
            missing_points,
            method=scipy_method,
        )

        # Linear/cubic interpolation cannot extrapolate past the outer known
        # rows. Fill those edge points with nearest-neighbor values instead.
        if np.any(np.isnan(interpolated)):
            nearest = griddata(
                known_points,
                values[known_mask],
                missing_points,
                method='nearest',
            )
            interpolated = np.where(np.isnan(interpolated), nearest, interpolated)
        return interpolated

    real_values = interpolate_2d(np.real(k_space))
    if is_complex:
        imag_values = interpolate_2d(np.imag(k_space))
        corrected_k_space[missing_mask] = real_values + 1j * imag_values
    else:
        corrected_k_space[missing_mask] = real_values

    return corrected_k_space, zero_lines_indices


def interpolation_reconstruction(
    harmed_k_space: np.ndarray,
    method: str = 'linear_1d',
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    if harmed_k_space.ndim != 2:
        raise ValueError('harmed_k_space must be 2D')

    if method == 'zero_filling':
        return kspace_to_image(harmed_k_space), harmed_k_space

    corrected_k_space, _ = k_space_interpolation(
        harmed_k_space,
        method=method,
        tolerance=tolerance,
    )
    lines_are_equal, number_of_non_zero_lines = verify_non_zero_lines(corrected_k_space, harmed_k_space)
    if not lines_are_equal:
        raise ValueError(f"Non-zero lines verification failed. Number of non-zero lines: {number_of_non_zero_lines}")
    reconstructed_image = kspace_to_image(corrected_k_space)
    return reconstructed_image, corrected_k_space
