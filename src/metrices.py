import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity


def _prepare_metric_images(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare two images for PSNR/SSIM calculation.

    Complex-valued images are converted to magnitude images.

    Parameters
    ----------
    reference:
        Ground-truth/reference 2D image.
    reconstructed:
        Reconstructed 2D image.

    Returns
    -------
    reference_real, reconstructed_real:
        Real-valued float64 images with matching shapes.
    """
    reference = np.asarray(reference)
    reconstructed = np.asarray(reconstructed)

    if reference.shape != reconstructed.shape:
        raise ValueError(
            "reference and reconstructed must have the same shape. "
            f"Got {reference.shape} and {reconstructed.shape}."
        )

    if reference.ndim != 2:
        raise ValueError(
            f"Expected 2D images, but got ndim={reference.ndim}."
        )

    if np.iscomplexobj(reference) or np.iscomplexobj(reconstructed):
        raise ValueError("only real images are supported")

    reference = reference.astype(np.float64, copy=False)
    reconstructed = reconstructed.astype(np.float64, copy=False)

    if not np.all(np.isfinite(reference)):
        raise ValueError("reference contains NaN or infinite values.")

    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("reconstructed contains NaN or infinite values.")

    return reference, reconstructed

def calculate_psnr(
    reference: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    reference, reconstructed = _prepare_metric_images(
        reference,
        reconstructed,
    )
    data_range = float(reference.max() - reference.min())
    if data_range <= 0:
        if np.array_equal(reference, reconstructed):
            return float("inf")
        else:
            raise ValueError(
                "The reference image has zero intensity range. "
                "Provide data_range explicitly."
            )
    """
    return float(
        peak_signal_noise_ratio(
            reference,
            reconstructed,
            data_range=data_range,
        )
    )
    """
    mask = reference > 0
    mse_roi = np.mean(
        (reference[mask] - reconstructed[mask]) ** 2
    )
    psnr_roi = 10 * np.log10(
        data_range ** 2 / mse_roi
    )
    return psnr_roi

def calculate_ssim(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float | None = None,
    win_size: int | None = None,
) -> float:

    reference, reconstructed = _prepare_metric_images(
        reference,
        reconstructed,
    )

    data_range = float(reference.max() - reference.min())

    if data_range <= 0:
        if np.array_equal(reference, reconstructed):
            return 1.0

        raise ValueError(
            "The reference image has zero intensity range. "
            "Provide data_range explicitly."
        )

    return float(
        structural_similarity(
            reference,
            reconstructed,
            data_range=data_range,
            win_size=win_size,
        )
    )


def calculate_nrmse(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Calculate ||reference - reconstructed||_2 / ||reference||_2."""
    reference, reconstructed = _prepare_metric_images(reference, reconstructed)
    reference_norm = np.linalg.norm(reference)
    error_norm = np.linalg.norm(reference - reconstructed)

    if reference_norm == 0:
        if error_norm == 0:
            return 0.0
        return float("inf")

    return float(error_norm / reference_norm)


def calculate_mae(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Calculate the mean absolute error over all image pixels."""
    reference, reconstructed = _prepare_metric_images(reference, reconstructed)
    return float(np.mean(np.abs(reference - reconstructed)))


def calculate_hfen(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    sigma: float = 1.5,
) -> float:
    """Calculate normalized high-frequency error using a LoG filter."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    reference, reconstructed = _prepare_metric_images(reference, reconstructed)
    reference_log = gaussian_laplace(reference, sigma=sigma)
    reconstructed_log = gaussian_laplace(reconstructed, sigma=sigma)
    reference_log_norm = np.linalg.norm(reference_log)
    error_log_norm = np.linalg.norm(reference_log - reconstructed_log)

    if reference_log_norm == 0:
        if error_log_norm == 0:
            return 0.0
        return float("inf")

    return float(error_log_norm / reference_log_norm)


def calculate_image_quality_metrics(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float]:
    """
    Calculate PSNR, SSIM, NRMSE, MAE, and HFEN between two 2D images.
    """
    reference, reconstructed = _prepare_metric_images(
        reference,
        reconstructed,
    )

    psnr = calculate_psnr(
        reference,
        reconstructed,
    )

    ssim = calculate_ssim(
        reference,
        reconstructed
    )
    nrmse = calculate_nrmse(reference, reconstructed)
    mae = calculate_mae(reference, reconstructed)
    hfen = calculate_hfen(reference, reconstructed)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "nrmse": nrmse,
        "mae": mae,
        "hfen": hfen,
    }