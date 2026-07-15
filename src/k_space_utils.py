import numpy as np


def image_to_kspace(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected a 2D image")

    return np.fft.fftshift(np.fft.fft2(image))


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    if kspace.ndim != 2:
        raise ValueError("Expected a 2D k-space array")

    return np.fft.ifft2(np.fft.ifftshift(kspace))


def kspace_log_magnitude(
    kspace: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    return np.log(np.abs(kspace) + eps)