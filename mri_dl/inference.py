from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .train_utils import choose_device
from src.k_space_utils import enforce_kspace_data_consistency, infer_zero_line_mask


def predict_tensor(model: nn.Module, input_tensor: torch.Tensor,
                   device: str | torch.device | None = None,
                   clamp_to_unit_range: bool = True) -> torch.Tensor:
    """Run inference on a tensor.

    Input rank is preserved in the output:
      - [H, W]       -> [H, W]
      - [C, H, W]    -> [C, H, W]
      - [N, C, H, W] -> [N, C, H, W]
    """
    resolved = choose_device(str(device) if device is not None else None)
    model.to(resolved); model.eval()

    input_ndim = input_tensor.ndim
    if input_tensor.ndim == 2:
        input_tensor = input_tensor[None, None, ...]   # -> [1,1,H,W]
    elif input_tensor.ndim == 3:
        input_tensor = input_tensor[None, ...]         # -> [1,C,H,W]
    if input_tensor.ndim != 4:
        raise ValueError('Expected [H,W], [C,H,W], or [N,C,H,W].')

    model_input = input_tensor.to(resolved, dtype=torch.float32)
    with torch.no_grad():
        predicted_delta = model(model_input)
    pred = model_input + predicted_delta
    if clamp_to_unit_range:
        pred = pred.clamp(0.0, 1.0)
    pred = pred.cpu()

    # Squeeze batch/channel dims back to match the caller's input rank.
    if input_ndim == 2:
        pred = pred[0, 0]   # [1,1,H,W] -> [H,W]
    elif input_ndim == 3:
        pred = pred[0]      # [1,C,H,W] -> [C,H,W]
    return pred


def predict_numpy(model: nn.Module, image: np.ndarray,
                  device: str | torch.device | None = None) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f'Expected 2D image, got {image.shape}')
    tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32, copy=False)))
    return predict_tensor(model, tensor, device=device).numpy()


def predict_file(model: nn.Module, input_file: str | Path,
                 output_file: str | Path | None = None,
                 device: str | torch.device | None = None) -> np.ndarray:
    image = np.load(input_file, allow_pickle=False)
    pred = predict_numpy(model, image, device=device)
    if output_file is not None:
        output_file = Path(output_file); output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, pred.astype(np.float32), allow_pickle=False)
    return pred


def predict_numpy_with_data_consistency(
    model: nn.Module,
    image: np.ndarray,
    acquired_k_space: np.ndarray,
    device: str | torch.device | None = None,
    clamp_to_unit_range: bool = True,
    zero_line_atol: float = 0.0,
    return_debug: bool = False,
):
    """Predict image then enforce row-wise k-space data consistency."""
    reconstructed = predict_numpy(model, image, device=device)
    zero_line_mask = infer_zero_line_mask(acquired_k_space, atol=zero_line_atol)
    final_image, cnn_k_space, merged_k_space = enforce_kspace_data_consistency(
        reconstructed_image=reconstructed,
        acquired_k_space=acquired_k_space,
        zero_line_mask=zero_line_mask,
        atol=zero_line_atol,
    )
    if clamp_to_unit_range:
        final_image = final_image.clip(0.0, 1.0)
    final_image = final_image.astype(np.float32, copy=False)
    if not return_debug:
        return final_image
    return {
        'reconstructed_image': reconstructed.astype(np.float32, copy=False),
        'final_image': final_image,
        'cnn_k_space': cnn_k_space,
        'merged_k_space': merged_k_space,
        'zero_line_mask': zero_line_mask,
    }


def predict_file_with_data_consistency(
    model: nn.Module,
    input_file: str | Path,
    k_space_file: str | Path,
    output_file: str | Path | None = None,
    device: str | torch.device | None = None,
    clamp_to_unit_range: bool = True,
    zero_line_atol: float = 0.0,
) -> np.ndarray:
    image = np.load(input_file, allow_pickle=False)
    acquired_k_space = np.load(k_space_file, allow_pickle=False)
    pred = predict_numpy_with_data_consistency(
        model=model,
        image=image,
        acquired_k_space=acquired_k_space,
        device=device,
        clamp_to_unit_range=clamp_to_unit_range,
        zero_line_atol=zero_line_atol,
    )
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, pred.astype(np.float32), allow_pickle=False)
    return pred

