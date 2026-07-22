from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .train_utils import choose_device


def predict_tensor(model: nn.Module, input_tensor: torch.Tensor,
                   device: str | torch.device | None = None,
                   clamp_to_unit_range: bool = True) -> torch.Tensor:
    resolved = choose_device(str(device) if device is not None else None)
    model.to(resolved); model.eval()
    if input_tensor.ndim == 2:
        input_tensor = input_tensor[None, None, ...]
    elif input_tensor.ndim == 3:
        input_tensor = input_tensor[None, ...]
    if input_tensor.ndim != 4:
        raise ValueError('Expected [H,W], [C,H,W], or [N,C,H,W].')
    with torch.no_grad():
        pred = model(input_tensor.to(resolved, dtype=torch.float32))
    if clamp_to_unit_range:
        pred = pred.clamp(0.0, 1.0)
    return pred.cpu()


def predict_numpy(model: nn.Module, image: np.ndarray,
                  device: str | torch.device | None = None) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f'Expected 2D image, got {image.shape}')
    tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32, copy=False)))
    return predict_tensor(model, tensor, device=device)[0, 0].numpy()


def predict_file(model: nn.Module, input_file: str | Path,
                 output_file: str | Path | None = None,
                 device: str | torch.device | None = None) -> np.ndarray:
    image = np.load(input_file, allow_pickle=False)
    pred = predict_numpy(model, image, device=device)
    if output_file is not None:
        output_file = Path(output_file); output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, pred.astype(np.float32), allow_pickle=False)
    return pred
