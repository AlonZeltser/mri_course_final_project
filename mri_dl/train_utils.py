from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import random
from typing import Any, TYPE_CHECKING
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .experiment_config import ModelConfig


@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    checkpoint_path: str
    seed: int



def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x = batch['input'].to(device, non_blocking=True)
            delta_target = batch['delta_target'].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predicted_delta = model(x)
            loss = criterion(predicted_delta, delta_target)
            if training:
                loss.backward(); optimizer.step()
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
    if count == 0:
        raise ValueError('DataLoader produced no samples.')
    return total / count


def save_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int,
                    validation_loss: float, model_kwargs: dict[str, Any],
                    train_config: TrainConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoint to {}".format(path))
    torch.save({
        'epoch': epoch,
        'validation_loss': validation_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_kwargs': model_kwargs,
        'train_config': asdict(train_config),
    }, path)


def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, train_config: TrainConfig,
                model_kwargs: dict[str, Any],
                experiment_config: ModelConfig) -> dict[str, list[float]]:
    experiment_config.save()
    set_seed(train_config.seed)
    device = choose_device()
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate,
                                 weight_decay=train_config.weight_decay)
    history = {'train_loss': [], 'val_loss': []}
    best = float('inf')
    print(f'Training on device: {device}')
    print(f"Training model. start: {datetime.now().strftime('%H:%M:%S')}")
    for epoch in range(1, train_config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        improved = val_loss < best
        print(f'Epoch {epoch:03d}/{train_config.epochs:03d} | train={train_loss:.6f} | val={val_loss:.6f}' + (' *' if improved else ''))
        if improved:
            best = val_loss
            save_checkpoint( train_config.checkpoint_path, model, optimizer, epoch,
                            val_loss, model_kwargs, train_config)
    print(f"Training model. end: {datetime.now().strftime('%H:%M:%S')}")
    return history


def load_checkpoint(checkpoint_path: str | Path, model_class: type[nn.Module]):
    device = choose_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = model_class(**checkpoint.get('model_kwargs', {}))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint
