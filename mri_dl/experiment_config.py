"""Small, reusable configuration for one MRI reconstruction experiment."""

from __future__ import annotations
from .train_utils import TrainConfig
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from src.general_utils import BRAIN_PLANES

RETAIN_RATIOS = (0.20, 0.30, 0.50)

@dataclass(frozen=True)
class ModelConfig:
    plane: str
    retain_ratio: float
    epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int
    num_workers: int
    device: str | None
    data_consistency_enabled: bool
    per_image_csv_logging: bool
    data_root: Path = Path("..").joinpath("reconstruction_dataset")
    model_name: str = "ResidualUNet"
    base_channels: int = 16
    input_channels: int = 1
    output_channels: int = 1
    prediction_type: str = "residual"

    # Evaluation and output policy.
    psnr_data_range: str = "reference_min_max"
    ssim_data_range: str = "reference_min_max"
    result_root: Path = Path("..").joinpath("results")

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane", self.plane.lower())
        self.validate()



    @property
    def test_results_path(self) -> Path:
        return self.result_dir / "test_per_image.csv"

    @property
    def history_path(self) -> Path:
        return self.result_dir / "training_history.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.result_dir / "best_model.pt"

    @property
    def experiment_relative_path(self) -> str:
        return f"{self.plane}/retain_{int(round(self.retain_ratio * 100))}"

    @property
    def result_dir(self) -> Path:
        return Path(self.result_root) / self.experiment_relative_path

    @property
    def train_data_root(self) -> Path:
        return Path(self.data_root) / "train"

    @property
    def validation_data_root(self) -> Path:
        return Path(self.data_root) / "val"

    @property
    def test_data_root(self) -> Path:
        return Path(self.data_root) / "test"

    @property
    def model_kwargs(self) -> dict[str, int]:
        return {
            "in_channels": self.input_channels,
            "out_channels": self.output_channels,
            "base_channels": self.base_channels,
        }


    def make_train_config(self) -> Any:
        return TrainConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            checkpoint_path=str(self.checkpoint_path),
            seed=self.random_seed,
            device=self.device,
        )

    def validate(self, check_data_paths: bool = False) -> None:
        if self.plane not in tuple(key.lower() for key in BRAIN_PLANES.keys()):
            raise ValueError(f"plane must be one of {BRAIN_PLANES.keys()}; got {self.plane!r}")
        if not any(abs(float(self.retain_ratio) - ratio) < 1e-8 for ratio in RETAIN_RATIOS):
            raise ValueError(f"retain_ratio must be one of {RETAIN_RATIOS}; got {self.retain_ratio!r}")
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("base_channels", self.base_channels),
            ("input_channels", self.input_channels),
            ("output_channels", self.output_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; got {self.learning_rate!r}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers cannot be negative; got {self.num_workers!r}")
        if check_data_paths:
            for split, path in (("train", self.train_data_root), ("validation", self.validation_data_root), ("test", self.test_data_root)):
                if not path.is_dir():
                    raise FileNotFoundError(f"{split} data directory not found: {path}")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe resolved settings, including smoke/full overrides."""
        values = asdict(self)
        values.update({
            "experiment_name": self.experiment_relative_path,
            "result_dir": str(self.result_dir),
            "train_data_root": str(self.train_data_root),
            "validation_data_root": str(self.validation_data_root),
            "test_data_root": str(self.test_data_root),
        })
        values["data_root"] = str(self.data_root)
        values["result_root"] = str(self.result_root)
        return values

    def save(self) -> Path:
        """Save resolved settings; by default use ``result_dir/config_used.json``."""
        output_path = self.result_dir / "config_used.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output_path

