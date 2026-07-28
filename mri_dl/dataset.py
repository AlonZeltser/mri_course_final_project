from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MRIUndersampledDataset(Dataset):
    def __init__(self, split_root: str | Path,
                 csv_name: str,
                 plane: str,
                 retain_ratio: float,
                 load_mask: bool) -> None:
        self.split_root = Path(split_root)
        csv_path = self.split_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f'CSV file not found: {csv_path}')
        self.samples = pd.read_csv(csv_path)
        required = {'sample_id','plane','retain_ratio','original_image_file',
                    'undersampled_image_file','mask_file'}
        missing = required - set(self.samples.columns)
        if missing:
            raise ValueError(f'CSV is missing required columns: {sorted(missing)}')
        self.samples = self.samples[self.samples['plane'].str.lower() == plane.lower()]
        keep = self.samples['retain_ratio'].apply(
            lambda x: np.isclose(float(x), float(retain_ratio), atol=1e-8))
        self.samples = self.samples[keep]
        self.samples = self.samples.reset_index(drop=True)
        self.load_mask = load_mask
        if len(self.samples) == 0:
            raise ValueError('No samples remain after applying dataset filters.')

    def __len__(self) -> int:
        return len(self.samples)

    def _load_array(self, relative_path: str) -> np.ndarray:
        path = self.split_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f'Dataset file not found: {path}')
        return np.load(path, allow_pickle=False)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.samples.iloc[index]
        x = self._load_array(str(row['undersampled_image_file'])).astype(np.float32, copy=False)
        y = self._load_array(str(row['original_image_file'])).astype(np.float32, copy=False)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f'Expected 2D images, received input={x.shape}, target={y.shape}')
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch for {row['sample_id']}: {x.shape} vs {y.shape}")
        sample: dict[str, object] = {
            'input': torch.from_numpy(np.ascontiguousarray(x[None, ...])),
            'target': torch.from_numpy(np.ascontiguousarray(y[None, ...])),
            # The persisted dataset remains input/original image pairs.  The
            # residual target is derived here so training learns y - x.
            'delta_target': torch.from_numpy(np.ascontiguousarray((y - x)[None, ...])),
            'sample_id': str(row['sample_id']),
            'plane': str(row['plane']),
            'retain_ratio': torch.tensor(float(row['retain_ratio']), dtype=torch.float32),
        }
        # Attach optional metadata columns when present for logging/reporting.
        optional_string_columns = (
            'subject_id',
            'original_volume_path',
            'resolved_volume_path',
            'mask_id',
            'mask_file',
            'k_space_file',
        )
        for column_name in optional_string_columns:
            if column_name in row.index and not pd.isna(row[column_name]):
                sample[column_name] = str(row[column_name])
        if 'slice_index' in row.index and not pd.isna(row['slice_index']):
            sample['slice_index'] = int(row['slice_index'])
        if self.load_mask:
            mask = self._load_array(str(row['mask_file'])).astype(np.float32, copy=False)
            if mask.ndim != 1:
                raise ValueError(f'Expected 1D row mask, received {mask.shape}')
            sample['mask'] = torch.from_numpy(np.ascontiguousarray(mask))
        return sample
