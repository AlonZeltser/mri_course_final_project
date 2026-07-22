# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from mri_dl import MRIUndersampledDataset, ResidualUNet, TrainConfig, load_checkpoint, predict_tensor, train_model

# %% Paths
DATASET_ROOT = Path('path/to/reconstruction_dataset')
CHECKPOINT_PATH = Path('checkpoints/axial_r30_best.pt')

# %% Datasets: begin with one plane and one retain ratio
train_dataset = MRIUndersampledDataset(DATASET_ROOT / 'train', planes=('Axial',), retain_ratios=(0.30,))
val_dataset = MRIUndersampledDataset(DATASET_ROOT / 'val', planes=('Axial',), retain_ratios=(0.30,))
test_dataset = MRIUndersampledDataset(DATASET_ROOT / 'test', planes=('Axial',), retain_ratios=(0.30,))
print(f'train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}')

# %% Inspect one sample
sample = train_dataset[0]
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample['input'][0], cmap='gray'); axes[0].set_title('Undersampled input'); axes[0].axis('off')
axes[1].imshow(sample['target'][0], cmap='gray'); axes[1].set_title('Original target'); axes[1].axis('off')
plt.tight_layout(); plt.show()

# %% DataLoaders
# num_workers=0 is safest in a Windows notebook. Raise it later on Linux/HPC.
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

# %% Model and training
model_kwargs = {'in_channels': 1, 'out_channels': 1, 'base_channels': 16}
model = ResidualUNet(**model_kwargs)
config = TrainConfig(epochs=10, learning_rate=1e-3, checkpoint_path=str(CHECKPOINT_PATH), seed=42)
history = train_model(model, train_loader, val_loader, config, model_kwargs)

# %% Training curves
plt.figure(figsize=(7, 4))
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch'); plt.ylabel('L1 loss'); plt.legend(); plt.tight_layout(); plt.show()

# %% Load best saved model and infer one sample
loaded_model, checkpoint = load_checkpoint(CHECKPOINT_PATH, ResidualUNet)
test_sample = test_dataset[0]
prediction = predict_tensor(loaded_model, test_sample['input'])[0, 0]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(test_sample['input'][0], cmap='gray'); axes[0].set_title('Undersampled'); axes[0].axis('off')
axes[1].imshow(prediction, cmap='gray'); axes[1].set_title('DL reconstruction'); axes[1].axis('off')
axes[2].imshow(test_sample['target'][0], cmap='gray'); axes[2].set_title('Original'); axes[2].axis('off')
plt.tight_layout(); plt.show()
