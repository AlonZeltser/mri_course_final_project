# MRI Course Final Project

Concise operation guide for the main experiment runner and the baseline notebook.

## What this repo does

- Creates undersampled MRI dataset splits.
- Trains and evaluates a reconstruction model (`ResidualUNet`).
- Produces per-image metrics and report outputs.
- Includes notebook-based baseline assessment and visual analysis.

## Environment setup

Use either Conda (`environment.yml`) or pip (`requirements.txt`).

```powershell
conda env create -f environment.yml
conda activate mri-project
```

Or:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main application (CLI)

Entry point: `experiment_sequence.py`

### Required parameters

- `--source_data_root`: existing source dataset root (read-only input). The folder must be equivalent to the `brain_age`
folder in the course data set: it must contain the files `student_train_metadata.csv` and `student_val_metadata.csv` 
(test csv is not required as the volumes were not given) that describe the existing volume pool, and a subfolder `selected_npy`
which in turns contain all the volumes in `.npy` format.
- `--destination_root`: output parent directory (created if missing) that will hold both the rearranged data set and 
the results. Must be located outside the given dataset and the call must have permission to write. 

The app writes generated dataset splits and results under `--destination_root`.

#### The main application requires quota of at least 12GB for the rearranged dataset and for the results.

### Quick run (smoke)

```powershell
python .\experiment_sequence.py `
  --mode smoke `
  --source_data_root "D:\path\to\source_dataset" `
  --destination_root "D:\path\to\outputs"
```

### Main experiment run

```powershell
python .\experiment_sequence.py `
  --mode main_experiment `
  --source_data_root "D:\path\to\source_dataset" `
  --destination_root "D:\path\to\outputs"
```

### Useful flags

- `--skip_split_creation`
- `--skip_train`
- `--skip_evaluation`
- `--model_path "D:\path\to\checkpoint_root"`

## Notebooks — learning and exploration path

The notebooks follow a progressive journey from raw data understanding through classical baselines to deep learning reconstruction.

### `01_data_exploration.ipynb`
First contact with the dataset. Explores the structure, dimensions, and visual characteristics of the raw MRI brain-age volumes. Visualises sample slices across subjects to build intuition about data variability before any processing is applied.

### `02_kspace_exploration.ipynb`
Dives into the frequency domain. Applies the Fourier transform to MRI slices, visualises k-space magnitude and phase, and demonstrates how different undersampling patterns (e.g. random, centre-weighted) corrupt the spatial image — setting up the core reconstruction problem.

### `03_single_image_baseline.ipynb`
Establishes the simplest possible reconstruction baselines on a single image. Tests zero-filling and basic interpolation approaches directly in k-space, and measures PSNR/SSIM so that every later method has a concrete lower bound to beat.

### `04_small_baseline_assessment.ipynb`
Systematic baseline comparison on a small data subset. Evaluates several classical reconstruction strategies side-by-side with detailed metric analysis and visual inspection. **This is the primary notebook for reproducing baseline results** (see *Notebook operation* below).

### `05_create_dl_small_data_set.ipynb`
Bridges classical and deep-learning work. Generates the train / validation / test dataset splits used by the CNN experiments, writing undersampled images and corresponding k-space masks to disk with configurable retain ratios (0.30 and 0.20).

### `06_first_cnn.ipynb`
Introduces the deep learning model. Trains a `ResidualUNet` on the dataset created in notebook 05, monitors validation metrics across ~65 epochs, and demonstrates that the learned reconstruction clearly outperforms the zero-filling baseline.

### `07_cnn_with_kspace_post_processing.ipynb`
Adds physics-based refinement on top of the CNN output. Applies k-space data-consistency post-processing to enforce that known frequency measurements are preserved, combining learned image features with hard constraints for a further quality boost.

### `08_full_test_set_evaluation.ipynb`
Final end-to-end assessment. Runs the trained model (with and without data-consistency post-processing) on the full 240-sample held-out test set, producing per-image PSNR/SSIM metrics and summary statistics that quantify the complete reconstruction pipeline.

---

## Notebook operation

Primary notebook for baseline alternatives:

- `notebooks/04_small_baseline_assessment.ipynb`

Launch Jupyter:

```powershell
jupyter notebook
```

Then open `notebooks/04_small_baseline_assessment.ipynb` and run cells top-to-bottom.

