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

## Notebook operation

Primary notebook for baseline alternatives:

- `notebooks/04_small_baseline_assessment.ipynb`

Launch Jupyter:

```powershell
jupyter notebook
```

Then open `notebooks/04_small_baseline_assessment.ipynb` and run cells top-to-bottom.

