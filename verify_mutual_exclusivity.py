"""
Non-destructive verification script to check if existing train, val, and test sets 
use mutually exclusive volumes. Does NOT recreate the dataset.
"""
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.general_utils import prepare_environment, SCV_FILES
from src.create_mri_dataset import SPLIT_VOLUME_ASSIGNMENT

HPC = False
prepare_environment(hpc=HPC)

# Point to existing dataset (do NOT create a new one)
data_sets_root = Path(os.getcwd() + r"/../reconstruction_dataset").resolve()

# Load the split assignment
assignment_csv_path = data_sets_root / SPLIT_VOLUME_ASSIGNMENT
if not assignment_csv_path.exists():
    print(f"ERROR: {assignment_csv_path} not found!")
    print(f"Expected at: {assignment_csv_path}")
    print("\nTo create a dataset, use the notebook 05_create_dl_small_data_set.ipynb instead.")
    sys.exit(1)

assignment_df = pd.read_csv(assignment_csv_path)

print("\n" + "=" * 80)
print("VERIFICATION: MUTUAL EXCLUSIVITY CHECK")
print("=" * 80)

# Get unique volumes per split
train_volumes = set(assignment_df[assignment_df['split'] == 'train']['resolved_volume_path'].unique())
val_volumes = set(assignment_df[assignment_df['split'] == 'val']['resolved_volume_path'].unique())
test_volumes = set(assignment_df[assignment_df['split'] == 'test']['resolved_volume_path'].unique())

print(f"\nTrain set: {len(train_volumes)} unique volumes")
print(f"Val set:   {len(val_volumes)} unique volumes")
print(f"Test set:  {len(test_volumes)} unique volumes")
print(f"Total:     {len(train_volumes) + len(val_volumes) + len(test_volumes)} volumes")

# Check for overlaps
overlap_train_val = train_volumes & val_volumes
overlap_train_test = train_volumes & test_volumes
overlap_val_test = val_volumes & test_volumes
overlap_all_three = train_volumes & val_volumes & test_volumes

print("\n" + "-" * 80)
print("OVERLAP ANALYSIS:")
print("-" * 80)

print(f"Overlap between Train and Val:  {len(overlap_train_val)} volumes")
if overlap_train_val:
    print("  Overlapping volumes:", list(overlap_train_val)[:5])

print(f"Overlap between Train and Test: {len(overlap_train_test)} volumes")
if overlap_train_test:
    print("  Overlapping volumes:", list(overlap_train_test)[:5])

print(f"Overlap between Val and Test:   {len(overlap_val_test)} volumes")
if overlap_val_test:
    print("  Overlapping volumes:", list(overlap_val_test)[:5])

print(f"Overlap in all three sets:      {len(overlap_all_three)} volumes")

# Final verdict
print("\n" + "=" * 80)
if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print("✓ VERDICT: Sets ARE mutually exclusive - NO overlaps detected!")
    print("=" * 80)
else:
    print("✗ VERDICT: Sets are NOT mutually exclusive - overlaps detected!")
    print("=" * 80)
    if overlap_train_val:
        print(f"  Train-Val overlap: {overlap_train_val}")
    if overlap_train_test:
        print(f"  Train-Test overlap: {overlap_train_test}")
    if overlap_val_test:
        print(f"  Val-Test overlap: {overlap_val_test}")

