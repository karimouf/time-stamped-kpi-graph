"""
Validate all model folders in data/output/trial-31 against the management
ground-truth JSONL. Results are written to data/output/trial-31-gt-validation/<model>/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "kpi_extraction_project"))
from validate import validate_folder_against_ground_truth

TRIAL_DIR = Path("data/output/trial-31/output")
GT_FILE = Path("data/output/management_ground-truth.jsonl")
OUTPUT_ROOT = Path("data/output/trial-31-gt-validation")

model_dirs = sorted([d for d in TRIAL_DIR.iterdir() if d.is_dir()])

for model_dir in model_dirs:
    print(f"\n{'#' * 70}")
    print(f"# Model: {model_dir.name}")
    print(f"{'#' * 70}")
    validate_folder_against_ground_truth(
        extraction_dir=model_dir,
        ground_truth_file=GT_FILE,
        output_dir=OUTPUT_ROOT / model_dir.name,
    )
