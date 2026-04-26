# Random KPI Extractor - Usage Examples

## Basic Usage

```bash
# Extract 4 KPIs from each table in trial-20 folder
python extract_random_kpis.py --folder data/output/trial-20 --per-table 4 --output my_kpis.json

# Extract 2 KPIs per table with a specific seed for reproducible results
python extract_random_kpis.py --folder data/output/trial-20 --per-table 2 --seed 123 --output reproducible_kpis.json

# Specify custom document name
python extract_random_kpis.py --folder data/output/trial-20 --per-table 3 --document "my-custom-doc" --output custom_kpis.json
```

## Python API Usage

```python
from extract_random_kpis import extract_random_kpis_from_folder

# Extract 5 random KPIs from each file
result = extract_random_kpis_from_folder(
    folder_path="data/output/trial-20",
    kpis_per_table=5,
    output_file="api_extracted_kpis.json",
    document_name="divisions-vw-ar23",
    seed=42  # For reproducible results
)

print(f"Extracted {result['total_kpis_selected']} KPIs from {result['total_files_processed']} files")
```

## Output Format

The script generates a JSON file with this structure:

```json
{
  "source_document": "divisions-vw-ar23",
  "extraction_folder": "trial-20", 
  "extraction_date": "2026-02-10",
  "description": "Random sample of 4 KPIs per table from 25 files",
  "total_files_processed": 24,
  "total_files_skipped": 1,
  "kpis_per_table": 4,
  "total_kpis_selected": 96,
  "random_seed": 42,
  "kpis": [
    {
      "name": "Sales Revenue",
      "key": "Core brand group",
      "country": "Worldwide", 
      "value": 137770,
      "year": 2023,
      "units": "€ million",
      "row_idx": 0,
      "col_idx": 3,
      "source_model": "Qwen2.5-VL-72B-Instruct",
      "source_image": "table_00.png",
      "source": "divisions-vw-ar23/page_003_table_00_kpis.json"
    }
  ]
}
```

## Features

- **Flexible KPI count**: Specify how many KPIs to extract per table
- **Reproducible results**: Use `--seed` for consistent random sampling
- **Error handling**: Skips corrupted files and continues processing
- **Consistent formatting**: Ensures all KPIs have the proper `source` field format
- **Rich metadata**: Includes extraction statistics and parameters
- **Command-line and API**: Use as a script or import as a Python module