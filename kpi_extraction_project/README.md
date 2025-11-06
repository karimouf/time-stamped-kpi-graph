# KPI Extraction Project

Complete self-contained project for extracting KPIs from financial tables using 3 LLMs.

## Directory Structure

```
kpi_extraction_project/
├── extract_kpis_multi_model.py    # Main Python script
├── run_kpi_extraction.sh          # SLURM job script
├── README.md                      # This file
├── data/
│   ├── tables/                    # Input data
│   │   ├── linked_tables(2022).jsonl
│   │   └── linked_tables(2023).jsonl
│   └── output/                    # Results will be saved here
└── (logs will be created here)
```

## Models Used

1. **Gemma 3 PT 27B** - Google's large instruction model
2. **DeepSeek V2.5** - Advanced reasoning
3. **Llama 3 8B Instruct** - Meta's instruction model

All models loaded from cluster's shared storage: `/storage/ukp/shared/shared_model_weights/`

## Quick Setup

### 1. Upload Entire Project Directory to Cluster

```bash
# From your local machine, upload the entire directory
scp -r kpi_extraction_project ouf@slurm.ukp.informatik.tu-darmstadt.de:/ukp-storage-1/ouf/
```

### 2. Install Dependencies (One-time setup)

```bash
# SSH to cluster
ssh slurm.ukp.informatik.tu-darmstadt.de

# Start interactive session
srun --pty bash -i

# Activate conda environment
source /ukp-storage-1/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test

# Install dependencies
pip install --no-cache-dir protobuf sentencepiece transformers accelerate

# Verify
python -c "import torch, transformers, sentencepiece, google.protobuf; print('✓ All OK')"

# Exit
exit
```

### 3. Run the Extraction

```bash
# Navigate to project directory
cd /ukp-storage-1/ouf/kpi_extraction_project

# Make script executable
chmod +x run_kpi_extraction.sh

# For testing: Edit to process only 5 tables
nano run_kpi_extraction.sh
# Change line: MAX_TABLES=""  to  MAX_TABLES="5"

# Submit job
sbatch run_kpi_extraction.sh

# Monitor
squeue --me
tail -f kpi_extraction_*.log
```

## Configuration

Edit `run_kpi_extraction.sh` to customize:

```bash
# Line ~115: Input file
INPUT_FILE="$SCRIPT_DIR/data/tables/linked_tables(2023).jsonl"

# Line ~119: Limit number of tables (for testing)
MAX_TABLES="5"          # Set to "" for all tables

# Line ~124: Select specific models
MODELS="llama-3-8b"     # Or "" for all 3 models
```

## Output

Results are saved to `data/output/kpis_extracted_TIMESTAMP.jsonl`

Each line contains:
```json
{
  "table_id": "VW2023_Td85de9",
  "models_used": ["gemma-3-27b", "deepseek-v2.5", "llama-3-8b"],
  "all_kpis": [
    {
      "metric_name": "Revenue",
      "value": "322284",
      "unit": "€ million",
      "time_period": "2023",
      "category": "financial",
      "source_model": "gemma-3-27b"
    }
  ],
  "total_kpis_extracted": 41
}
```

## Command Line Usage

You can also run the Python script directly:

```bash
python extract_kpis_multi_model.py \
    --input data/tables/linked_tables\(2023\).jsonl \
    --output data/output/my_results.jsonl \
    --models llama-3-8b \
    --max-tables 5
```

## Resource Requirements

| Models | Memory | GPU | Time/Table |
|--------|--------|-----|------------|
| 1 model | 32GB | 1x GPU | 30-60s |
| 2 models | 48GB | 1x GPU | 1-2 min |
| 3 models | 80GB | 1x GPU | 2-4 min |

## Troubleshooting

### Missing Dependencies
```bash
pip install --no-cache-dir protobuf sentencepiece transformers
```

### Out of Memory
Edit `run_kpi_extraction.sh`:
- Change `#SBATCH --mem=80GB` to `#SBATCH --mem=96GB`
- Or use fewer models: `MODELS="llama-3-8b"`

### Check Model Paths
```bash
ls /storage/ukp/shared/shared_model_weights/ | grep -E "gemma|deepseek|llama"
```

## Support

- Script documentation: Read comments in `extract_kpis_multi_model.py`
- SLURM help: https://wiki.ukp.informatik.tu-darmstadt.de/Services/Campus/ComputeCluster
- Job logs: Check `kpi_extraction_<JOBID>.log` in this directory
