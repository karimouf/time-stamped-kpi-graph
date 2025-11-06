# Multi-Model KPI Extraction - Quick Guide

## Overview
Extract KPIs from financial tables using 3 state-of-the-art models:
- **Gemma 3 PT 27B** - Google's large instruction model
- **DeepSeek V2.5** - Advanced reasoning
- **Llama 3 8B Instruct** - Meta's instruction model (handles prompt echoing)

## Files
- `extract_kpis_multi_model.py` - Main Python script (well-documented)
- `run_kpi_extraction.sh` - SLURM job script

## Quick Start

### 1. Upload to Cluster
```bash
# From your local machine
scp extract_kpis_multi_model.py ouf@slurm.ukp.informatik.tu-darmstadt.de:/ukp-storage-1/ouf/
scp run_kpi_extraction.sh ouf@slurm.ukp.informatik.tu-darmstadt.de:/ukp-storage-1/ouf/
```

### 2. Setup on Cluster
```bash
# SSH to cluster
ssh slurm.ukp.informatik.tu-darmstadt.de

# Start interactive session
srun --pty --gres=gpu:1 --mem=32G bash -i

# Activate environment
source /ukp-storage-1/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test

# Install dependencies
pip install --no-cache-dir protobuf sentencepiece transformers

# Verify
python -c "import torch, transformers, sentencepiece, google.protobuf; print('✓ OK')"
```

### 3. Quick Test
```bash
# Test with 2 tables using only Llama
python extract_kpis_multi_model.py \
    --input data/tables/linked_tables\(2023\).jsonl \
    --output test_output.jsonl \
    --models llama-3-8b \
    --max-tables 2
```

### 4. Submit Full Job
```bash
# Edit run_kpi_extraction.sh to set your paths
nano run_kpi_extraction.sh

# Make executable
chmod +x run_kpi_extraction.sh

# Submit
sbatch run_kpi_extraction.sh

# Monitor
squeue --me
tail -f kpi_extraction_*.log
```

## Command Line Options

```bash
python extract_kpis_multi_model.py \
    --input INPUT.jsonl \              # Required: input file
    --output OUTPUT.jsonl \             # Required: output file
    --models gemma-3-27b llama-3-8b \  # Optional: specific models
    --max-tables 10 \                   # Optional: limit tables
    --max-tokens 1024 \                 # Optional: tokens per model
    --temperature 0.1 \                 # Optional: sampling temp
    --debug                             # Optional: verbose logging
```

## Model Selection

**Use all 3 models (default):**
```bash
python extract_kpis_multi_model.py --input in.jsonl --output out.jsonl
```

**Use specific model(s):**
```bash
# Just Llama
python extract_kpis_multi_model.py --input in.jsonl --output out.jsonl --models llama-3-8b

# Gemma and DeepSeek
python extract_kpis_multi_model.py --input in.jsonl --output out.jsonl --models gemma-3-27b deepseek-v2.5
```

## Output Format

Each line in the output JSONL contains:
```json
{
  "table_id": "VW2023_Td85de9",
  "doc_id": "VW2023",
  "year": 2023,
  "models_used": ["gemma-3-27b", "deepseek-v2.5", "llama-3-8b"],
  "individual_results": {
    "gemma-3-27b": { "kpis": [...], "num_kpis": 15 },
    "deepseek-v2.5": { "kpis": [...], "num_kpis": 12 },
    "llama-3-8b": { "kpis": [...], "num_kpis": 14 }
  },
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

## Resource Requirements

| Models | Memory | GPU | Time/Table |
|--------|--------|-----|------------|
| 1 model | 32GB | 1x GPU | 30-60s |
| 2 models | 48GB | 1x GPU | 1-2 min |
| 3 models | 80GB | 1x GPU | 2-4 min |

## Troubleshooting

### Out of Memory
- Use fewer models: `--models llama-3-8b`
- Process fewer tables: `--max-tables 10`
- Request more memory in SBATCH: `#SBATCH --mem=96GB`

### Models Not Found
Check available models:
```bash
ls /storage/ukp/shared/shared_model_weights/ | grep -E "gemma|deepseek|llama"
```

### Verify Dependencies
```bash
python - <<'EOF'
import torch, transformers, sentencepiece, google.protobuf
print("✓ All dependencies OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
EOF
```

### View Results
```bash
# Pretty print first result
head -1 output.jsonl | python -m json.tool

# Count total KPIs extracted
cat output.jsonl | jq '.total_kpis_extracted' | awk '{s+=$1} END {print s}'

# List unique metric names
cat output.jsonl | jq -r '.all_kpis[].metric_name' | sort | uniq
```

## Example Workflow

```bash
# 1. Interactive test
srun --pty --gres=gpu:1 --mem=32G bash -i
source /ukp-storage-1/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test

# 2. Quick test with 2 tables, 1 model
python extract_kpis_multi_model.py \
    --input data/tables/linked_tables\(2023\).jsonl \
    --output test.jsonl \
    --models llama-3-8b \
    --max-tables 2

# 3. Check results
cat test.jsonl | python -m json.tool | head -50

# 4. If OK, exit and submit full job
exit
sbatch run_kpi_extraction.sh

# 5. Monitor
watch -n 10 'squeue --me'
```

## Notes

- **Llama 3** includes the prompt in its output - the script handles this automatically
- **Gemma 3** and **DeepSeek** don't echo prompts
- Results are written line-by-line (safe if job crashes)
- All models run on the same table for ensemble results
- Lower temperature (0.1) = more consistent results
- Higher temperature (0.7-0.9) = more creative but less consistent

## Support

- Script documentation: Read comments in `extract_kpis_multi_model.py`
- SLURM help: https://wiki.ukp.informatik.tu-darmstadt.de/Services/Campus/ComputeCluster
- Job logs: `/ukp-storage-1/ouf/kpi_extraction_<JOBID>.log`
