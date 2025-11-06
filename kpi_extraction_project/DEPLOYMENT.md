# DEPLOYMENT GUIDE

## What You Have

A complete, self-contained project directory with:
- ✅ Python script (`extract_kpis_multi_model.py`)
- ✅ SLURM job script (`run_kpi_extraction.sh`) 
- ✅ Input data tables (2022 & 2023)
- ✅ All paths correctly configured (relative paths)
- ✅ Output directory structure ready

## Single Command Upload

### On Windows (PowerShell):
```powershell
.\upload_to_cluster.ps1
```

### On Linux/Mac:
```bash
bash upload_to_cluster.sh
```

This uploads the **entire directory** to the cluster at `/ukp-storage-1/ouf/kpi_extraction_project/`

## After Upload - 3 Simple Steps

### Step 1: Install Dependencies (One-time)
```bash
ssh ouf@slurm.ukp.informatik.tu-darmstadt.de
srun --pty bash -i
source /ukp-storage-1/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test
pip install --no-cache-dir protobuf sentencepiece transformers accelerate
python -c "import torch, transformers, sentencepiece, google.protobuf; print('✓ OK')"
exit
```

### Step 2: Test Run (5 tables)
```bash
cd /ukp-storage-1/ouf/kpi_extraction_project
chmod +x run_kpi_extraction.sh
nano run_kpi_extraction.sh  # Change MAX_TABLES="" to MAX_TABLES="5"
sbatch run_kpi_extraction.sh
squeue --me
```

### Step 3: Full Run (All tables)
```bash
nano run_kpi_extraction.sh  # Change MAX_TABLES="5" back to MAX_TABLES=""
sbatch run_kpi_extraction.sh
tail -f kpi_extraction_*.log
```

## Results

Check `data/output/` for results files:
```bash
ls -lh data/output/
cat data/output/kpis_extracted_*.jsonl | python -m json.tool | head -50
```

## Everything is Self-Contained

✅ **No absolute paths** - script finds its own location
✅ **Data included** - tables are in `data/tables/`
✅ **Output organized** - results go to `data/output/`
✅ **Logs in place** - job logs saved in project root

## File Locations After Upload

```
/ukp-storage-1/ouf/kpi_extraction_project/
├── extract_kpis_multi_model.py    ← Python script
├── run_kpi_extraction.sh          ← SLURM script
├── README.md                      ← Full documentation
├── DEPLOYMENT.md                  ← This file
├── data/
│   ├── tables/
│   │   ├── linked_tables(2022).jsonl   ← Input data
│   │   └── linked_tables(2023).jsonl   ← Input data
│   └── output/                         ← Results appear here
└── kpi_extraction_*.log                ← Job logs appear here
```

## Quick Commands Reference

```bash
# Upload (from local Windows)
.\upload_to_cluster.ps1

# SSH to cluster
ssh ouf@slurm.ukp.informatik.tu-darmstadt.de

# Navigate to project
cd /ukp-storage-1/ouf/kpi_extraction_project

# Submit job
sbatch run_kpi_extraction.sh

# Check status
squeue --me

# View log
tail -f kpi_extraction_*.log

# View results
ls data/output/
cat data/output/kpis_extracted_*.jsonl | python -m json.tool | less
```

## That's It!

The project is completely portable and self-contained. All paths are relative to the project directory.
