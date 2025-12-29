#!/bin/bash
#
#SBATCH --job-name=kpi-extraction
#SBATCH --output=kpi_extraction_%j.log
#SBATCH --mail-user=karim.ouf@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=yolo
#SBATCH --qos=yolo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --gres=gpu:a180:2
#SBATCH --time=40:00:00

################################################################################
# Multi-Model KPI Extraction Job
# 
# This script runs the KPI extraction pipeline using:
# - Deepseek distilled Llama 3 70B 
#
# GPU Configuration:
# - 2x A100 80GB GPUs requested
# - DeepSeek uses both GPUs (model parallelism via device_map="auto")
# - Llama uses only 1 GPU (small enough to fit)
#
# Usage: sbatch run_kpi_extraction.sh
################################################################################

echo "=========================================="
echo "Multi-Model KPI Extraction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Set the project directory (absolute path on cluster)
SCRIPT_DIR="/ukp-storage-1/ouf/kpi_extraction_project"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "[1/5] Setting up environment..."

# Set HOME to accessible storage location (fix permission issues)
export HOME=/ukp-storage-1/ouf
echo "  ✓ HOME set to: $HOME"

# Set cache directories to prevent /home/ouf access
export HF_HOME=/ukp-storage-1/ouf/.cache/huggingface
export TRANSFORMERS_CACHE=/ukp-storage-1/ouf/.cache/huggingface
export HF_DATASETS_CACHE=/ukp-storage-1/ouf/.cache/huggingface/datasets
export TORCH_HOME=/ukp-storage-1/ouf/.cache/torch
export XDG_CACHE_HOME=/ukp-storage-1/ouf/.cache

# Create cache directories
mkdir -p /ukp-storage-1/ouf/.cache/huggingface
mkdir -p /ukp-storage-1/ouf/.cache/torch

# Activate conda environment
source /ukp-storage-1/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'test'"
    exit 1
fi

echo "  ✓ Conda environment activated: $CONDA_DEFAULT_ENV"

# Load CUDA module
module purge
module load cuda/12.1

# Set PyTorch memory optimization for fragmented memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "  ✓ CUDA module loaded"
echo "  ✓ PyTorch memory optimization enabled (expandable_segments:True)"
echo ""

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

echo "[2/5] Verifying dependencies..."

python - <<'EOF'
import sys
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"    - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"    - Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
    
    import sentencepiece
    print(f"  ✓ SentencePiece installed")
    
    import bitsandbytes
    print(f"  ✓ BitsAndBytes {bitsandbytes.__version__} (for 4-bit quantization)")
    
    print("\n✓ All dependencies OK!")
    
except Exception as e:
    print(f"\n✗ Dependency error: {e}", file=sys.stderr)
    print("Please install missing dependencies in your conda environment:", file=sys.stderr)
    print("  pip install --no-cache-dir protobuf sentencepiece transformers bitsandbytes accelerate", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Missing dependencies. Install them first."
    exit 1
fi

echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

echo "[3/5] Configuration..."

# Database and Output paths - using relative paths from script directory
DB_PATH="$SCRIPT_DIR/data/pack_context.db"
OUTPUT_DIR="$SCRIPT_DIR/data/output"

# Processing options
YEAR_FILTER=""          # Leave empty to process all years, or set a year like "2019"
MAX_TABLES=""           # Leave empty to process all tables, or set a number like "5"
TEMPERATURE=0.0         # Sampling temperature (0.0 = deterministic)
NO_RESUME=""            # Leave empty to resume from checkpoint, set to "--no-resume" to start fresh

# Model selection (leave empty to use all models)
# Options: deepseek-r1-distill-llama-70b, deepseek-v2.5, llama-3-8b, gemma-3-pt-27b
MODELS=""               # Empty = use all models

echo "  Database: $DB_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Year filter: ${YEAR_FILTER:-All years}"
echo "  Max tables: ${MAX_TABLES:-All}"
echo "  Temperature: $TEMPERATURE"
echo "  Resume from checkpoint: ${NO_RESUME:+No (starting fresh)}"
echo "  Resume from checkpoint: ${NO_RESUME:-Yes (if checkpoint exists)}"
echo "  Models: ${MODELS:-All available models}"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Note: Model processes all tables once (load → process all → unload)"
echo ""

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found: $DB_PATH"
    exit 1
fi

# Get table count from database
echo "  Checking database contents..."
python - <<EOF
import sqlite3
db_path = "$DB_PATH"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Total tables
cursor.execute("SELECT COUNT(*) FROM context_packs")
total = cursor.fetchone()[0]
print(f"    Total tables in database: {total}")

# Tables by year
if "$YEAR_FILTER":
    year_filter = "$YEAR_FILTER"
    cursor.execute("SELECT COUNT(*) FROM context_packs WHERE substr(table_id, 3, 4) = ?", (year_filter,))
    year_count = cursor.fetchone()[0]
    print(f"    Tables for year {year_filter}: {year_count}")

conn.close()
EOF

echo ""

# Create output directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/data/output"

# ============================================================================
# RUN EXTRACTION
# ============================================================================

echo "[4/5] Running KPI extraction..."
echo ""

# Build command for database processing
CMD="python \"$SCRIPT_DIR/extract_kpis.py\" \
    --db \"$DB_PATH\" \
    --output-dir \"$OUTPUT_DIR\" \
    --temperature $TEMPERATURE \
    --job-id $SLURM_JOB_ID"

# Add optional arguments
if [ ! -z "$YEAR_FILTER" ]; then
    CMD="$CMD --year $YEAR_FILTER"
fi

if [ ! -z "$MAX_TABLES" ]; then
    CMD="$CMD --max-tables $MAX_TABLES"
fi

if [ ! -z "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

if [ ! -z "$NO_RESUME" ]; then
    CMD="$CMD $NO_RESUME"
fi

# Run the extraction (database mode with checkpointing)
echo "Command: $CMD"
echo ""
echo "----------------------------------------"
eval $CMD
EXIT_CODE=$?
echo "----------------------------------------"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "[5/5] Job Summary"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Database processing completed!"
    echo ""
    
    # Show output files info
    echo "Output files created:"
    ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | tail -20
    echo ""
    
    # Count total files
    NUM_FILES=$(ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
    echo "Total JSON output files: $NUM_FILES"
    echo ""
    
    # Show sample from first file
    FIRST_FILE=$(ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | head -1)
    if [ -f "$FIRST_FILE" ]; then
        echo "Sample from $(basename $FIRST_FILE) (first 30 lines):"
        echo "---"
        head -30 "$FIRST_FILE"
        echo "..."
    fi
else
    echo "✗ Extraction failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the log file for details:"
    echo "  $SCRIPT_DIR/kpi_extraction_${SLURM_JOB_ID}.log"
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="

exit $EXIT_CODE
