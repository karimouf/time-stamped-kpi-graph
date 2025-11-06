#!/bin/bash
#
#SBATCH --job-name=kpi-extract-2models
#SBATCH --output=kpi_extraction_%j.log
#SBATCH --mail-user=karim.ouf@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

################################################################################
# Multi-Model KPI Extraction Job
# 
# This script runs the KPI extraction pipeline using 3 models:
# - Gemma 3 PT 27B
# - DeepSeek V2.5
# - Llama 3 8B Instruct
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

echo "  ✓ CUDA module loaded"
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
        print(f"    - GPU: {torch.cuda.get_device_name(0)}")
        print(f"    - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
    
    import sentencepiece
    print(f"  ✓ SentencePiece installed")
    
    print("\n✓ All dependencies OK!")
    
except Exception as e:
    print(f"\n✗ Dependency error: {e}", file=sys.stderr)
    print("Please install missing dependencies in your conda environment:", file=sys.stderr)
    print("  pip install --no-cache-dir protobuf sentencepiece transformers", file=sys.stderr)
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

# Input/Output paths - using relative paths from script directory
INPUT_FILE="$SCRIPT_DIR/data/tables/linked_tables(2023).jsonl"
OUTPUT_FILE="$SCRIPT_DIR/data/output/kpis_extracted_$(date +%Y%m%d_%H%M%S).jsonl"

# Processing options
MAX_TABLES="5"          # Leave empty to process all tables, or set a number like "5"
MAX_TOKENS=2048         # Maximum tokens to generate per model (increased to handle large tables)
TEMPERATURE=0.1         # Sampling temperature (0.1 = mostly deterministic)

# Model selection (leave empty to use all 3 models)
# Options: gemma-3-27b, deepseek-v2.5, llama-3-8b
MODELS=""               # Empty = use all models

echo "  Input file: $INPUT_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Max tables: ${MAX_TABLES:-All}"
echo "  Max tokens: $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Models: ${MODELS:-All 3 models}"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/data/output"

# ============================================================================
# RUN EXTRACTION
# ============================================================================

echo "[4/5] Running KPI extraction..."
echo ""

# Build command - using relative path to Python script
# Quote file paths to handle special characters like parentheses
CMD="python \"$SCRIPT_DIR/extract_kpis_multi_model.py\" \
    --input \"$INPUT_FILE\" \
    --output \"$OUTPUT_FILE\" \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE"

# Add optional arguments
if [ ! -z "$MAX_TABLES" ]; then
    CMD="$CMD --max-tables $MAX_TABLES"
fi

if [ ! -z "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

# Run the extraction
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
    echo "✓ Extraction completed successfully!"
    echo ""
    
    # Show output file info
    if [ -f "$OUTPUT_FILE" ]; then
        NUM_LINES=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Output file: $OUTPUT_FILE"
        echo "  - Tables processed: $NUM_LINES"
        echo "  - File size: $FILE_SIZE"
        echo ""
        
        # Show sample of first result
        echo "Sample output (first result):"
        echo "---"
        head -1 "$OUTPUT_FILE" | python -m json.tool 2>/dev/null | head -30
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
