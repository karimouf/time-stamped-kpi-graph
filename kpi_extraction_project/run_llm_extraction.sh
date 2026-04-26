#!/bin/bash
#
#SBATCH --job-name=llm-text-kpi-extraction
#SBATCH --output=llm_extraction_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-user=karim.ouf@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a180:2
#SBATCH --time=48:00:00

################################################################################
# LLM Text-Based KPI Extraction Job
#
# This script runs KPI extraction from table text (no image inference):
# - Uses tables.json from detect_tables.py
# - Calls extract_kpis_llm.py
# - Can run one or multiple models
#
# Usage:
#   sbatch run_llm_extraction.sh
################################################################################

set -u

# ============================================================================
# CONFIGURATION FLAGS
# ============================================================================

# Set to "true" to use existing tables.json, "false" to rerun table detection
SKIP_TABLE_DETECTION=true

# Optional defaults passed to validation (can be empty)
DEFAULT_YEAR=""
DEFAULT_BUCKET=""

# ============================================================================
# LOG SEPARATOR - Track restarts
# ============================================================================

SCRIPT_DIR="/storage/ukp/work/ouf/kpi_extraction_project"
RUN_COUNTER_FILE="$SCRIPT_DIR/data/output/.llm_run_counter_${SLURM_JOB_ID}"

if [ -f "$RUN_COUNTER_FILE" ]; then
    RUN_NUMBER=$(cat "$RUN_COUNTER_FILE")
    RUN_NUMBER=$((RUN_NUMBER + 1))
else
    RUN_NUMBER=1
fi
echo "$RUN_NUMBER" > "$RUN_COUNTER_FILE"

echo ""
echo "############################################################################"
echo "############################################################################"
echo "##                                                                        ##"
echo "##             LLM TEXT KPI EXTRACTION - RUN #${RUN_NUMBER} - $(date '+%Y-%m-%d %H:%M:%S')          ##"
echo "##                                                                        ##"
echo "############################################################################"
echo "############################################################################"
echo ""
echo "=========================================="
echo "LLM Text KPI Extraction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Run Number: $RUN_NUMBER (restarts: $((RUN_NUMBER - 1)))"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
echo "Working directory: $SCRIPT_DIR"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "[1/6] Setting up environment..."

export HOME=/storage/ukp/work/ouf
export HF_HOME=/storage/ukp/work/ouf/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/ukp/work/ouf/.cache/huggingface
export HF_DATASETS_CACHE=/storage/ukp/work/ouf/.cache/huggingface/datasets
export TORCH_HOME=/storage/ukp/work/ouf/.cache/torch
export XDG_CACHE_HOME=/storage/ukp/work/ouf/.cache

mkdir -p /storage/ukp/work/ouf/.cache/huggingface
mkdir -p /storage/ukp/work/ouf/.cache/torch

echo "  Activating conda environment 'test'..."
source /storage/ukp/work/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'test'"
    exit 1
fi

echo "  ✓ Conda environment activated: $CONDA_DEFAULT_ENV"

source /etc/profile.d/modules.sh || echo "  No module system available"
module load cuda/11.8 || echo "  No CUDA module loaded (expected for PyTorch cu118)"

echo ""

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

echo "[2/6] Verifying dependencies..."

pip install --no-cache-dir pymupdf pillow vllm > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ PyMuPDF, Pillow, and vLLM installed"
else
    echo "  ⚠ Installation had issues (packages may already be installed)"
fi

python - <<'EOF'
import sys
try:
    import torch
    from vllm import LLM, SamplingParams
    import fitz
    from PIL import Image

    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ vLLM available")
    print(f"  ✓ PyMuPDF {fitz.__version__}")
    print(f"  ✓ Pillow installed")

    if torch.cuda.is_available():
        print(f"  ✓ CUDA GPUs: {torch.cuda.device_count()}")
    else:
        print("  ⚠ CUDA not available")

except Exception as e:
    print(f"\n✗ Dependency error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Missing dependencies."
    exit 1
fi

echo ""

# ============================================================================
# TABLE DETECTION (OPTIONAL)
# ============================================================================

DETECTED_TABLES_DIR="$SCRIPT_DIR/data/detected_tables_test"
TABLES_JSON="$DETECTED_TABLES_DIR/tables_management.json"

if [ "$SKIP_TABLE_DETECTION" = "true" ]; then
    echo "[3/6] Skipping table detection (using existing tables.json)..."
    if [ ! -f "$TABLES_JSON" ]; then
        echo "ERROR: SKIP_TABLE_DETECTION=true but tables.json not found: $TABLES_JSON"
        exit 1
    fi
    TABLE_COUNT=$(python -c "import json; data=json.load(open('$TABLES_JSON')); print(len(data.get('tables', [])))")
    echo "  ✓ Using existing tables.json with $TABLE_COUNT table(s)"
else
    echo "[3/6] Running table detection..."

    INPUT_PDF_DIR="$SCRIPT_DIR/data/input/management"
    DETECT_TABLES_SCRIPT="$SCRIPT_DIR/detect_tables.py"

    if [ ! -d "$INPUT_PDF_DIR" ]; then
        echo "ERROR: Input PDF directory not found: $INPUT_PDF_DIR"
        exit 1
    fi

    python "$DETECT_TABLES_SCRIPT" \
        "$INPUT_PDF_DIR" \
        "$TABLES_JSON" \
        --output-dir "$DETECTED_TABLES_DIR" \
        --dpi 300 \
        --model-name deepseek-ocr

    if [ $? -ne 0 ]; then
        echo "ERROR: Table detection failed"
        exit 1
    fi
fi

echo ""

# ============================================================================
# VLLM RUNTIME CONFIG
# ============================================================================

echo "[4/6] Configuring vLLM runtime..."

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "  Detected $NUM_GPUS GPU(s)"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

VLLM_COMPILE_CACHE="$HOME/.cache/vllm/torch_compile_cache"
if [ -d "$VLLM_COMPILE_CACHE" ]; then
    echo "  Clearing stale vLLM compile cache: $VLLM_COMPILE_CACHE"
    rm -rf "$VLLM_COMPILE_CACHE"
fi

echo ""

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

echo "[5/6] LLM configuration..."

TEMPERATURE=0.0
DB_PATH="$SCRIPT_DIR/data/pack_context.db"

# Keep model names aligned with MODEL_CONFIGS in model.py
declare -a MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL-32B-Instruct"
    "Qwen2.5-VL-72B-Instruct"
)

declare -a OUTPUT_DIRS=(
    "$SCRIPT_DIR/data/output/llm_qwen_7b"
    "$SCRIPT_DIR/data/output/llm_qwen_32b"
    "$SCRIPT_DIR/data/output/llm_qwen_72b"
)

if [ ! -f "$TABLES_JSON" ]; then
    echo "ERROR: Tables JSON not found: $TABLES_JSON"
    exit 1
fi

TABLE_COUNT=$(python - <<EOF
import json
try:
    with open("$TABLES_JSON", "r") as f:
        data = json.load(f)
    print(len(data.get("tables", [])))
except Exception:
    print(0)
EOF
)

if [ "$TABLE_COUNT" -eq 0 ]; then
    echo "ERROR: No tables found in $TABLES_JSON"
    exit 1
fi

echo "  Tables JSON: $TABLES_JSON"
echo "  Table count: $TABLE_COUNT"
echo "  Temperature: $TEMPERATURE"
echo "  Database path: $DB_PATH"
echo ""

for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    mkdir -p "$OUTPUT_DIR"
done

# ============================================================================
# RUN LLM TEXT EXTRACTION
# ============================================================================

echo "[6/6] Running LLM text KPI extraction..."
echo ""

OVERALL_EXIT_CODE=0

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"

    echo "=========================================="
    echo "Processing with model: $MODEL_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="

    CMD="python \"$SCRIPT_DIR/extract_kpis_llm.py\" \
        --tables-json \"$TABLES_JSON\" \
        --output-dir \"$OUTPUT_DIR\" \
        --temperature $TEMPERATURE \
        --model-name \"$MODEL_NAME\" \
        --db-path \"$DB_PATH\""

    if [ -n "$DEFAULT_YEAR" ]; then
        CMD="$CMD --year $DEFAULT_YEAR"
    fi

    if [ -n "$DEFAULT_BUCKET" ]; then
        CMD="$CMD --bucket $DEFAULT_BUCKET"
    fi

    echo "Command: $CMD"
    echo "----------------------------------------"
    eval $CMD
    EXIT_CODE=$?
    echo "----------------------------------------"
    echo ""

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Model $MODEL_NAME completed successfully"
    else
        echo "✗ Model $MODEL_NAME failed with exit code: $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
    fi

done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=========================================="
if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo "✓ LLM text extraction completed for all models"
    for i in "${!MODELS[@]}"; do
        MODEL_NAME="${MODELS[$i]}"
        OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
        echo ""
        echo "Output files for $MODEL_NAME:"
        echo "Directory: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | tail -10
    done

    if [ -f "$RUN_COUNTER_FILE" ]; then
        rm "$RUN_COUNTER_FILE"
    fi
else
    echo "✗ LLM text extraction failed for at least one model (exit=$OVERALL_EXIT_CODE)"
    echo "Check: $SCRIPT_DIR/llm_extraction_${SLURM_JOB_ID}.log"
fi

echo "Run #${RUN_NUMBER} finished at: $(date)"
echo "Runtime: $SECONDS seconds"
echo "=========================================="

exit $OVERALL_EXIT_CODE
