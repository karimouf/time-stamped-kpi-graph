#!/bin/bash
#
#SBATCH --job-name=vlm-kpi-extraction
#SBATCH --output=vlm_extraction_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-user=karim.ouf@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:a180:2
#SBATCH --time=48:00:00

################################################################################
# VLM-Based KPI Extraction Job
# 
# This script runs KPI extraction from table images using Vision-Language Model:
# - Qwen3-VL-30B-A3B-Instruct (Multimodal model with 30B parameters)
# - Processes tables.json from detect_tables.py with table images and page markdown
#
# OCR Support:
# - DeepSeek OCR: GPU-based, high accuracy, requires transformers
#
# GPU Configuration:
# - 2x A180 GPUs requested (VLM requires significant VRAM)
# - Model uses device_map="auto" for automatic distribution
# - PyTorch cu118 bundles its own CUDA runtime (do NOT load system CUDA modules)
#
# Input: tables.json file from detect_tables.py (contains table images + page markdown)
# Output: KPI JSON files
#
# Log Handling:
# - Uses --open-mode=append to preserve logs across job restarts
# - Each run is clearly separated with timestamps
# - Run counter tracks number of restarts
#
# Usage: sbatch run_vlm_extraction.sh
################################################################################

# ============================================================================
# CONFIGURATION FLAGS
# ============================================================================

# Skip table detection if tables.json already exists (set to true to skip)
# Set to "true" to use existing tables.json, "false" to rerun table detection
SKIP_TABLE_DETECTION=true

# ============================================================================
# LOG SEPARATOR - Track restarts
# ============================================================================

# Set the project directory early for run counter
SCRIPT_DIR="/storage/ukp/work/ouf/kpi_extraction_project"
RUN_COUNTER_FILE="$SCRIPT_DIR/data/output/.vlm_run_counter_${SLURM_JOB_ID}"

# Increment and get run number
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
echo "##              VLM KPI EXTRACTION - RUN #${RUN_NUMBER} - $(date '+%Y-%m-%d %H:%M:%S')            ##"
echo "##                                                                        ##"
echo "############################################################################"
echo "############################################################################"
echo ""
echo "=========================================="
echo "Vision-Language Model KPI Extraction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Run Number: $RUN_NUMBER (restarts: $((RUN_NUMBER - 1)))"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Set the project directory (absolute path on cluster)
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# ============================================================================
# ENVIRONMENT SETUP FOR DEEPSEEK OCR
# ============================================================================

echo "[1/6] Setting up DeepSeek OCR environment..."

# Set HOME to accessible storage location (fix permission issues)
export HOME=/storage/ukp/work/ouf
echo "  ✓ HOME set to: $HOME"

# Set cache directories to prevent /home/ouf access
export HF_HOME=/storage/ukp/work/ouf/.cache/huggingface
export TRANSFORMERS_CACHE=/storage/ukp/work/ouf/.cache/huggingface
export HF_DATASETS_CACHE=/storage/ukp/work/ouf/.cache/huggingface/datasets
export TORCH_HOME=/storage/ukp/work/ouf/.cache/torch
export XDG_CACHE_HOME=/storage/ukp/work/ouf/.cache

# Create cache directories
mkdir -p /storage/ukp/work/ouf/.cache/huggingface
mkdir -p /storage/ukp/work/ouf/.cache/torch

# Activate conda environment
echo "  Activating conda environment 'test'..."
# Activate conda environment
source /storage/ukp/work/ouf/miniconda3/etc/profile.d/conda.sh
conda activate test

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'test'"
    echo "Please ensure conda is properly installed and initialized"
    echo "Run: conda init --all && source ~/.bashrc"
    exit 1
fi

echo "  ✓ Conda environment activated: $CONDA_DEFAULT_ENV"

# # Install Tesseract OCR and Python packages for OCR
# echo "  Installing Tesseract OCR and dependencies..."

# # Try to install tesseract via conda first
# echo "  → Attempting conda installation of Tesseract..."
# conda install -c conda-forge tesseract --yes --quiet 2>/dev/null
# if [ $? -eq 0 ]; then
#     echo "  ✓ Tesseract installed via conda"
# else
#     echo "  → Conda installation failed, checking system modules..."
#     # Try to load tesseract module if available
#     module avail tesseract 2>/dev/null && module load tesseract 2>/dev/null
#     if [ $? -eq 0 ]; then
#         echo "  ✓ Tesseract loaded via system module"
#     else
#         echo "  → No system module found, checking if already installed..."
#     fi
# fi

# # Install Python dependencies
# pip install --quiet pytesseract opencv-python-headless pillow numpy pandas

# # Check if tesseract is available
# if command -v tesseract >/dev/null 2>&1; then
#     TESSERACT_VERSION=$(tesseract --version 2>&1 | head -n1)
#     echo "  ✓ Tesseract OCR available: $TESSERACT_VERSION"
#     OCR_AVAILABLE="tesseract"
# else
#     echo "  ⚠ Tesseract OCR not found in system"
#     echo "  → Falling back to DeepSeek OCR (requires transformers)"
#     # Install transformers for DeepSeek OCR fallback
#     echo "  Installing transformers 4.46.3 for DeepSeek OCR fallback..."
#     pip install --quiet transformers==4.46.3
#     echo "  ✓ DeepSeek OCR fallback installed"
#     OCR_AVAILABLE="deepseek-ocr"
# fi

echo "  Checking module system..."
source /etc/profile.d/modules.sh || echo "  No module system available (not all clusters have this)"

module load cuda/11.8 || echo "  No CUDA module loaded (expected for PyTorch cu118)"

echo "  Listing CUDA modules before load..."
module list cuda || echo "  No modules loaded (not all clusters have this)"


# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

echo "[2/6] Verifying and installing dependencies..."

# Install required packages
echo "  Installing required libraries..."
# DeepSeek OCR currently expects LlamaFlashAttention2 from transformers.
# Pin to a known-compatible version only when table detection will run.
if [ "$SKIP_TABLE_DETECTION" = "true" ]; then
    export REQUIRE_DEEPSEEK_TRANSFORMERS=0
else
    export REQUIRE_DEEPSEEK_TRANSFORMERS=1
    pip install --no-cache-dir transformers==4.46.3 pymupdf pillow > /dev/null 2>&1
fi
if [ $? -eq 0 ]; then
    echo "  ✓ Transformers, PyMuPDF, Pillow"
else
    echo "  ⚠ Installation had issues (packages may already be installed)"
fi

python - <<'EOF'
import sys
import os
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

    try:
        from vllm import LLM, SamplingParams
        import vllm
        print(f"  ✓ vLLM {vllm.__version__}")
    except ImportError:
        print(f"  ✗ vLLM not found - REQUIRED for model inference")
        print(f"    Install with: pip install vllm")
        sys.exit(1)

    from PIL import Image
    print(f"  ✓ Pillow (PIL) installed")

    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")

    # DeepSeek OCR compatibility check (only required when table detection runs)
    if os.environ.get("REQUIRE_DEEPSEEK_TRANSFORMERS") == "1":
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        print(f"  ✓ LlamaFlashAttention2 available")

    # Verify PyMuPDF (fitz) is installed
    import fitz
    print(f"  ✓ PyMuPDF (fitz) {fitz.__version__}")

    print("\n✓ All dependencies OK!")

except Exception as e:
    print(f"\n✗ Dependency error: {e}", file=sys.stderr)
    print("Please install missing dependencies in your conda environment:", file=sys.stderr)
    if os.environ.get("REQUIRE_DEEPSEEK_TRANSFORMERS") == "1":
        print("  pip install --no-cache-dir transformers==4.46.3 pymupdf pillow vllm", file=sys.stderr)
    else:
        print("  pip install --no-cache-dir transformers pymupdf pillow vllm", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Missing dependencies. Install them first."
    exit 1
fi

echo ""

# ============================================================================
# TABLE DETECTION WITH DEEPSEEK OCR
# ============================================================================

# Output paths for table detection
DETECTED_TABLES_DIR="$SCRIPT_DIR/data/detected_tables_test"
TABLES_JSON="$DETECTED_TABLES_DIR/tables_management.json"

if [ "$SKIP_TABLE_DETECTION" = "true" ]; then
    echo "[3/6] Skipping table detection (using existing tables.json)..."
    
    # Verify tables.json exists
    if [ ! -f "$TABLES_JSON" ]; then
        echo "ERROR: SKIP_TABLE_DETECTION=true but tables.json not found at: $TABLES_JSON"
        echo "Set SKIP_TABLE_DETECTION=false to run table detection, or ensure tables.json exists"
        exit 1
    fi
    
    # Count tables in existing JSON
    TABLE_COUNT=$(python -c "import json; data=json.load(open('$TABLES_JSON')); print(len(data.get('tables', [])))")
    echo "  ✓ Using existing tables.json"
    echo "  Found $TABLE_COUNT table(s) in tables.json"
    echo ""
else
    echo "[3/6] Detecting tables in PDF with DeepSeek OCR..."

    # Input PDF directory path
    INPUT_PDF_DIR="$SCRIPT_DIR/data/input"  # Directory containing PDF files

    # Check if input directory exists
    if [ ! -d "$INPUT_PDF_DIR" ]; then
        echo "ERROR: Input directory not found: $INPUT_PDF_DIR"
        echo ""
        echo "Please create the directory and place your PDF files at: $INPUT_PDF_DIR"
        echo "Or update the INPUT_PDF_DIR variable in this script"
        exit 1
    fi

    # Check if directory has PDF files
    PDF_COUNT=$(find "$INPUT_PDF_DIR" -name "*.pdf" -type f | wc -l)
    if [ $PDF_COUNT -eq 0 ]; then
        echo "ERROR: No PDF files found in directory: $INPUT_PDF_DIR"
        echo ""
        echo "Please place PDF files in the input directory"
        exit 1
    fi

    echo "  Input directory: $INPUT_PDF_DIR"
    echo "  Found $PDF_COUNT PDF file(s) to process"

    DETECT_TABLES_SCRIPT="$SCRIPT_DIR/detect_tables.py"
    DPI=300
    OCR_MODEL="deepseek-ocr"

    # Check if detect_tables.py exists
    if [ ! -f "$DETECT_TABLES_SCRIPT" ]; then
        echo "ERROR: detect_tables.py not found at: $DETECT_TABLES_SCRIPT"
        echo "Please ensure detect_tables.py is in kpi_extraction_project/"
        exit 1
    fi

    echo "  Running DeepSeek OCR table detection..."
    echo "    DPI: $DPI (native resolution, scales from OCR's 1024px)"
    echo "    OCR Model: $OCR_MODEL"
    python "$DETECT_TABLES_SCRIPT" \
        "$INPUT_PDF_DIR" \
        "$TABLES_JSON" \
        --output-dir "$DETECTED_TABLES_DIR" \
        --dpi $DPI \
        --model-name "$OCR_MODEL"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: DeepSeek OCR table detection failed"
        exit 1
    fi

    echo "  ✓ Table detection with DeepSeek OCR completed"
    echo ""
fi

# ============================================================================
# VLLM TENSOR PARALLEL CONFIGURATION
# ============================================================================

echo "[4/6] Configuring vLLM tensor parallelism..."

# Detect number of GPUs for tensor_parallel_size (vLLM reads this at runtime)
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "  Detected $NUM_GPUS GPU(s) — vLLM will use tensor_parallel_size=$NUM_GPUS"

# Optional: limit visible GPUs if you want fewer than allocated
# export CUDA_VISIBLE_DEVICES=0,1

# Disable NCCL peer-to-peer transfers — required on clusters where PCIe/NVLink P2P
# is blocked (prevents NCCL from hanging after initialization)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Redirect Triton and TorchInductor autotune caches to a persistent location.
# By default PyTorch uses /tmp or /scratch/<job_id>, which is deleted after each
# SLURM job. Cached compiled kernels and autotune results from a previous job
# therefore point to a non-existent path in the next job, causing:
#   PermissionError: [Errno 13] Permission denied: '/scratch/<old_job_id>'
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Clear the vLLM AOT compile cache before each run.
# Compiled artifacts embed the Triton autotune key paths from the original job's
# /scratch/<job_id> directory. Reusing them in a new job causes:
#   PermissionError: [Errno 13] Permission denied: '/scratch/<old_job_id>'
# Deleting the cache forces vLLM to recompile fresh (~60s) with correct paths.
VLLM_COMPILE_CACHE="$HOME/.cache/vllm/torch_compile_cache"
if [ -d "$VLLM_COMPILE_CACHE" ]; then
    echo "  Clearing stale vLLM compile cache: $VLLM_COMPILE_CACHE"
    rm -rf "$VLLM_COMPILE_CACHE"
fi

echo "  ✓ vLLM tensor parallel configuration ready"
echo ""

# ============================================================================
# CONFIGURATION FOR VLM
# ============================================================================

echo "[5/6] VLM Configuration..."

# Input and Output paths
# TABLES_JSON is already set above from table detection
OUTPUT_DIR="$SCRIPT_DIR/data/output/vlm"                         # Output directory for extracted KPIs

# Processing options
TEMPERATURE=0.0         # Sampling temperature (0.0 = deterministic, 0.1 = slightly random)

# Define models and their output directories
declare -a MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL-32B-Instruct"
    "Qwen2.5-VL-72B-Instruct"
)

declare -a OUTPUT_DIRS=(
    "$SCRIPT_DIR/data/output/vlm_qwen_7b"
    "$SCRIPT_DIR/data/output/vlm_qwen_32b"
    "$SCRIPT_DIR/data/output/vlm_qwen_72b"
)

# Validation options
DB_PATH="$SCRIPT_DIR/data/pack_context.db"    # Path to validation database

echo "  Tables JSON: $TABLES_JSON"
echo "  Temperature: $TEMPERATURE"
echo "  Database path: $DB_PATH"
echo "  Job ID: $SLURM_JOB_ID"
echo ""
echo "  Models to run:"
for i in "${!MODELS[@]}"; do
    echo "    ${MODELS[$i]} -> ${OUTPUT_DIRS[$i]}"
done
echo ""

# Check if tables.json exists
if [ ! -f "$TABLES_JSON" ]; then
    echo "ERROR: Tables JSON not found: $TABLES_JSON"
    echo ""
    echo "DeepSeek OCR table detection may have failed."
    echo "Check the error messages above."
    exit 1
fi

# Count tables in JSON
TABLE_COUNT=$(python - <<EOF
import json
try:
    with open("$TABLES_JSON", 'r') as f:
        data = json.load(f)
    print(len(data.get("tables", [])))
except:
    print(0)
EOF
)

if [ $TABLE_COUNT -eq 0 ]; then
    echo "ERROR: No tables found in $TABLES_JSON"
    echo "The tables.json file appears to be empty or invalid"
    exit 1
fi

echo "  Found $TABLE_COUNT table(s) to process with each model"
echo ""

# Create output directories if they don't exist
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    mkdir -p "$OUTPUT_DIR"
done

# ============================================================================
# RUN VLM EXTRACTION FOR BOTH MODELS
# ============================================================================

echo "[6/6] Running VLM KPI extraction for both models..."
echo ""

# Initialize overall exit code
OVERALL_EXIT_CODE=0

# Run extraction for each model
for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    
    echo "=========================================="
    echo "Processing with model: $MODEL_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="
    
    # Build command for tables.json processing
    CMD="python \"$SCRIPT_DIR/extract_kpis_vlm.py\" \
        --tables-json \"$TABLES_JSON\" \
        --output-dir \"$OUTPUT_DIR\" \
        --temperature $TEMPERATURE \
        --model-name \"$MODEL_NAME\" \
        --db-path \"$DB_PATH\""

    # Run the extraction
    echo "Command: $CMD"
    echo ""
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
    
    # ============================================================================
    # EXTRACTION STATISTICS ANALYSIS FOR THIS MODEL
    # ============================================================================
    
    echo "Analyzing extraction statistics for $MODEL_NAME..."
    echo ""
    
    # Path to the extraction summary file for this model
    EXTRACTION_SUMMARY="$OUTPUT_DIR/extraction_summary.json"
    
    # Check if extraction summary exists
    if [ -f "$EXTRACTION_SUMMARY" ]; then
        echo "  Found extraction summary: $(basename "$EXTRACTION_SUMMARY")"
        
        # Run the statistics analyzer
        echo "  Running extraction statistics analysis..."
        python "$SCRIPT_DIR/analyze_extraction_stats.py" \
            "$EXTRACTION_SUMMARY" \
            --output-json "$OUTPUT_DIR/extraction_statistics.json"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Extraction statistics analysis completed for $MODEL_NAME"
            
            # Show summary of statistics
            if [ -f "$OUTPUT_DIR/extraction_statistics.json" ]; then
                echo ""
                echo "  Statistics Summary for $MODEL_NAME:"
                python - <<EOF
import json
try:
    with open("$OUTPUT_DIR/extraction_statistics.json", 'r') as f:
        stats = json.load(f)
    
    print(f"    Tables processed: {stats.get('tables_processed', 0)}")
    print(f"    Success rate: {stats.get('success_rate', 0):.1f}%")
    print(f"    Total KPIs: {stats.get('total_kpis', 0)}")
    
    validation = stats.get('validation_performance', {})
    if validation:
        print(f"    Validation rate: {validation.get('validation_rate', 0):.1f}%")
        print(f"    Accuracy: {validation.get('accuracy', 0):.1f}%")
        print(f"    Avg confidence: {validation.get('average_confidence', 0):.1f}")
    
except Exception as e:
    print(f"    Could not display statistics: {e}")
EOF
            fi
        else
            echo "  ⚠ Statistics analysis failed for $MODEL_NAME"
        fi
    else
        echo "  ⚠ No extraction summary found for $MODEL_NAME: $(basename "$EXTRACTION_SUMMARY")"
    fi
    
    echo ""
    echo "=========================================="
    echo ""
    
done

# ============================================================================
# SUMMARY FOR ALL MODELS
# ============================================================================

echo "[7/7] Job Summary"
echo ""

if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo "✓ VLM extraction completed for all models!"
    echo ""
    
    # Show output files info for each model
    for i in "${!MODELS[@]}"; do
        MODEL_NAME="${MODELS[$i]}"
        OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
        
        echo "----------------------------------------"
        echo "Output files for $MODEL_NAME:"
        echo "Directory: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | tail -10
        echo ""
        
        # Count total files for this model
        NUM_FILES=$(ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
        echo "Total JSON output files for $MODEL_NAME: $NUM_FILES"
        
        # Show summary file if it exists
        SUMMARY_FILE="$OUTPUT_DIR/extraction_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "Extraction summary for $MODEL_NAME:"
            echo "---"
            python - <<EOF
import json
try:
    with open("$SUMMARY_FILE", 'r') as f:
        summary = json.load(f)
    print(f"  Model: {summary.get('model', 'N/A')}")
    print(f"  Total tables processed: {summary.get('total_tables', 0)}")
    print(f"  Total KPIs extracted: {summary.get('total_kpis', 0)}")
    print(f"  Extraction date: {summary.get('extraction_date', 'N/A')}")
except Exception as e:
    print(f"  Could not read summary: {e}")
EOF
        fi
        echo ""
    done
    
    # Clean up run counter on successful completion
    if [ -f "$RUN_COUNTER_FILE" ]; then
        rm "$RUN_COUNTER_FILE"
        echo "  Cleaned up run counter file"
    fi
else
    echo "✗ VLM extraction failed for at least one model with exit code: $OVERALL_EXIT_CODE"
    echo ""
    echo "Check the log file for details:"
    echo "  $SCRIPT_DIR/vlm_extraction_${SLURM_JOB_ID}.log"
    echo ""
    echo "This was run #${RUN_NUMBER}."
fi

echo ""
echo "=========================================="
echo "Run #${RUN_NUMBER} finished at: $(date)"
echo "Run runtime: $SECONDS seconds"
echo "=========================================="
echo ""
echo "############################################################################"
echo "##                    END OF RUN #${RUN_NUMBER}                                         ##"
echo "############################################################################"
echo ""

exit $OVERALL_EXIT_CODE
