#!/bin/bash
#
#SBATCH --job-name=table-detection
#SBATCH --output=table_detection_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-user=karim.ouf@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

################################################################################
# Table Detection Job
# 
# This script runs table detection from PDFs using DeepSeek OCR model:
# - DeepSeek OCR: GPU-based, high accuracy, requires transformers
# - Processes PDF files and exports tables.json with table images and page markdown
#
# GPU Configuration:
# - 1x GPU requested (DeepSeek OCR requires GPU)
# - Model uses device_map="auto" for automatic distribution
#
# Input: PDF files in data/input directory
# Output: tables.json file and cropped table images
#
# Usage: sbatch run_table_detection.sh
################################################################################

# ============================================================================
# LOG SEPARATOR - Track restarts
# ============================================================================

# Set the project directory early for run counter
SCRIPT_DIR="/storage/ukp/work/ouf/kpi_extraction_project"
RUN_COUNTER_FILE="$SCRIPT_DIR/data/output/.table_detection_run_counter_${SLURM_JOB_ID}"

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
echo "##              TABLE DETECTION - RUN #${RUN_NUMBER} - $(date '+%Y-%m-%d %H:%M:%S')              ##"
echo "##                                                                        ##"
echo "############################################################################"
echo "############################################################################"
echo ""
echo "=========================================="
echo "DeepSeek OCR Table Detection Job"
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

echo "[1/4] Setting up DeepSeek OCR environment..."

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

# Install DeepSeek OCR dependencies
echo "  Installing transformers 4.46.3 for DeepSeek OCR..."
pip install --quiet transformers==4.46.3

# Try to disable flash attention if it causes GLIBC issues
export FLASH_ATTENTION_FORCE_DISABLE=1
echo "  ✓ Flash attention disabled (compatibility mode)"

# Load CUDA module (ignore errors if not available)
module load cuda/12.1 2>/dev/null || echo "  ⚠ CUDA module not available"
echo "  ✓ CUDA environment configured"

# Enable PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "  ✓ PyTorch memory optimization enabled (expandable_segments:True)"

echo ""

# ============================================================================
# VERIFY DEPENDENCIES
# ============================================================================

echo "[2/4] Verifying and installing dependencies..."

# Install required libraries
echo "  Installing required libraries..."
pip install --quiet pymupdf pillow

python - <<EOF
import sys

try:
    # Verify PyTorch and CUDA
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"    - CUDA available: True")
        print(f"    - Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"    - GPU {i}: {props.name}")
            print(f"      Memory: {memory_gb:.1f} GB")
    else:
        print(f"    - CUDA available: False")
        print("    - Will use CPU (slower)")
    
    # Verify transformers
    import transformers
    print(f"  ✓ Transformers {transformers.__version__}")
    
    # Verify PIL
    from PIL import Image
    print(f"  ✓ Pillow (PIL) installed")
    
    # Verify PyMuPDF (fitz)
    import fitz
    print(f"  ✓ PyMuPDF (fitz) {fitz.__version__}")
    
    print("\n✓ All dependencies OK!")
    
except Exception as e:
    print(f"\n✗ Dependency error: {e}", file=sys.stderr)
    print("Please install missing dependencies in your conda environment:", file=sys.stderr)
    print("  pip install --no-cache-dir pymupdf pillow transformers", file=sys.stderr)
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

echo "[3/4] Detecting tables in PDF with DeepSeek OCR..."

# Input PDF directory path
INPUT_PDF_DIR="$SCRIPT_DIR/data/input_test"  # Directory containing PDF files

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

# Output paths for table detection
DETECTED_TABLES_DIR="$SCRIPT_DIR/data/detected_tables_test"
TABLES_JSON="$DETECTED_TABLES_DIR/tables.json"
DETECT_TABLES_SCRIPT="$SCRIPT_DIR/detect_tables.py"
DPI=300
OCR_MODEL="deepseek-ocr"

# Check if detect_tables.py exists
if [ ! -f "$DETECT_TABLES_SCRIPT" ]; then
    echo "ERROR: detect_tables.py not found at: $DETECT_TABLES_SCRIPT"
    echo "Please ensure detect_tables.py is in kpi_extraction_project/"
    exit 1
fi

echo "  Running DeepSeek OCR table detection."
echo "    DPI: $DPI (native resolution, scales from OCR's 1024px)"
echo "    OCR Model: $OCR_MODEL"
echo "    Output directory: $DETECTED_TABLES_DIR"
echo ""

python "$DETECT_TABLES_SCRIPT" \
    "$INPUT_PDF_DIR" \
    "$TABLES_JSON" \
    --output-dir "$DETECTED_TABLES_DIR" \
    --dpi $DPI \
    --model-name "$OCR_MODEL"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: DeepSeek OCR table detection failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "  ✓ Table detection with DeepSeek OCR completed"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "[4/4] Table Detection Summary"
echo ""

if [ -f "$TABLES_JSON" ]; then
    echo "✓ Table detection completed successfully!"
    echo ""
    
    # Count tables detected
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
    
    echo "Results:"
    echo "  Tables JSON: $TABLES_JSON"
    echo "  Output directory: $DETECTED_TABLES_DIR"
    echo "  Total tables detected: $TABLE_COUNT"
    echo ""
    
    # Show some statistics
    if [ $TABLE_COUNT -gt 0 ]; then
        echo "Table distribution by PDF:"
        python - <<EOF
import json
from collections import defaultdict

try:
    with open("$TABLES_JSON", 'r') as f:
        data = json.load(f)
    
    pdf_counts = defaultdict(int)
    for table in data.get("tables", []):
        pdf_name = table.get("pdf_file", "unknown")
        pdf_counts[pdf_name] += 1
    
    for pdf_name, count in sorted(pdf_counts.items()):
        print(f"    {pdf_name}: {count} table(s)")
        
except Exception as e:
    print(f"    Could not analyze table distribution: {e}")
EOF
        echo ""
    fi
    
    # Show output directory contents
    echo "Output directory contents:"
    ls -la "$DETECTED_TABLES_DIR" 2>/dev/null | head -10
    echo ""
    
    # Clean up run counter on successful completion
    if [ -f "$RUN_COUNTER_FILE" ]; then
        rm "$RUN_COUNTER_FILE"
        echo "  Cleaned up run counter file"
    fi
else
    echo "✗ Table detection failed!"
    echo ""
    echo "No tables.json file was created."
    echo "Check the log file for details:"
    echo "  $SCRIPT_DIR/table_detection_${SLURM_JOB_ID}.log"
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

exit $EXIT_CODE