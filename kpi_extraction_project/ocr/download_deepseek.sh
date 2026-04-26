#!/bin/bash
#SBATCH --job-name=download_deepseek
#SBATCH --output=download_deepseek_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-user=karim.ouf@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

################################################################################
# DeepSeek Model Download Job
# 
# Downloads DeepSeek VL models to UKP shared storage following cluster conventions.
# Models are stored in: /storage/ukp/shared/shared_model_weights/
# 
# Usage: sbatch download_deepseek.sh
################################################################################

echo ""
echo "############################################################################"
echo "##                                                                        ##"
echo "##                  DeepSeek Model Download - $(date '+%Y-%m-%d %H:%M:%S')              ##"
echo "##                                                                        ##"
echo "############################################################################"
echo ""

# Configuration - UKP Shared Storage
MODEL_ID="deepseek-ai/DeepSeek-OCR"
OUTPUT_DIR="/storage/ukp/shared/shared_model_weights/models--deepseek-ai--DeepSeek-OCR"

# Alternative models (uncomment to use):
# MODEL_ID="deepseek-ai/deepseek-vl-1.3b-chat"  # Smaller variant (faster, less memory)
# OUTPUT_DIR="/storage/ukp/shared/shared_model_weights/models--deepseek-ai--deepseek-vl-1.3b-chat"

echo "=========================================="
echo "DeepSeek Model Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Model: $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "[1/3] Setting up environment..."

# Set HOME to accessible storage location
export HOME=/ukp-storage-1/ouf
echo "  ✓ HOME set to: $HOME"

# Set cache directories
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

echo "  ✓ CUDA module loaded"
echo ""

# Verify huggingface_hub is installed
echo "[2/3] Verifying dependencies..."
python -c "import huggingface_hub; print(f'  ✓ huggingface_hub {huggingface_hub.__version__}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  Installing huggingface_hub..."
    pip install huggingface_hub
fi

echo ""

# Run download script
echo "[3/3] Downloading model..."
echo ""

# Get directory of this script for relative path to download_deepseek.py
SCRIPT_DIR="/ukp-storage-1/ouf/kpi_extraction_project/ocr"

python "$SCRIPT_DIR/download_deepseek.py" \
    --model "$MODEL_ID" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Download completed successfully!"
    echo ""
    
    # Create README following UKP conventions
    README_FILE="$OUTPUT_DIR/README.md"
    
    if [ ! -f "$README_FILE" ]; then
        echo "Creating README file..."
        cat > "$README_FILE" << EOFREADME
# DeepSeek VL Model

## Model Information

**Model Name:** DeepSeek VL 7B Chat  
**Added by:** Karim Ouf  
**Date:** $(date '+%Y-%m-%d')  
**Source:** https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat  

## Description

DeepSeek-VL is a vision-language model designed for multimodal understanding tasks including:
- Optical Character Recognition (OCR)
- Table extraction and understanding
- Visual question answering
- Document understanding

**Architecture:**
- Size: 7B parameters
- Vision encoder + Language model
- Chat-tuned for instruction following

**Alternative models:**
- \`deepseek-ai/deepseek-vl-1.3b-chat\` - Smaller, faster variant (1.3B parameters)

## How to Load the Model

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/storage/ukp/shared/shared_model_weights/models--deepseek-ai--deepseek-vl-7b-chat"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
\`\`\`

## Usage Notes

- Requires GPU with at least 16GB VRAM (7B model)
- Uses \`trust_remote_code=True\` for custom model components
- Supports 4-bit quantization with bitsandbytes for reduced memory usage

## Resources

- Hugging Face: https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat
- Official repo: https://github.com/deepseek-ai/DeepSeek-VL
- Paper: [Add paper link when available]

EOFREADME
        echo "  ✓ README created at: $README_FILE"
    else
        echo "  README already exists, skipping creation"
    fi
    
    echo ""
    echo "Model location: $OUTPUT_DIR"
    echo "Disk usage:"
    du -sh "$OUTPUT_DIR"
fi

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Runtime: $SECONDS seconds"
echo "=========================================="
echo ""
echo "############################################################################"
echo "##                    END OF DOWNLOAD JOB                                ##"
echo "############################################################################"
echo ""

exit $EXIT_CODE
