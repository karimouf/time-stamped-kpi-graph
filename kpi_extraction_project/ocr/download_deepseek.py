#!/usr/bin/env python3
"""
Download DeepSeek VL model for OCR/table extraction tasks.

This script downloads the model to a local directory without symlinks,
ensuring a self-contained repository that won't break if cache is cleaned.

Usage:
    python download_deepseek.py --model deepseek-ai/deepseek-vl-7b-chat --output-dir /path/to/models
    
On SLURM cluster:
    sbatch download_deepseek.sh
"""

import argparse
import os
from pathlib import Path

try:
    import huggingface_hub
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install it with: pip install huggingface_hub")
    exit(1)


def download_model(repo_id: str, local_dir: str, force_download: bool = False):
    """
    Download a model from Hugging Face Hub to a local directory.
    
    Args:
        repo_id: Hugging Face model repository ID (e.g., 'deepseek-ai/deepseek-vl-7b-chat')
        local_dir: Local directory to save the model
        force_download: Whether to force re-download even if files exist
    """
    print(f"Downloading model: {repo_id}")
    print(f"Target directory: {local_dir}")
    print(f"Force download: {force_download}")
    print("-" * 80)
    
    # Create directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model without symlinks for self-contained repository
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Avoid symlink issues with cache
            force_download=force_download,
        )
        print(f"\n✓ Successfully downloaded {repo_id} to {local_dir}")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download DeepSeek models from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download DeepSeek VL 7B model to UKP shared storage
  python download_deepseek.py --model deepseek-ai/DeepSeek-OCR \
      --output-dir /storage/ukp/shared/shared_model_weights/models--deepseek-ai--DeepSeek-OCR
  
  # Download with force re-download
  python download_deepseek.py --model deepseek-ai/DeepSeek-OCR \
      --output-dir /storage/ukp/shared/shared_model_weights/models--deepseek-ai--DeepSeek-OCR  --force
  
  # Download 1.3B variant (smaller, faster)
  python download_deepseek.py --model deepseek-ai/deepseek-vl-1.3b-chat \
      --output-dir /storage/ukp/shared/shared_model_weights/models--deepseek-ai--deepseek-vl-1.3b-chat
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Hugging Face model repository ID (default: deepseek-ai/DeepSeek-OCR)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to save the model"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist"
    )
    
    args = parser.parse_args()
    
    download_model(
        repo_id=args.model,
        local_dir=args.output_dir,
        force_download=args.force
    )


if __name__ == "__main__":
    main()
