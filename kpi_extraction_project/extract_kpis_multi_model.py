#!/usr/bin/env python3
"""
Multi-Model KPI Extraction System
==================================

Extracts Key Performance Indicators from financial tables using three state-of-the-art models:
1. Gemma 3 PT 27B - Google's large instruction-tuned model
2. DeepSeek-V2.5 - Advanced reasoning model
3. Llama 3 8B Instruct - Meta's instruction-following model

Models are loaded from the cluster's shared storage and used in ensemble to extract
structured KPI data from JSON-formatted financial tables.

Author: Karim Ouf
Date: November 2025
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Base path for shared model weights on the UKP cluster
SHARED_MODELS_BASE = "/storage/ukp/shared/shared_model_weights"

# Model configurations with metadata
MODEL_CONFIGS = {
    # "gemma-3-27b": {
    #     "path": f"{SHARED_MODELS_BASE}/models--google--gemma-3-27b-it",
    #     "includes_prompt_in_output": False,  # Gemma models don't echo prompts
    #     "description": "Google Gemma 3 27B - Excellent reasoning and instruction following"
    # },
    # "deepseek-v2.5": {
    #     "path": f"{SHARED_MODELS_BASE}/models-deepseek-ai--DeepSeek-V2.5-1210",
    #     "includes_prompt_in_output": False,  # DeepSeek typically doesn't echo
    #     "description": "DeepSeek V2.5 - Advanced reasoning and coding capabilities"
    # },
    "llama-3-8b": {
        "path": f"{SHARED_MODELS_BASE}/models--llama-3/8B-Instruct",
        "includes_prompt_in_output": True,   # Llama models echo the full prompt
        "description": "Meta Llama 3.1 8B Instruct - Strong general-purpose instruction following"
    }
}

# System prompt for KPI extraction
SYSTEM_PROMPT = """You are a financial data extraction AI specialized in identifying Key Performance Indicators (KPIs) from structured tables.

Your task is to extract ALL numerical metrics and their metadata from the provided financial table.

Guidelines:
- Extract metric names exactly as they appear
- Capture all numerical values with their units
- Identify time periods (years, quarters, etc.)
- Categorize KPIs (financial, operational, market, etc.)
- Include context from headers and row labels

Return ONLY valid JSON in this exact format (no additional text):
{
  "kpis": [
    {
      "metric_name": "Revenue",
      "value": "322284",
      "unit": "€ million",
      "time_period": "2023",
      "category": "financial",
      "context": "Volkswagen Group total revenue"
    }
  ]
}"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_json_response(text: str, remove_prompt: str = None) -> str:
    """
    Clean and extract JSON from model output.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Prompt echoing (removes the original prompt if present)
    - Extra whitespace and formatting
    - Truncated JSON (attempts to repair)
    
    Args:
        text: Raw model output
        remove_prompt: Original prompt to remove if model echoed it
        
    Returns:
        Cleaned JSON string
    """
    # If we need to remove the prompt (for models like Llama that echo it)
    if remove_prompt and remove_prompt in text:
        text = text.replace(remove_prompt, "", 1).strip()
    
    # Extract JSON from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        # Handle generic code blocks
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    
    # Remove common prefixes
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    
    # Attempt to repair truncated JSON
    text = repair_truncated_json(text)
    
    return text


def repair_truncated_json(text: str) -> str:
    """
    Attempt to repair truncated JSON by closing incomplete structures.
    
    Common issues from token limit truncation:
    - Unterminated strings
    - Unclosed arrays
    - Unclosed objects
    
    Args:
        text: Potentially truncated JSON string
        
    Returns:
        Repaired JSON string
    """
    if not text:
        return text
    
    # Count opening vs closing brackets/braces
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    
    # Check for unterminated string (odd number of quotes on last line)
    lines = text.split('\n')
    if lines:
        last_line = lines[-1]
        # Count quotes in last line (ignore escaped quotes)
        quote_count = last_line.count('"') - last_line.count('\\"')
        if quote_count % 2 == 1:
            # Odd number of quotes = unterminated string
            # Remove the incomplete last line
            text = '\n'.join(lines[:-1])
            # Recalculate bracket counts after removing last line
            open_braces = text.count('{')
            close_braces = text.count('}')
            open_brackets = text.count('[')
            close_brackets = text.count(']')
    
    # Close any unclosed arrays and objects
    repairs = []
    
    # Close arrays first (innermost structures)
    for _ in range(open_brackets - close_brackets):
        repairs.append(']')
    
    # Close objects
    for _ in range(open_braces - close_braces):
        repairs.append('}')
    
    if repairs:
        logger.info(f"    → JSON repair: closing {len(repairs)} unclosed structures")
        text = text.rstrip().rstrip(',') + '\n' + ''.join(repairs)
    
    return text


def validate_kpi_structure(kpi: Dict) -> bool:
    """
    Validate that a KPI dictionary has the required fields.
    
    Args:
        kpi: Dictionary representing a single KPI
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["metric_name", "value"]
    return all(field in kpi for field in required_fields)


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class MultiModelKPIExtractor:
    """
    Ensemble KPI extractor using multiple LLMs.
    
    This class manages loading, inference, and result aggregation from
    three different language models to extract KPIs from financial tables.
    """
    
    def __init__(self, models_to_use: Optional[List[str]] = None):
        """
        Initialize the extractor with specified models.
        
        Args:
            models_to_use: List of model names to load (default: all 3)
        """
        self.models_to_use = models_to_use or list(MODEL_CONFIGS.keys())
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {}
        
        logger.info(f"Initializing Multi-Model KPI Extractor with {len(self.models_to_use)} models")
        logger.info("=" * 70)
        
        # Load each model
        for model_name in self.models_to_use:
            if model_name not in MODEL_CONFIGS:
                logger.warning(f"Model '{model_name}' not found in configuration, skipping")
                continue
            self._load_model(model_name)
        
        logger.info("=" * 70)
        logger.info(f"Successfully loaded {len(self.models)} models")
    
    def _load_model(self, model_name: str) -> None:
        """
        Load a single model and its tokenizer.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]
            
            logger.info(f"Loading {model_name}...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")
            
            # Load tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                legacy=False,
                padding_side="left",
                trust_remote_code=False
            )
            
            # Load model with optimizations (same pattern as Llama-2 example)
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=False
            )
            
            # Configure padding token if not set (essential for batch tokenizing)
            if self.tokenizers[model_name].pad_token is None:
                self.tokenizers[model_name].add_special_tokens({"pad_token": "<pad>"})
                self.models[model_name].resize_token_embeddings(len(self.tokenizers[model_name]))
                self.models[model_name].config.pad_token_id = self.tokenizers[model_name].pad_token_id
                self.models[model_name].generation_config.pad_token_id = self.tokenizers[model_name].pad_token_id
            
            # Store config for later use
            self.model_configs[model_name] = config
            
            logger.info(f"  ✓ Successfully loaded {model_name}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_name}: {str(e)}")
            # Clean up partial loads
            if model_name in self.models:
                del self.models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
    
    def extract_kpis_single_model(
        self,
        table_data: Dict[str, Any],
        model_name: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract KPIs using a single model.
        
        Args:
            table_data: Dictionary containing table information
            model_name: Name of the model to use
            max_new_tokens: Maximum tokens to generate (default: 2048)
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Dictionary with extracted KPIs and metadata
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded, skipping")
            return {
                "kpis": [],
                "model": model_name,
                "error": "Model not available"
            }
        
        try:
            # Prepare the extraction prompt
            table_json = json.dumps(table_data, indent=2, ensure_ascii=False)
            prompt = f"{SYSTEM_PROMPT}\n\nTable Data:\n{table_json}\n\nExtracted KPIs (JSON only):"
            logger.info(f"prompt {prompt}")
            
            # Tokenize input
            inputs = self.tokenizers[model_name](
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.models[model_name].device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Generate response
            logger.info(f"    → Generating {max_new_tokens} tokens (temperature={temperature})...")
            with torch.inference_mode():
                outputs = self.models[model_name].generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95 if temperature > 0 else None,
                    pad_token_id=self.tokenizers[model_name].pad_token_id,
                    eos_token_id=self.tokenizers[model_name].eos_token_id
                )
            logger.info(f"    → Generation complete. Decoding output...")
            
            # Decode response
            # Only decode the newly generated tokens (skip input prompt)
            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizers[model_name].decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Clean response (remove prompt if model echoed it)
            if self.model_configs[model_name]["includes_prompt_in_output"]:
                generated_text = clean_json_response(generated_text, remove_prompt=prompt)
            else:
                generated_text = clean_json_response(generated_text)
            
            logger.info(f"    → Cleaned response length: {len(generated_text)} characters")
            
            # Parse JSON
            logger.info(f"    → Parsing JSON response...")
            try:
                result = json.loads(generated_text)
                
                # Validate structure
                if "kpis" in result and isinstance(result["kpis"], list):
                    # Add source model to each KPI
                    for kpi in result["kpis"]:
                        kpi["source_model"] = model_name
                    
                    # Filter out invalid KPIs
                    valid_kpis = [kpi for kpi in result["kpis"] if validate_kpi_structure(kpi)]
                    
                    result["kpis"] = valid_kpis
                    result["model"] = model_name
                    result["num_kpis"] = len(valid_kpis)
                    
                    logger.info(f"    ✓ Extracted {len(valid_kpis)} KPIs from {model_name}")
                    return result
                else:
                    logger.warning(f"  Invalid JSON structure from {model_name}")
                    return {
                        "kpis": [],
                        "model": model_name,
                        "error": "Invalid JSON structure"
                    }
                    
            except json.JSONDecodeError as e:
                logger.warning(f"  JSON parsing failed for {model_name}: {str(e)}")
                logger.warning(f"  Raw output (last 500 chars): ...{generated_text[-500:]}")
                logger.warning(f"  Total output length: {len(generated_text)} characters")
                return {
                    "kpis": [],
                    "model": model_name,
                    "raw_output": generated_text,  # Store FULL output for debugging
                    "error": f"JSON parsing failed: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"  Error during extraction with {model_name}: {str(e)}")
            return {
                "kpis": [],
                "model": model_name,
                "error": str(e)
            }
    
    def extract_kpis_ensemble(
        self,
        table_data: Dict[str, Any],
        max_new_tokens: int = 2048,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract KPIs using all loaded models and combine results.
        
        Uses ensemble approach where each model contributes its extractions.
        
        Args:
            table_data: Dictionary containing table information
            max_new_tokens: Maximum tokens to generate per model (default: 2048)
            temperature: Sampling temperature
            
        Returns:
            Dictionary with combined results from all models
        """
        table_id = table_data.get('table_id', 'unknown')
        logger.info(f"Processing table: {table_id}")
        
        # Run each model
        all_results = {}
        all_kpis = []
        
        for model_name in self.models.keys():
            logger.info(f"  Running {model_name}...")
            result = self.extract_kpis_single_model(
                table_data,
                model_name,
                max_new_tokens,
                temperature
            )
            all_results[model_name] = result
            
            # Collect KPIs from this model
            if "kpis" in result and isinstance(result["kpis"], list):
                all_kpis.extend(result["kpis"])
        
        # Compile ensemble result
        ensemble_result = {
            "table_id": table_data.get("table_id"),
            "doc_id": table_data.get("doc_id"),
            "year": table_data.get("year"),
            "section_name": table_data.get("section_name"),
            "title": table_data.get("title"),
            "extraction_timestamp": datetime.now().isoformat(),
            "models_used": list(self.models.keys()),
            "num_models": len(self.models),
            "individual_results": all_results,
            "all_kpis": all_kpis,
            "total_kpis_extracted": len(all_kpis)
        }
        
        logger.info(f"  ✓ Extracted {len(all_kpis)} total KPIs from {len(self.models)} models")
        
        return ensemble_result
    
    def process_jsonl_file(
        self,
        input_file: str,
        output_file: str,
        max_tables: Optional[int] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.1
    ) -> None:
        """
        Process a JSONL file containing tables.
        
        Reads tables line-by-line, extracts KPIs, and writes results.
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            max_tables: Maximum number of tables to process (None = all)
            max_new_tokens: Maximum tokens to generate per model (default: 2048)
            temperature: Sampling temperature (default: 0.1)
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info(f"Processing file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Max tables: {max_tables if max_tables else 'All'}")
        logger.info("=" * 70)
        
        processed = 0
        errors = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                # Check if we've hit the limit
                if max_tables and processed >= max_tables:
                    logger.info(f"Reached maximum table limit ({max_tables})")
                    break
                
                try:
                    # Parse table data
                    table_data = json.loads(line)
                    
                    # Extract KPIs
                    result = self.extract_kpis_ensemble(
                        table_data,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    
                    # Write result
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()  # Ensure write in case of crash
                    
                    processed += 1
                    
                    # Progress update every 5 tables
                    if processed % 5 == 0:
                        logger.info(f"Progress: {processed} tables processed")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {str(e)}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {str(e)}")
                    errors += 1
        
        # Final summary
        logger.info("=" * 70)
        logger.info(f"Processing complete!")
        logger.info(f"  Tables processed: {processed}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Output saved to: {output_file}")
        logger.info("=" * 70)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract KPIs from financial tables using multiple LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tables with all 3 models
  python extract_kpis_multi_model.py --input tables.jsonl --output results.jsonl
  
  # Process first 10 tables only
  python extract_kpis_multi_model.py --input tables.jsonl --output results.jsonl --max-tables 10
  
  # Use only specific models
  python extract_kpis_multi_model.py --input tables.jsonl --output results.jsonl --models llama-3-8b gemma-3-27b
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file containing tables"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file for extracted KPIs"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific models to use (default: all 3)"
    )
    
    parser.add_argument(
        "--max-tables",
        type=int,
        default=None,
        help="Maximum number of tables to process (default: all)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per model (default: 2048)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1, range: 0.0-1.0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize extractor
    extractor = MultiModelKPIExtractor(models_to_use=args.models)
    
    # Check if any models were loaded
    if not extractor.models:
        logger.error("No models were successfully loaded. Exiting.")
        return 1
    
    # Process the file
    try:
        extractor.process_jsonl_file(
            args.input,
            args.output,
            args.max_tables,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
