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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Any, Optional
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Import JSON utilities
from json_utils import clean_json_response, parse_json_safely

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
    #     "description": "Google Gemma 3 27B - Excellent reasoning and instruction following",
    #     "max_new_tokens": 4096  # Generous limit for complete JSON generation
    # },
    # "deepseek-v2.5": {
    #     "path": f"{SHARED_MODELS_BASE}/models-deepseek-ai--DeepSeek-V2.5-1210",
    #     "includes_prompt_in_output": False,  # DeepSeek typically doesn't echo
    #     "description": "DeepSeek V2.5 - Advanced reasoning and coding capabilities (4-bit quantized, 2 GPUs + CPU offload)",
    #     "max_new_tokens": 2048,  # Reduced for memory efficiency
    #     "quantization": "4bit",  # 4-bit NF4 quantization for 236B MoE model
    #     "max_memory": {0: "75GB", 1: "75GB", "cpu": "60GB"},  # Distribute across 2 GPUs + CPU offload
    #     "llm_int8_enable_fp32_cpu_offload": True  # Enable CPU offload for layers that don't fit
    # },
    "deepseek-r1-distill-llama-70b": {
        "path": f"{SHARED_MODELS_BASE}/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
        "includes_prompt_in_output": True,   # Llama-based models echo the full prompt
        "description": "DeepSeek R1 Distill Llama 70B - Distilled reasoning model based on Llama architecture",
        "max_new_tokens": 16384,  # Maximum output tokens for complete KPI extraction
        "max_memory": {0: "75GB", 1: "75GB"}  # Distribute across 2 GPUs
    },
    # "llama-3-70b": {
    #     "path": f"{SHARED_MODELS_BASE}/models--llama-3/70B-Instruct",
    #     "includes_prompt_in_output": True,   # Llama models echo the full prompt
    #     "description": "Meta Llama 3 70B Instruct - Strong general-purpose instruction following",
    #     "max_new_tokens": 8192,  # Maximum output tokens for complete KPI extraction
    #     "max_memory": {0: "75GB", 1: "75GB"}  # Distribute across 2 GPUs
    # },
    # "llama-3-8b": {
    #     "path": f"{SHARED_MODELS_BASE}/models--llama-3/8B-Instruct",
    #     "includes_prompt_in_output": True,   # Llama models echo the full prompt
    #     "description": "Meta Llama 3 8B Instruct - Strong general-purpose instruction following",
    #     "max_new_tokens": 8192  # Maximum output tokens for complete KPI extraction
    # }
}

# System prompt for KPI extraction
SYSTEM_PROMPT = """You are a financial data extraction AI specialized in identifying Key Performance Indicators (KPIs) from structured tables.

Your task is to extract ALL numerical metrics and their metadata from the provided financial table.

CRITICAL OUTPUT CONTROL:
- Think briefly (max 50 words) if needed, then output JSON immediately
- Do NOT engage in lengthy deliberation about edge cases
- If uncertain about a field value, make the most reasonable choice and proceed
- Repetitive reasoning will cause extraction failure - decide quickly and continue

CRITICAL INDEXING RULES:
- Use ZERO-BASED indexing for both row_idx and col_idx
- row_idx: Position in the stub_col array (first row = 0)
- column 1 (col_idx=1) is the first data column containing values while column 0 (col_idx=0) is ALWAYS the row label column, NOT data
- NEVER mix 0-based and 1-based conventions

BEFORE finalizing each KPI extraction:
1. CHECK: Is col_idx = 0? If YES, you're extracting a row label, NOT data - SKIP IT!
2. CHECK: Is merged_headers[col_idx] empty or a label word? If YES, it's probably the row label column - use col_idx+1 instead
3. Verify row_idx by checking: stub_col[row_idx] == row_name
4. Verify col_idx by checking: merged_headers[col_idx] == col_name (should be a year, percentage, or data label)
5. If merged_headers[col_idx] is empty (""), increment col_idx by 1
6. Remember: First element is ALWAYS index 0, but first DATA column is usually index 1

Guidelines:
- Extract metric names exactly as they appear in the table
- Capture all numerical values with their units
- Identify time periods (years, quarters, etc.)
- Record the exact row name (stub_col entry or row entry)) and column name (merged_headers entry) for each value

Return ONLY valid JSON in this exact format (no additional text):
{
  "kpis": [
   {
    "name": "<KPI metric name, e.g., Sales, Operating Cost, Deliveries>",
    "key": "<entity or context for the KPI, e.g., Audi, Core Brand Group, market segment>",
    "units": "<measurement units or 'N/A'>",
    "value": <numeric value or null>,
    "year": <year as integer>,
    "row_name": "<exact text from stub_col or from row>",
    "row_idx": <integer row index>,
    "col_name": "<exact text from merged_headers>"
    "col_idx": <integer column index>
    }
  ]
}

Important:
- row_name: Use the EXACT text from the stub_col array for the row containing this value
- col_name: Use the EXACT text from the merged_headers array for the column containing this value
- These names enable deterministic validation - they must match the source table exactly
- Extract ALL numerical cells in the table
}"""

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
        # Don't load models upfront - load them on-demand
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        logger.info(f"Initializing Multi-Model KPI Extractor with {len(self.models_to_use)} models")
        logger.info("Models will be loaded sequentially on-demand to save memory")
        logger.info("=" * 70)
    
    def _load_model(self, model_name: str) -> bool:
        """
        Load a single model and its tokenizer.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = MODEL_CONFIGS[model_name]
            model_path = config["path"]
            
            logger.info(f"Loading {model_name}...")
            logger.info(f"  Path: {model_path}")
            logger.info(f"  Description: {config['description']}")
            
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                legacy=False,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Configure quantization if specified
            quantization_config = None
            llm_int8_enable_fp32_cpu_offload = config.get("llm_int8_enable_fp32_cpu_offload", False)
            
            if config.get("quantization") == "4bit":
                logger.info(f"  Using 4-bit NF4 quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",              # NormalFloat4 - optimal for LLMs
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16 for speed
                    bnb_4bit_use_double_quant=True,         # Nested quantization for extra memory savings
                    llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload  # Enable CPU offload if needed
                )
            elif config.get("quantization") == "8bit":
                logger.info(f"  Using 8-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
                )
            
            # Get max_memory config for multi-GPU distribution
            max_memory = config.get("max_memory", None)
            if max_memory:
                logger.info(f"  Using multi-GPU setup with memory limits: {max_memory}")
                if llm_int8_enable_fp32_cpu_offload:
                    logger.info(f"  CPU offload enabled for layers that don't fit in GPU")
            
            # Load model with optimizations
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                max_memory=max_memory,  # Distribute across GPUs + CPU if specified
                torch_dtype=torch.bfloat16 if quantization_config is None else None,
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            
            # Configure padding token if not set (essential for batch tokenizing)
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.current_model.resize_token_embeddings(len(self.current_tokenizer))
                self.current_model.config.pad_token_id = self.current_tokenizer.pad_token_id
                self.current_model.generation_config.pad_token_id = self.current_tokenizer.pad_token_id
            
            # Store current model name and config
            self.current_model_name = model_name
            
            # Log GPU memory usage after loading
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9  # Convert to GB
                reserved = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                available = total - allocated
                logger.info(f"  GPU Memory: {allocated:.2f}GB allocated, {available:.2f}GB available (of {total:.2f}GB total)")
            
            logger.info(f"  ✓ Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load {model_name}: {str(e)}")
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            return False
    
    def _unload_model(self) -> None:
        """
        Unload the current model to free GPU memory.
        """
        if self.current_model is not None:
            logger.info(f"  Unloading {self.current_model_name}...")
            
            # Log GPU memory before unloading
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"  GPU Memory before unload: {allocated_before:.2f}GB allocated")
            
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log GPU memory after unloading
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                freed = allocated_before - allocated_after
                available = total - allocated_after
                logger.info(f"  GPU Memory after unload: {allocated_after:.2f}GB allocated, {available:.2f}GB available")
                logger.info(f"  ✓ Freed {freed:.2f}GB of GPU memory")
            else:
                logger.info(f"  ✓ Model unloaded")
    
    def extract_kpis_single_model(
        self,
        table_data: Dict[str, Any],
        model_name: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Extract KPIs using a single model.
        
        Args:
            table_data: Dictionary containing table information
            model_name: Name of the model to use
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Dictionary with extracted KPIs and metadata
        """
        # Check if model is available
        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Model {model_name} not in configuration, skipping")
            return {
                "kpis": [],
                "model": model_name,
                "error": "Model not in configuration"
            }
        
        # Load model if not already loaded or if different model is loaded
        if self.current_model_name != model_name:
            # Unload previous model if any
            if self.current_model is not None:
                self._unload_model()
            
            # Load new model
            if not self._load_model(model_name):
                return {
                    "kpis": [],
                    "model": model_name,
                    "error": "Failed to load model"
                }
        
        try:
            # Prepare the extraction prompt
            table_json = json.dumps(table_data, ensure_ascii=False)
            prompt = f"{SYSTEM_PROMPT}\n\n### 📥 **Input Placeholder**\n\n```\n{table_json}\n```"

            
            # Tokenize input
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.current_model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            # Get model-specific max_new_tokens limit
            config = MODEL_CONFIGS[model_name]
            max_new_tokens = config.get("max_new_tokens", 2048)
            
            # Generate response with model-specific token limit
            logger.info(f"    → Generating tokens (max: {max_new_tokens}, temperature={temperature})...")
            with torch.inference_mode():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95 if temperature > 0 else None,
                    pad_token_id=self.current_tokenizer.pad_token_id,
                    eos_token_id=self.current_tokenizer.eos_token_id
                )
            logger.info(f"    → Generation complete. Decoding output...")
            
            # Decode response
            # Only decode the newly generated tokens (skip input prompt)
            generated_ids = outputs[0][input_length:]
            generated_text = self.current_tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Save raw output to file before cleaning
            # table_id = table_data.get('table_id', 'unknown')
           

            # Clean response (remove prompt if model echoed it)
            config = MODEL_CONFIGS[model_name]
            if config["includes_prompt_in_output"]:
                cleaned_text = clean_json_response(generated_text, remove_prompt=prompt)
            else:
                cleaned_text = clean_json_response(generated_text)

            # Parse JSON
            logger.info(f"    → Parsing JSON response...")
            try:
                result = json.loads(cleaned_text)

                # Validate structure
                if "kpis" in result and isinstance(result["kpis"], list):
                    # Add source model to each KPI
                    for kpi in result["kpis"]:
                        kpi["source_model"] = model_name
                    
                    # Filter out invalid KPIs
                    
                    result["kpis"]
                    result["model"] = model_name
                    result["num_kpis"] = len(result["kpis"])

                    logger.info(f"    ✓ Extracted {len(result['kpis'])} KPIs from {model_name}")
                    return result
                else:
                    logger.warning(f"  Invalid JSON structure from {model_name}")
                    return {
                        "kpis": [],
                        "model": model_name,
                        "error": "Invalid JSON structure"
                    }
                    
            except json.JSONDecodeError as e:
                # Save raw and cleaned output for debugging when JSON parsing fails
                table_id = table_data.get('table_id', 'unknown')
                logger.warning(f"  JSON parsing failed for {model_name}: {str(e)}")
                logger.warning(f"  Saving raw/cleaned output for debugging...")
                
                raw_output_path = f"/ukp-storage-1/ouf/kpi_extraction_project/data/output/raw_cleaned_{model_name}_{table_id}.txt"
                with open(raw_output_path, "w", encoding="utf-8") as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Table: {table_id}\n")
                    f.write(f"\n{'=' * 50}\n")
                    f.write(f"RAW OUTPUT\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write(generated_text)
                    f.write(f"\n{'=' * 50}\n")
                    f.write(f"CLEANED OUTPUT\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write(cleaned_text)
                    f.write(f"\n{'=' * 50}\n")
                    f.write(f"ERROR\n")
                    f.write(f"{'=' * 50}\n\n")
                    f.write(f"JSON parsing error: {str(e)}\n")
                
                logger.warning(f"  Debug file saved: {raw_output_path}")
                
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
        finally:
            # Clean up GPU memory after each extraction to prevent buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_tables_with_model(
        self,
        tables: List[Dict[str, Any]],
        model_name: str,
        temperature: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Process all tables with a single model (load once, process all, unload).
        
        Args:
            tables: List of table data dictionaries
            model_name: Name of the model to use
            temperature: Sampling temperature
            
        Returns:
            List of extraction results, one per table
        """
        logger.info(f"=" * 70)
        logger.info(f"Processing {len(tables)} tables with {model_name}")
        logger.info(f"=" * 70)
        
        # Load the model once
        if not self._load_model(model_name):
            logger.error(f"Failed to load {model_name}, skipping all tables for this model")
            # Return empty results for all tables with error message
            return [{
                "table_id": table.get("table_id"),
                "doc_id": table.get("doc_id"),
                "year": table.get("year"),
                "model": model_name,
                "error": "Failed to load model - possibly out of memory",
                "extraction_timestamp": datetime.now().isoformat()
            } for table in tables]
        
        results = []
        
        # Process all tables with this model
        for idx, table_data in enumerate(tables, 1):
            table_id = table_data.get('table_id', 'unknown')
            logger.info(f"[{idx}/{len(tables)}] Processing table: {table_id}")
            
            try:
                result = self.extract_kpis_single_model(
                    table_data,
                    model_name,
                    temperature
                )
                
                # Add table metadata to result
                result_with_metadata = {
                    "table_id": table_data.get("table_id"),
                    "doc_id": table_data.get("doc_id"),
                    "year": table_data.get("year"),
                    "section_name": table_data.get("section_name"),
                    "title": table_data.get("title"),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "extraction_result": result
                }
                
                results.append(result_with_metadata)
                
                # Progress update
                if idx % 5 == 0:
                    logger.info(f"  Progress: {idx}/{len(tables)} tables processed with {model_name}")
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"  CUDA OOM error on table {table_id} with {model_name}: {str(e)}")
                logger.error(f"  Stopping processing for {model_name} due to memory constraints")
                # Add error for this table
                results.append({
                    "table_id": table_data.get("table_id"),
                    "model": model_name,
                    "error": f"CUDA out of memory: {str(e)}"
                })
                # Stop processing remaining tables with this model
                break
                    
            except Exception as e:
                logger.error(f"  Error processing table {table_id} with {model_name}: {str(e)}")
                results.append({
                    "table_id": table_data.get("table_id"),
                    "model": model_name,
                    "error": str(e)
                })
        
        # Unload model after processing all tables
        logger.info(f"Completed {len(results)}/{len(tables)} tables with {model_name}")
        self._unload_model()
        
        return results
    
    def process_jsonl_files(
        self,
        input_files: List[str],
        output_dir: str,
        max_tables: Optional[int] = None,
        temperature: float = 0.1,
        job_id: Optional[str] = None
    ) -> None:
        """
        Process multiple JSONL files containing tables.
        
        For each model:
        1. Load the model once
        2. Process all tables from all input files with that model
        3. Save results to separate JSON files (one per input file per model)
        4. Unload model and move to next
        
        Args:
            input_files: List of paths to input JSONL files
            output_dir: Directory for output files
            max_tables: Maximum number of tables to process per file (None = all)
            temperature: Sampling temperature (default: 0.1)
            job_id: Optional SLURM job ID for filename
        """
        output_path = Path(output_dir)
        
        # Validate all input files exist
        valid_files = []
        for input_file in input_files:
            input_path = Path(input_file)
            if not input_path.exists():
                logger.error(f"Input file not found: {input_file}")
            else:
                valid_files.append(input_file)
        
        if not valid_files:
            logger.error("No valid input files to process!")
            return
        
        # Create output directory if needed
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info(f"Multi-Model KPI Extraction Pipeline")
        logger.info(f"Input files: {len(valid_files)}")
        for f in valid_files:
            logger.info(f"  - {Path(f).name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Models: {', '.join(self.models_to_use)}")
        logger.info(f"Max tables per file: {max_tables if max_tables else 'All'}")
        logger.info(f"Temperature: {temperature}")
        logger.info("=" * 70)
        
        # Load all tables from all JSONL files
        logger.info("Loading tables from input files...")
        all_file_tables = []  # List of (input_file, tables) tuples
        
        for input_file in valid_files:
            logger.info(f"  Loading: {Path(input_file).name}")
            tables = []
            errors = 0
            
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line_num, line in enumerate(f_in, 1):
                    # Check if we've hit the limit
                    if max_tables and len(tables) >= max_tables:
                        logger.info(f"    Reached maximum table limit ({max_tables})")
                        break
                    
                    try:
                        table_data = json.loads(line)
                        tables.append(table_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"    Invalid JSON on line {line_num}: {str(e)}")
                        errors += 1
            
            logger.info(f"    Loaded {len(tables)} tables (errors: {errors})")
            
            if tables:
                all_file_tables.append((input_file, tables))
        
        total_tables = sum(len(tables) for _, tables in all_file_tables)
        logger.info(f"Total tables loaded: {total_tables} from {len(all_file_tables)} files")
        
        if not all_file_tables:
            logger.error("No tables to process!")
            return
        
        # Process each model separately
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name in self.models_to_use:
            logger.info("")
            logger.info("#" * 70)
            logger.info(f"Starting extraction with model: {model_name}")
            logger.info("#" * 70)
            
            # Load model once for all files
            if not self._load_model(model_name):
                logger.error(f"Failed to load {model_name}, skipping all files for this model")
                continue
            
            # Process each input file with this model (model stays loaded)
            for file_idx, (input_file, tables) in enumerate(all_file_tables, 1):
                input_filename = Path(input_file).stem  # e.g., "linked_tables(2023)"
                logger.info("")
                logger.info(f"Processing file {file_idx}/{len(all_file_tables)}: {Path(input_file).name}")
                logger.info(f"  Tables in this file: {len(tables)}")
                
                # Process all tables from this file
                model_results = []
                for idx, table_data in enumerate(tables, 1):
                    table_id = table_data.get('table_id', 'unknown')
                    logger.info(f"  [{idx}/{len(tables)}] Processing table: {table_id}")
                    
                    try:
                        result = self.extract_kpis_single_model(
                            table_data,
                            model_name,
                            temperature
                        )
                        
                        # Add table metadata to result
                        result_with_metadata = {
                            "table_id": table_data.get("table_id"),
                            "doc_id": table_data.get("doc_id"),
                            "year": table_data.get("year"),
                            "section_name": table_data.get("section_name"),
                            "title": table_data.get("title"),
                            "extraction_timestamp": datetime.now().isoformat(),
                            "model": model_name,
                            "extraction_result": result
                        }
                        
                        model_results.append(result_with_metadata)
                        
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"    CUDA OOM error on table {table_id}: {str(e)}")
                        logger.error(f"    Stopping processing for {model_name} on this file")
                        model_results.append({
                            "table_id": table_data.get("table_id"),
                            "model": model_name,
                            "error": f"CUDA out of memory: {str(e)}"
                        })
                        break
                        
                    except Exception as e:
                        logger.error(f"    Error processing table {table_id}: {str(e)}")
                        model_results.append({
                            "table_id": table_data.get("table_id"),
                            "model": model_name,
                            "error": str(e)
                        })
                
                # Create filename for this input file + model combination
                if job_id:
                    output_filename = f"{job_id}_{timestamp}_{model_name}_{input_filename}.json"
                else:
                    output_filename = f"kpis_{timestamp}_{model_name}_{input_filename}.json"
                
                output_file = output_path / output_filename
                
                # Write results to JSON file
                logger.info(f"  Writing results to: {output_filename}")
                
                output_data = {
                    "metadata": {
                        "model": model_name,
                        "input_file": str(input_file),
                        "extraction_timestamp": datetime.now().isoformat(),
                        "num_tables_processed": len(model_results),
                        "temperature": temperature,
                        "job_id": job_id
                    },
                    "tables": model_results
                }
                
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    json.dump(output_data, f_out, ensure_ascii=False, indent=2)
                
                # Calculate statistics for this file
                total_kpis = 0
                successful = 0
                failed = 0
                
                for result in model_results:
                    if "error" in result:
                        failed += 1
                    else:
                        successful += 1
                        extraction_result = result.get("extraction_result", {})
                        if "kpis" in extraction_result:
                            total_kpis += len(extraction_result.get("kpis", []))
                
                logger.info(f"  ✓ Completed {Path(input_file).name}:")
                logger.info(f"    - Tables: {len(model_results)}, Successful: {successful}, Failed: {failed}")
                logger.info(f"    - KPIs extracted: {total_kpis}")
            
            # Unload model after processing all files
            logger.info(f"\n✓ {model_name} completed all {len(all_file_tables)} files")
            self._unload_model()
        
        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"All models completed!")
        logger.info(f"  Total input files: {len(all_file_tables)}")
        logger.info(f"  Total tables processed: {total_tables}")
        logger.info(f"  Models used: {len(self.models_to_use)}")
        logger.info(f"  Output directory: {output_dir}")
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
  python extract_kpis_multi_model.py --input tables.jsonl --output-dir ./output
  
  # Process first 10 tables only
  python extract_kpis_multi_model.py --input tables.jsonl --output-dir ./output --max-tables 10
  
  # Use only specific models
  python extract_kpis_multi_model.py --input tables.jsonl --output-dir ./output --models llama-3-8b gemma-3-27b
  
  # With SLURM job ID for filename
  python extract_kpis_multi_model.py --input tables.jsonl --output-dir ./output --job-id $SLURM_JOB_ID
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input JSONL file(s) containing tables (can specify multiple files)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output JSON files (one per model)"
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
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1, range: 0.0-1.0)"
    )
    
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="SLURM job ID for output filename (optional)"
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
    
    # Process the files
    try:
        extractor.process_jsonl_files(
            args.input,
            args.output_dir,
            args.max_tables,
            temperature=args.temperature,
            job_id=args.job_id
        )
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
