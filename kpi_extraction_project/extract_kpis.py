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
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
from datetime import datetime

# Import from project modules
from json_utils import clean_json_response
from logger import logger
from model import MODEL_CONFIGS, ModelManager
from validate import validate_kpi_indexed
from loader import load_tables_from_db, load_existing_results, save_checkpoint

# ============================================================================
# CONFIGURATION
# ============================================================================

# System prompt for KPI extraction
SYSTEM_PROMPT = """Extract ALL numerical KPIs from ALL years in the financial table as JSON.

CRITICAL: Financial tables contain MULTIPLE years (typically current year and previous year).
- Example: A 2015 table contains data for BOTH 2015 AND 2014
- Example: A 2018 table contains data for BOTH 2018 AND 2017
- You MUST extract KPIs from EVERY year column in the table
- DO NOT skip any year columns - extract from ALL of them

Think briefly (max 50 words), then output JSON immediately.

FIELD RULES:
- "name": KPI metric - check in this order merged_headers, title/section, stub_col or row, context
  Examples: "Sales Revenue", "Operating Result", "Vehicle Sales", "Production", "Deliveries" etc
  EXCEPTION: Leave empty ("") ONLY if row is a subtotal/total row with empty row_name
  IMPORTANT RULE: If "Units" is found in merged_headers, then "name" MUST always be "Production (found in title and/or section_name)"
- "key": Entity/segment from row or context
  Examples: "Scania", "Audi", "Volkswagen Group", "Core Brand", "Europe/Other markets"
  EXCEPTION: Leave empty ("") ONLY if row is a subtotal/total row with empty row_name
- "key" and "name" should NEVER both be the same
- "units": Measurement units - Examples: "€ million", "thousand units", "%", "Units", if not explicit infer from context
- "value": Numeric value - Handle European format (comma as decimal): "1,4864" → 1.4864, "−2,5" → -2.5
  Examples: 92718, 30289.0, 16.5, 1.4864 (from "1,4864"), -2.5 (from "−2,5"), null
- "year": Integer year from column - Examples: 2021, 2020, 2019, null
- "row_name": Exact stub_col text at row_idx - Can be empty ("") for total rows
- "col_name": Exact merged_headers text at col_idx - Examples: "2021", "2020", "%", "% change"


INDEXING (ZERO-BASED):
- row_idx: Position in stub_col array (first row = 0)
- col_idx: Data column position starting from 1 (col_idx=0 is row labels, never extract from it)
- All KPIs must have col_idx >= 1
- Verify: stub_col[row_idx] == row_name AND merged_headers[col_idx] == col_name
- IMPORTANT: For hierarchical tables (parent-child rows), use the ACTUAL row index where the data appears, not the parent row
- IMPORTANT: Process ALL columns in merged_headers (except col_idx=0) - do not skip any year columns

EXTRACTION PROCESS:
1. Identify ALL year columns in merged_headers (typically 2+ consecutive years)
2. For EACH row in stub_col, extract values from EVERY year column
3. Create one KPI entry per (row, year) combination
4. Repeat for all rows and all year columns

EXAMPLE:
Given table:
{
  "title": "Key Figures",
  "merged_headers": ["", "2015", "2014", "%"],
  "stub_col": ["Sales revenue (€ million)", "Operating profit (€ million)"],
  "rows": [
    ["Sales revenue (€ million)", "106240", "99764", "6.5"],
    ["Operating profit (€ million)", "2102", "2476", "-15.1"]
  ]
}

Extract 4 KPIs (2 rows × 2 year columns):
{
  "kpis": [
    {
      "name": "Sales revenue",
      "key": "Company",
      "units": "€ million",
      "value": 106240,
      "year": 2015,
      "row_name": "Sales revenue (€ million)",
      "row_idx": 0,
      "col_name": "2015",
      "col_idx": 1
    },
    {
      "name": "Sales revenue",
      "key": "Company",
      "units": "€ million",
      "value": 99764,
      "year": 2014,
      "row_name": "Sales revenue (€ million)",
      "row_idx": 0,
      "col_name": "2014",
      "col_idx": 2
    },
    {
      "name": "Operating profit",
      "key": "Company",
      "units": "€ million",
      "value": 2102,
      "year": 2015,
      "row_name": "Operating profit (€ million)",
      "row_idx": 1,
      "col_name": "2015",
      "col_idx": 1
    },
    {
      "name": "Operating profit",
      "key": "Company",
      "units": "€ million",
      "value": 2476,
      "year": 2014,
      "row_name": "Operating profit (€ million)",
      "row_idx": 1,
      "col_name": "2014",
      "col_idx": 2
    }
  ]
}

OUTPUT FORMAT (JSON only, no extra text):
{
  "kpis": [
    {
      "name": "<metric, mostly from merged_headers, never empty>",
      "key": "<entity from row, never empty>",
      "units": "<units from header or infer fromcontext, never empty>",
      "value": <number or null>,
      "year": <integer or null>,
      "row_name": "<exact stub_col text, can be empty>",
      "row_idx": <integer>,
      "col_name": "<exact merged_headers text>",
      "col_idx": <integer, minimum 1>
    }
  ]
}"""

# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class KPIExtractor:
    """
    Ensemble KPI extractor using multiple LLMs.
    
    This class manages loading, inference, and result aggregation from
    three different language models to extract KPIs from financial tables.
    """
    
    def __init__(self, models_to_use: Optional[List[str]] = None, temperature: float = 0.1):
        """
        Initialize the extractor with specified models.
        
        Args:
            models_to_use: List of model names to load (default: all 3)
        """
        self.models_to_use = models_to_use or list(MODEL_CONFIGS.keys())
        # Use ModelManager for model loading/unloading
        self.model_manager = ModelManager(temperature=temperature)
        
        logger.info(f"Initializing Multi-Model KPI Extractor with {len(self.models_to_use)} models")
        logger.info("Models will be loaded sequentially on-demand to save memory")
        logger.info("=" * 70)
    
    def extract_kpis(
        self,
        table_data: Dict[str, Any],
        model_name: str,
        max_correction_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Extract KPIs using a single model with validation-based correction.
        
        Args:
            table_data: Dictionary containing table information
            model_name: Name of the model to use
            max_correction_iterations: Maximum correction attempts (0 = no validation)
            
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
        if self.model_manager.current_model_name != model_name:
            # Unload previous model if any
            if self.model_manager.current_model is not None:
                self.model_manager.unload_model()
            
            # Load new model
            if not self.model_manager.load_model(model_name):
                return {
                    "kpis": [],
                    "model": model_name,
                    "error": "Failed to load model"
                }
        
        try:
            # Prepare the extraction prompt
            table_json = json.dumps(table_data, ensure_ascii=False)
            prompt = f"{SYSTEM_PROMPT}\n\n### 📥 **Input Placeholder**\n\n```\n{table_json}\n```"

            # Generate response
            config = MODEL_CONFIGS[model_name]
            max_new_tokens = config.get("max_new_tokens", 2048)
            logger.info(f"    → Generating tokens (max: {max_new_tokens} ...")
            generated_text = self.model_manager.generate_text(prompt)
            logger.info(f"    → Generation complete. Decoding output...")
            
            # Save initial extraction output to file
            table_id = table_data.get('table_id', 'unknown')
           
           

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
                    
                    result["model"] = model_name
                    result["num_kpis"] = len(result["kpis"])
                    logger.info(f"    ✓ Extracted {len(result['kpis'])} KPIs from {model_name}")
                    
                    # Run validation loop if enabled
                    if max_correction_iterations > 0 or result["num_kpis"] != 0:
                        result = self._validate_and_correct(
                            table_data,
                            result,
                            model_name,
                            max_correction_iterations
                        )
                    
                    return result
                else:
                    logger.warning(f"  Invalid JSON structure from {model_name}")
                    return {
                        "kpis": [],
                        "model": model_name,
                        "error": "Invalid JSON structure"
                    }
                    
            except json.JSONDecodeError as e:
                # Try to recover by asking LLM to continue/fix the JSON
                table_id = table_data.get('table_id', 'unknown')
                logger.warning(f"  JSON parsing failed for {model_name}: {str(e)}")
                
                result = self._recover_json(
                    cleaned_text,
                    str(e),
                    table_id,
                    model_name,
                    initial_prompt=prompt
                )
                
                if "error" in result:
                    return {
                        "kpis": [],
                        "model": model_name,
                        "error": f"JSON parsing failed: {str(e)}. {result['error']}"
                    }
                
                # Run validation loop if enabled
                if max_correction_iterations > 0:
                    result = self._validate_and_correct(
                        table_data,
                        result,
                        model_name,
                        max_correction_iterations
                    )
                
                return result
                
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
    
    def _validate_and_correct(
        self,
        table_data: Dict[str, Any],
        extraction_result: Dict[str, Any],
        model_name: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """
        Validate extracted KPIs and iteratively correct invalid ones.
        
        Args:
            table_data: Original table data
            extraction_result: Initial extraction result with KPIs
            model_name: Model name for logging
            max_iterations: Maximum correction iterations
            
        Returns:
            Updated extraction result with validation stats
        """
        main_kpis = extraction_result.get("kpis", [])
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"    → Validation iteration {iteration}/{max_iterations}...")
            
            # Validate all KPIs
            valid_kpis = []
            invalid_kpis = []
            
            for kpi in main_kpis:
                validation = validate_kpi_indexed(kpi, table_data)
                
                if validation["is_valid"]:
                    valid_kpis.append(kpi)
                else:
                    invalid_kpis.append({
                        "kpi": kpi,
                        "validation": validation
                    })
            
            total = len(main_kpis)
            valid_count = len(valid_kpis)
            invalid_count = len(invalid_kpis)
            accuracy = (valid_count / total * 100) if total > 0 else 0
            
            logger.info(f"       Valid: {valid_count}/{total} ({accuracy:.1f}%)")
            
            # Store validation stats
            extraction_result["validation_stats"] = {
                "iteration": iteration,
                "total_kpis": total,
                "valid_kpis": valid_count,
                "invalid_kpis": invalid_count,
                "accuracy": accuracy
            }
            
            # If all valid or max iterations reached, stop
            if invalid_count == 0:
                logger.info(f"    ✓ 100% valid after {iteration} iteration(s)")
                extraction_result["kpis"] = valid_kpis
                break
            
            if iteration >= max_iterations:
                logger.warning(f"    ⚠ Max iterations reached. {invalid_count} KPIs still invalid")
                extraction_result["kpis"] = valid_kpis  # Only keep valid KPIs
                extraction_result["invalid_kpis"] = [inv["kpi"] for inv in invalid_kpis]
                break
            
            # Attempt correction
            logger.info(f"    → Correcting {invalid_count} invalid KPIs...")
            corrected_result = self._correct_invalid_kpis(
                table_data,
                main_kpis,
                invalid_kpis,
                model_name,
            )
            
            if "error" in corrected_result:
                # If correction failed due to JSON error, try recovery
                logger.warning(f"       Correction failed: {corrected_result['error']}")
                
                # Check if it's a JSON parsing error and we have raw output
                if "JSON parsing failed" in corrected_result["error"] and "raw_output" in corrected_result:
                    logger.info(f"    → Attempting to recover corrected JSON...")
                    table_id = table_data.get('table_id', 'unknown')
                    
                    # Create the initial prompt for context
                    table_json = json.dumps(table_data, ensure_ascii=False)
                    initial_prompt = f"{SYSTEM_PROMPT}\n\n### 📥 **Input Placeholder**\n\n```\n{table_json}\n```"
                    
                    # Try to recover the malformed correction output
                    recovery_result = self._recover_json(
                        corrected_result["raw_output"],
                        corrected_result.get("parse_error", "Unknown parse error"),
                        table_id,
                        model_name,
                        initial_prompt,
                        original_kpis=main_kpis
                    )
                    
                    if "error" not in recovery_result:
                        logger.info(f"    ✓ Recovery successful after correction failure!")
                        main_kpis = recovery_result.get("kpis", [])
                        continue  # Continue validation loop with recovered KPIs
                    else:
                        logger.warning(f"       Recovery also failed: {recovery_result['error']}")
                
                extraction_result["kpis"] = valid_kpis  # Keep only valid
                extraction_result["invalid_kpis"] = [inv["kpi"] for inv in invalid_kpis]
                break
            
            # Update KPIs with corrected version
            main_kpis = corrected_result.get("kpis", [])
        
        return extraction_result
    
    def _recover_json(
        self,
        incomplete_output: str,
        parse_error: str,
        table_id: str,
        model_name: str,
        initial_prompt: str,
        original_kpis: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Attempt to recover from JSON parsing failure by asking LLM to fix/complete the JSON.
        
        Args:
            incomplete_output: The cleaned but unparseable JSON output
            parse_error: The JSON parsing error message
            table_id: Table identifier for debug files
            model_name: Model name
            initial_prompt: The original extraction prompt (SYSTEM_PROMPT + table data)
            original_kpis: Optional reference KPIs to maintain count consistency
            
        Returns:
            Recovered result dict or error dict
        """
        logger.info(f"    → Attempting JSON recovery with LLM...")
        
        # Create recovery prompt with initial extraction context
        initial_context = f"""INITIAL EXTRACTION PROMPT:
{initial_prompt}

"""
        
        # Add KPI count reference if available
        kpi_count_hint = ""
        if original_kpis is not None:
            kpi_count_hint = f" with exactly {len(original_kpis)} KPIs"
        
        recovery_prompt = f"""{initial_context}The previous JSON output was incomplete or malformed. Here's what was generated:

INCOMPLETE OUTPUT:
{incomplete_output}

PARSING ERROR:
{parse_error}

INSTRUCTIONS:
1. Complete or fix the JSON to make it valid
2. Ensure all brackets and quotes are properly closed
3. Output ONLY the complete, valid JSON""" + kpi_count_hint + """

OUTPUT FORMAT (JSON only, no extra text):
{
  "kpis": [
    {
      "name": "<metric, mostly from merged_headers, never empty>",
      "key": "<entity from row, never empty>",
      "units": "<units from header or infer from context, never empty>",
      "value": <number or null>,
      "year": <integer or null>,
      "row_name": "<exact stub_col text, can be empty>",
      "row_idx": <integer>,
      "col_name": "<exact merged_headers text>",
      "col_idx": <integer, minimum 1>
    }, ...
  ]
}"""
        
        try:
            # Generate recovery
            logger.info(f"    → Generating recovery JSON...")
            recovery_text = self.model_manager.generate_text(recovery_prompt)
            
            # Clean recovery response
            config = MODEL_CONFIGS[model_name]
            if config["includes_prompt_in_output"]:
                recovery_cleaned = clean_json_response(recovery_text, remove_prompt=recovery_prompt)
            else:
                recovery_cleaned = clean_json_response(recovery_text)
            
            # Try parsing recovered JSON
            result = json.loads(recovery_cleaned)
            
            if "kpis" in result and isinstance(result["kpis"], list):
                logger.info(f"    ✓ JSON recovery successful! Extracted {len(result['kpis'])} KPIs")
                
                # Add source model to each KPI
                for kpi in result["kpis"]:
                    kpi["source_model"] = model_name
                
                result["model"] = model_name
                result["num_kpis"] = len(result["kpis"])
                result["recovered"] = True
                
                return result
            else:
                raise ValueError("Invalid JSON structure after recovery")
                
        except Exception as recovery_error:
            logger.error(f"  ✗ JSON recovery failed: {str(recovery_error)}")
            
            # Save debug information
            raw_output_path = f"/ukp-storage-1/ouf/kpi_extraction_project/data/output/raw_cleaned_{model_name}_{table_id}.txt"
            with open(raw_output_path, "a", encoding="utf-8") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Table: {table_id}\n")
                f.write(f"\n{'=' * 50}\n")
                f.write(f"ORIGINAL OUTPUT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(incomplete_output)
                f.write(f"\n{'=' * 50}\n")
                f.write(f"ORIGINAL ERROR\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"{parse_error}\n")
                f.write(f"\n{'=' * 50}\n")
                f.write(f"RECOVERY ATTEMPT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(recovery_cleaned if 'recovery_cleaned' in locals() else "No recovery output")
                f.write(f"\n{'=' * 50}\n")
                f.write(f"RECOVERY ERROR\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"{str(recovery_error)}\n")
            
            logger.warning(f"  Debug file saved: {raw_output_path}")
            
            return {
                "error": f"JSON recovery failed: {str(recovery_error)}"
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _correct_invalid_kpis(
        self,
        table_data: Dict[str, Any],
        all_kpis: List[Dict],
        invalid_kpis: List[Dict],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Use LLM to correct invalid KPIs based on validation feedback.
        
        Args:
            table_data: Original table data
            all_kpis: All KPIs (valid + invalid)
            invalid_kpis: List of invalid KPIs with validation details
            model_name: Model name
            
        Returns:
            Corrected extraction result
        """
        try:
            # Format validation errors
            error_details = []
            for i, inv in enumerate(invalid_kpis, 1):
                kpi = inv["kpi"]
                val = inv["validation"]
                error_msg = f"""ERROR {i}:
  KPI: {json.dumps(kpi, indent=2)}
  Issues: {', '.join(val['errors'])}
  Expected: row_idx={val['row_idx']}, col_idx={val['col_idx']}
  stub_col[{val['row_idx']}] = '{val.get('row_name_match', '')}'
  merged_headers[{val['col_idx']}] = '{val.get('col_name_match', '')}'
  Source: {val['source_cell_value']} (text: \"{val['source_cell_text']}\")
  Extracted: {val['extracted_value']}"""
                
                # Add specific fix instructions from validation
                if val.get('fix_instructions'):
                    error_msg += "\n\n  " + "\n  ".join(val['fix_instructions'])
                
                error_details.append(error_msg)
            
            correction_prompt = f"""The following KPIs are INVALID. Fix them using validation errors.

CRITICAL: The PRIMARY validation is based on row_idx and col_idx (the indices that access the cell).
- row_name and col_name are SECONDARY validation only
- If there's a mismatch, FIX THE INDICES (row_idx/col_idx), NOT the names
- The correct cell is at: data[row_idx][col_idx] = rows[row_idx][col_idx]
- After fixing indices, row_name MUST equal stub_col[row_idx]
- After fixing indices, col_name MUST equal merged_headers[col_idx]

VALIDATION ERRORS:
{''.join(error_details)}

ALL EXTRACTED KPIs (for context):
{json.dumps(all_kpis, ensure_ascii=False, indent=2)}

ORIGINAL TABLE:
{json.dumps(table_data, ensure_ascii=False, indent=2)}

HOW TO FIX ERRORS:

1. KEY = NAME ERROR (IDENTICAL VALUES):
   - This means "key" and "name" have the same value, which is not allowed
   - SOLUTION: Search the table to understand what this KPI represents
   - Look at the table's title, section_name, and context
   - "name" = the KPI metric (what is being measured) - usually from row label (stub_col) or table title
   - "key" = the entity/segment (what/who it applies to) - from context, section, or broader category
   
   EXAMPLES:
   - Table: "Volkswagen Passenger Cars - Production" with row "Golf"
     → name: "Production" (from title), key: "Golf" (from row)
   - Table: "Key Figures" with rows "Sales revenue", "Operating profit"
     → name: "Sales revenue"/"Operating profit" (from row), key: "Company"/"Volkswagen Group" (from context)
   - Table: "Brand Production" with row "Audi" 
     → name: "Production" (from title), key: "Audi" (from row)

2. VALUE MISMATCH: Adjust row_idx or col_idx to point to the cell with the correct value
   - Check if value appears in adjacent cells (row_idx±1 or col_idx±1)
   - Update row_idx/col_idx to the correct position
   - Then update row_name = stub_col[row_idx] and col_name = merged_headers[col_idx]

3. ROW_NAME MISMATCH: The row_idx is pointing to wrong row
   - Find the correct row_idx where stub_col[row_idx] matches the intended row
   - Update row_idx to the correct value
   - Keep row_name as stub_col[row_idx] (don't change the name to match wrong index!)

4. COL_NAME MISMATCH: The col_idx is pointing to wrong column
   - Find the correct col_idx where merged_headers[col_idx] matches the intended year/column
   - Update col_idx to the correct value
   - Keep col_name as merged_headers[col_idx] (don't change the name to match wrong index!)

5. EUROPEAN DECIMALS: Handle comma as decimal separator
   - "1,4864" → 1.4864
   - "−2,5" → -2.5

REMEMBER: 
- Fix indices first, then names will automatically align with stub_col and merged_headers!
- For key=name errors, determine the correct "name" (metric) and "key" (entity) from table context

"""
            
            # Build OUTPUT FORMAT section separately to avoid any string formatting issues
            output_format_section = f"""OUTPUT FORMAT (JSON only, output should be exactly {len(all_kpis)} KPIs):
{{
  "kpis": [
    {{
      "name": "metric name from table",
      "key": "entity or segment",
      "units": "measurement units",
      "value": 12345,
      "year": 2024,
      "row_name": "exact text from stub_col",
      "row_idx": 0,
      "col_name": "exact text from merged_headers",
      "col_idx": 1
    }}
  ]
}}"""
            
            correction_prompt = correction_prompt + "\n" + output_format_section
           
            # Generate correction
            generated_text = self.model_manager.generate_text(correction_prompt)
            
            # Clean and parse
            config = MODEL_CONFIGS[model_name]
            if config["includes_prompt_in_output"]:
                cleaned_text = clean_json_response(generated_text, remove_prompt=correction_prompt)
            else:
                cleaned_text = clean_json_response(generated_text)
            
            # Save correction attempt to debug file
            table_id = table_data.get('table_id', 'unknown')
            correction_output_path = f"/ukp-storage-1/ouf/kpi_extraction_project/data/output/correction_{model_name}_{table_id}.txt"
            with open(correction_output_path, "a", encoding="utf-8") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Table: {table_id}\n")
                f.write(f"\n{'=' * 50}\n")
                f.write(f"CORRECTION PROMPT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(correction_prompt)
                f.write(f"\n\n{'=' * 50}\n")
                f.write(f"GENERATED TEXT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(generated_text)
                f.write(f"\n\n{'=' * 50}\n")
                f.write(f"CLEANED TEXT\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(cleaned_text)
                f.write("\n")
            
            logger.info(f"       Correction output saved to: {correction_output_path}")
            
            result = json.loads(cleaned_text)
            
            if "kpis" in result and isinstance(result["kpis"], list):
                return result
            else:
                return {"error": "Invalid correction JSON structure"}
                
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_output": cleaned_text if 'cleaned_text' in locals() else "",
                "parse_error": str(e)
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_database(
        self,
        db_path: str,
        output_dir: str,
        year_filter: Optional[str] = None,
        max_tables: Optional[int] = None,
        job_id: Optional[str] = None,
        max_correction_iterations: int = 3,
        resume: bool = True
    ) -> None:
        """
        Process tables directly from database instead of JSONL files.
        Supports resuming from checkpoints if job is preempted.
        
        For each model:
        1. Load tables from database with optional year filter
        2. Check for existing checkpoint and resume if available
        3. Process tables incrementally, saving after each table
        4. Unload model and move to next
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory for output files
            year_filter: Optional year to filter (e.g., "2019")
            max_tables: Maximum number of tables to process (None = all)
            job_id: Optional SLURM job ID for filename
            max_correction_iterations: Maximum validation/correction iterations (0 = disabled)
            resume: Whether to resume from checkpoint if available (default: True)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info(f"Multi-Model KPI Extraction Pipeline (Database Mode)")
        logger.info(f"Database: {db_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Year filter: {year_filter if year_filter else 'Years 2015-2024'}")
        logger.info(f"Models: {', '.join(self.models_to_use)}")
        logger.info(f"Max tables: {max_tables if max_tables else 'All'}")
        logger.info(f"Resume from checkpoint: {resume}")
        logger.info("=" * 70)
        
        # Define years to process
        if year_filter:
            years_to_process = [year_filter]
        else:
            years_to_process = [str(year) for year in range(2015, 2025)]  # 2015-2024
        
        logger.info(f"Years to process: {', '.join(years_to_process)}")
        
        # Process each model separately
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name in self.models_to_use:
            logger.info("")
            logger.info("#" * 70)
            logger.info(f"Starting extraction with model: {model_name}")
            logger.info("#" * 70)
            
            # Load model once for all years
            if not self.model_manager.load_model(model_name):
                logger.error(f"Failed to load {model_name}, skipping")
                continue
            
            # Process each year separately
            for year in years_to_process:
                logger.info("")
                logger.info(f"Loading tables for year: {year}")
                logger.info("-" * 50)
                
                # Load tables for this specific year from database
                year_tables = load_tables_from_db(db_path, year_filter=year, max_tables=max_tables)
                
                if not year_tables:
                    logger.warning(f"No tables found for year {year}, skipping")
                    continue
                
                logger.info(f"Loaded {len(year_tables)} tables for year {year}")
                
                # Determine output file name for this year (with checkpoint prefix)
                job_suffix = f"_{job_id}" if job_id else ""
                checkpoint_file = output_path / f"checkpoint_{timestamp}{job_suffix}_{model_name}_year{year}_kpis.json"
                output_file = checkpoint_file  # Start with checkpoint file
                
                # Check for existing checkpoint
                processed_ids = set()
                model_results = []
                
                if resume:
                    # Try to find existing checkpoint file for this model/year combination
                    # Look for any file matching the pattern (with checkpoint prefix)
                    pattern = f"checkpoint_*_{model_name}_year{year}_kpis.json"
                    existing_files = list(output_path.glob(pattern))
                    
                    if existing_files:
                        # Use the most recent checkpoint file
                        checkpoint_file = max(existing_files, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Found checkpoint file: {checkpoint_file.name}")
                        model_results, processed_ids = load_existing_results(checkpoint_file)
                        
                        # Use the existing checkpoint file
                        output_file = checkpoint_file
                        
                        if processed_ids:
                            logger.info(f"  Resuming from checkpoint: {len(processed_ids)} tables already processed")
                
                # Filter out already processed tables
                tables_to_process = [t for t in year_tables if t.get('table_id') not in processed_ids]
                
                if not tables_to_process:
                    logger.info(f"All tables already processed for {model_name} year {year}, skipping")
                    continue
                
                logger.info(f"Tables to process: {len(tables_to_process)} (skipped: {len(processed_ids)})")
                
                # Process tables incrementally with checkpointing
                total_to_process = len(tables_to_process)
                for idx, table_data in enumerate(tables_to_process, 1):
                    table_id = table_data.get('table_id', 'unknown')
                    overall_idx = len(processed_ids) + idx
                    total_tables = len(year_tables)
                    
                    logger.info(f"[{overall_idx}/{total_tables}] Processing {table_id} with {model_name}")
                    
                    try:
                        # Extract KPIs
                        result = self.extract_kpis(
                            table_data,
                            model_name,
                            max_correction_iterations=max_correction_iterations
                        )
                        
                        # Add table metadata
                        result['table_id'] = table_id
                        result['table_data'] = table_data
                        result['processing_timestamp'] = datetime.now().isoformat()
                        
                        # Append to results
                        model_results.append(result)
                        processed_ids.add(table_id)
                        
                        # Log summary
                        num_kpis = len(result.get('kpis', []))
                        logger.info(f"    → Extracted {num_kpis} KPIs")
                        
                        # Save checkpoint after each table
                        save_checkpoint(output_file, model_results, model_name, year)
                        logger.info(f"    → Checkpoint saved ({len(model_results)} tables total)")
                        
                    except Exception as e:
                        logger.error(f"    ✗ Error processing {table_id}: {str(e)}")
                        # Save error in result
                        error_result = {
                            'table_id': table_id,
                            'error': str(e),
                            'processing_timestamp': datetime.now().isoformat()
                        }
                        model_results.append(error_result)
                        processed_ids.add(table_id)
                        
                        # Save checkpoint even with error
                        save_checkpoint(output_file, model_results, model_name, year)
                        logger.info(f"    → Checkpoint saved (with error)")
                
                # Final save with complete metadata (remove checkpoint prefix)
                logger.info("")
                logger.info(f"Finalizing results: {output_file.name}")
                final_data = {
                    "metadata": {
                        "model": model_name,
                        "year": year,
                        "completed": datetime.now().isoformat(),
                        "total_tables_processed": len(model_results),
                        "total_kpis_extracted": sum(len(r.get('kpis', [])) for r in model_results if 'kpis' in r),
                        "checkpoint": False,
                        "complete": True
                    },
                    "results": model_results
                }
                
                # Create final filename without checkpoint prefix
                final_filename = output_file.name.replace("checkpoint_", "")
                final_file = output_path / final_filename
                
                with open(final_file, 'w', encoding='utf-8') as f_out:
                    json.dump(final_data, f_out, indent=2, ensure_ascii=False)
                
                # Remove checkpoint file if it exists and is different from final
                if output_file.exists() and output_file != final_file:
                    output_file.unlink()
                    logger.info(f"  Removed checkpoint file: {output_file.name}")
                
                logger.info(f"  Final file: {final_filename}")
                logger.info(f"✓ Completed year {year}")
            
            # Unload model after processing all years
            logger.info(f"\nUnloading {model_name}...")
            self.model_manager.unload_model()
            
            logger.info(f"✓ Completed {model_name}")
        
        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"All models completed!")
        logger.info(f"  Years processed: {', '.join(years_to_process)}")
        logger.info(f"  Models used: {len(self.models_to_use)}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info("=" * 70)
    
    def process_jsonl_files(
        self,
        input_files: List[str],
        output_dir: str,
        max_tables: Optional[int] = None,
        job_id: Optional[str] = None,
        max_correction_iterations: int = 3
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
            job_id: Optional SLURM job ID for filename
            max_correction_iterations: Maximum validation/correction iterations (0 = disabled)
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
            if not self.model_manager.load_model(model_name):
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
                        result = self.extract_kpis(
                            table_data,
                            model_name,
                            max_correction_iterations
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
            self.model_manager.unload_model()
        
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
  # Process tables from database for specific year
  python extract_kpis.py --db data/pack_context.db --output-dir ./output --year 2019
  
  # Process all tables from database with specific model
  python extract_kpis.py --db data/pack_context.db --output-dir ./output --models deepseek-v2.5
  
  # Process first 10 tables only
  python extract_kpis.py --db data/pack_context.db --output-dir ./output --max-tables 10
  
  # Legacy: Process JSONL files
  python extract_kpis.py --input tables.jsonl --output-dir ./output
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--db",
        type=str,
        help="Path to SQLite database containing tables"
    )
    input_group.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Path(s) to input JSONL file(s) containing tables (legacy mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output JSON files (one per model)"
    )
    
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Filter tables by year (e.g., 2019) - only for database mode"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Specific models to use (default: all)"
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
        default=0.0,
        help="Sampling temperature (default: 0.0, range: 0.0-1.0)"
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
    
    parser.add_argument(
        "--max-correction-iterations",
        type=int,
        default=3,
        help="Maximum validation/correction iterations per table (default: 3, 0=disabled)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from checkpoint (start fresh)"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize extractor
    extractor = KPIExtractor(models_to_use=args.models, temperature=args.temperature)
    
    # Process based on input mode
    try:
        if args.db:
            # Database mode (new)
            extractor.process_database(
                args.db,
                args.output_dir,
                year_filter=args.year,
                max_tables=args.max_tables,
                job_id=args.job_id,
                max_correction_iterations=args.max_correction_iterations,
                resume=not args.no_resume
            )
        else:
            # JSONL file mode (legacy)
            extractor.process_jsonl_files(
                args.input,
                args.output_dir,
                args.max_tables,
                job_id=args.job_id,
                max_correction_iterations=args.max_correction_iterations
            )
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
