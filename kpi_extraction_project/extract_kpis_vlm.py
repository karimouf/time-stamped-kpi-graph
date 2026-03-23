#!/usr/bin/env python3
"""
VLM-Based KPI Extraction System
================================

Extracts Key Performance Indicators from financial table images using Vision-Language Models.
Utilizes the Qwen3-VL-30B-A3B-Instruct model to process table images and extract
structured KPI data in the same format as the text-based extraction system.

This script is designed for extracting KPIs when data is available only as images
(screenshots, scans, or exported table images) rather than structured JSON.

Author: Karim Ouf
Date: January 2026
"""

import json
import torch
import time
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

# Import from project modules
from json_utils import clean_json_response
from logger import logger
from model import MODEL_CONFIGS, ModelManager
from validate import validate_kpi_indexed, validate_kpis

# ============================================================================
# CONFIGURATION
# ============================================================================

# VLM model name
VLM_MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"

# System prompt for VLM-based KPI extraction
VLM_SYSTEM_PROMPT = """You are analyzing a financial table image to extract Key Performance Indicators (KPIs).

## INPUT SOURCES

You are provided with TWO sources of information:
1. **TABLE IMAGE**: A cropped image showing a specific table from the report
2. **SECTION CONTEXT**: Information about which section of the report this table belongs to

The section context tells you what part of the annual report this table comes from (e.g., "Brands and Business Fields", "Volkswagen Passenger Cars", "Audi", etc.). This is crucial for understanding:
- Which business unit or brand the data relates to
- What type of metrics to expect (sales, production, financial performance, etc.)
- How to properly categorize and name the KPIs

Use the section context to:
- Set appropriate 'key' values that reflect the business unit/brand
- Understand the business context of the metrics
- Properly categorize KPIs based on their organizational context

## IMPORTANT: VERIFY THE IMAGE IS A DATA TABLE

Before extracting, check if the image actually contains a data table with numerical KPIs:
- **Valid tables** have: rows/columns, numerical data, headers, organized structure
- **NOT valid tables**: organizational charts, flowcharts, diagrams, images without numerical data, text-only content, logos, blank spaces

If the image does NOT contain a valid data table with numerical KPIs, return:
```json
{
  "kpis": [],
  "metadata": {
    "table_description": "Not a data table",
    "extraction_notes": "Image does not contain numerical data table (detected: organizational chart/diagram/text-only/etc)",
  }
}
```

## YOUR TASK

Examine the table image carefully and extract ALL numerical KPIs in a structured JSON format.

## STEP 1: ANALYZE THE IMAGE

Before extracting, identify:
- **Table Structure**: How many columns and rows? Are there headers, merged cells?
- **Metrics**: What KPIs are shown? (Revenue, Profit, Deliveries, Production, etc.)
- **Entities/Segments**: What organizations, divisions, or categories are listed?
- **Time Periods**: What years or periods are shown in column headers?
- **Geographic Scope**: Are there country/region breakdowns in the rows?
- **Units**: What measurement units are used? (€ million, thousand units, %, etc.)

## STEP 2: EXTRACT ALL KPIs

CRITICAL RULES:
1. **Multi-Year Data**: Financial tables typically show 2+ years (e.g., 2023 and 2022)
   - Extract KPIs from EVERY year column - do not skip any
   - Create separate KPI entries for each year

2. **Complete Extraction**: Extract from ALL rows and ALL data columns
   - Don't skip rows with data, even if values seem less important
   - Skip completely empty rows (no data in any column) but count them for row indexing
   - Include totals, subtotals, and detail rows
   - For rows with some empty cells, extract KPIs using null for missing values

3. **Handle Missing Data**: Use null for empty cells or dashes (-, –, N/A)

## FIELD REQUIREMENTS

Each KPI must have these fields:

- **name** (string): The GENERIC KPI type being measured
    - Keep it generic and reusable across entities
    - Do NOT include specific business unit, brand, country, or segment names in `name`
    - NEVER empty

- **key** (string): The entity/segment (CANNOT be a country name)
    - Use the MOST SPECIFIC business entity available from row labels and section context
    - Specificity rule: if a candidate key is broad/generic (e.g., category-level wording like "trucks"), resolve it to the most specific entity/sub-entity/business unit available from row/header/section context
    - Vehicle-category disambiguation is mandatory: when rows include categories like "Trucks", "Buses", "Light Commercial Vehicles", etc., bind them to their parent entity/group from the same table hierarchy
    - Use combined keys such as "Scania Trucks", "MAN Buses", "Navistar Trucks", "Volkswagen Truck & Bus Buses" instead of generic keys like "Trucks" or "Buses" alone
    - Never output ambiguous combinations like "Production Trucks" when the producer/entity is identifiable; include the producer/entity in `key`
    - Example: keep `name` as a generic KPI type such as "production", and set `key` to the specific entity label found in context rather than a broad category word
    - If row shows a country, keep country in `country` and use business entity in `key`
    - For totals with no row label: use a business-scope key such as "Total" or "Subtotal"
    - NEVER empty

- **country** (string): Geographic location
  - If row represents a country: use that country name
  - Otherwise: check table title/headers for country mentions
  - Default: "Worldwide" if no country identified
  - NEVER empty

- **value** (number or null): The numerical value
  - Parse carefully: "1,234.56" → 1234.56
  - European format: "1.234,56" → 1234.56
  - Negative: "-123" or "−123" → -123
  - Missing data: null
  - Remove footnote markers (¹, ², *, ^1, etc.)

- **year** (integer or null): The year from column header
  - null if year cannot be determined

- **units** (string): Measurement units
  - Extract from headers or infer from context
  - NEVER empty

- **row_idx** (integer): Zero-based row index in the table
  - ALWAYS start from 0 for the first data row (after headers)
  - Header rows do NOT count in the index
  - Increment by 1 for each subsequent row
  - IMPORTANT: Skip completely empty rows (rows with no data in any column) but still count them for row_idx increment
  - Example: If row 2 is completely empty, skip extracting KPIs from it but the next non-empty row becomes row_idx=3
  - However, if a row has some empty values but contains at least one data value, extract KPIs from that row (using null for empty cells)
  - IMPORTANT: First data row is ALWAYS row_idx=0

- **col_idx** (integer): One-based column index for data values
  - ALWAYS start from 1 for the first value column
  - Column 0 contains row labels/names and should be SKIPPED
  - First value column = 1, second value column = 2, etc.
  - IMPORTANT: NEVER use col_idx=0 (that's the row label column use "col_idx=1" for the first data column)

## CRITICAL CONSTRAINT

**name and key must be COMPLETELY DIFFERENT**: 
- The 'name' field (metric) and 'key' field (entity/segment) cannot have identical values OR similar meanings
- They must represent fundamentally different concepts:
  - 'name' = WHAT is being measured (the metric/KPI type)
  - 'key' = WHO/WHAT ENTITY the measurement applies to (company, division, category, etc.)
- Disambiguation requirement: whenever `key` is too broad, keep `name` generic and refine `key` to the most specific entity available in the table context
- For hierarchical row structures (entity parent row + child row like Trucks/Buses), preserve both levels in `key` so the extracted KPI clearly indicates which trucks/buses are being measured
- This ensures complete separation between WHAT is measured and WHICH ENTITY it applies to

## INDEX CONSISTENCY VALIDATION

CRITICAL: After extracting all KPIs, verify index consistency:
1. **Cross-verify with actual values**:
   - Ensure row_idx and col_idx accurately point to the correct cell
   - Double-check by matching extracted value with cell position

## SPECIAL CASES

### Total/Subtotal Rows
- If a row represents a total with no specific entity name:
  - name: the metric being totaled
  - key: "Total", "Subtotal", or describe scope

### Percentage Columns
- Often show year-over-year changes
- Extract as separate KPIs with:
  - units: "%"
  - year: null (or the comparison year if clear)
  - value: the percentage (15.5 for 15.5%, not 0.155)

### Footnotes and Annotations
- Remove all footnote markers from values
- Ignore footnote text in extraction

## OUTPUT FORMAT

Return ONLY valid JSON with this structure, no additional text:

```json
{
  "kpis": [
    {
      "name": "string (metric name)",
      "key": "string (entity/segment)",
      "country": "string (location)",
      "value": number or null,
      "year": integer or null,
      "units": "string (measurement unit)",
      "row_idx": integer (zero-based),
      "col_idx": integer (one-based, starts at 1)
    }
  ],
}
```

## QUALITY CHECKS

Before finalizing:
1. ✓ All required fields present and non-empty (except value/year can be null)
2. ✓ One KPI per (row, year column) combination
3. ✓ Values match what's visible in the image
4. ✓ Units are explicit and consistent
5. ✓ Years correctly identified from column headers
6. ✓ All year columns processed (not just the first one)
7. ✓ row_idx and col_idx are consistent across all KPIs
8. ✓ Same row has same row_idx, same column has same col_idx
9. ✓ Vehicle-category keys are fully disambiguated (not generic "Trucks"/"Buses" when parent entity is available)

Extract ALL data comprehensively and accurately."""


def _resolve_page_markdown_path(image_path: str, markdown_path: Optional[str]) -> Optional[Path]:
    if markdown_path:
        return Path(markdown_path)

    image_path_obj = Path(image_path)
    candidates = []

    if image_path_obj.parent.name.startswith("page_") and "_table_" in image_path_obj.parent.name:
        page_prefix = image_path_obj.parent.name.split("_table_")[0]
        candidates.append(image_path_obj.parent.parent / f"{page_prefix}.md")
        candidates.append(image_path_obj.parent.parent / f"{page_prefix}.markdown")

    candidates.append(image_path_obj.parent / "page.md")
    candidates.append(image_path_obj.parent / "page.markdown")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_page_markdown_context(image_path: str, markdown_path: Optional[str]) -> Optional[str]:
    path_obj = _resolve_page_markdown_path(image_path, markdown_path)
    if not path_obj:
        return None
    if not path_obj.exists():
        logger.warning(f"  Page markdown file not found: {path_obj}")
        return None
    try:
        return path_obj.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(f"  Failed to read page markdown file {path_obj}: {exc}")
        return None

# ============================================================================
# VLM KPI EXTRACTOR CLASS
# ============================================================================

class VLMKPIExtractor:
    """
    Vision-Language Model KPI extractor for processing table images.
    
    Uses Qwen3-VL-30B model to analyze financial table images and extract
    structured KPI data.
    """
    
    def __init__(self, temperature: float = 0.1, model_name: Optional[str] = None):
        """
        Initialize the VLM extractor.
        
        Args:
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            model_name: Optional model name (defaults to Qwen3-VL-30B-A3B-Instruct)
        """
        self.model_name = model_name or VLM_MODEL_NAME
        self.model_manager = ModelManager(temperature=temperature)
        
        logger.info(f"Initializing VLM KPI Extractor with {self.model_name}")
        logger.info("=" * 70)
    
    def load_model(self) -> bool:
        """
        Load the VLM model.
        
        Returns:
            True if successful, False otherwise
        """
        return self.model_manager.load_vlm_model(self.model_name)
    
    def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        self.model_manager.unload_model()
    
    def _extract_partial_kpis(self, truncated_text: str) -> List[Dict[str, Any]]:
        """
        Extract valid KPI entries from a truncated JSON output.
        
        When the model hits max_new_tokens, the JSON is often cut mid-way through
        a KPI entry. This function salvages all complete KPI objects from the
        truncated output.
        
        Args:
            truncated_text: The raw or cleaned text that failed JSON parsing
            
        Returns:
            List of successfully parsed KPI dictionaries
        """
        salvaged_kpis = []
        
        # Find the kpis array start
        kpis_start = truncated_text.find('"kpis"')
        if kpis_start == -1:
            return salvaged_kpis
        
        # Find the opening bracket of the array
        bracket_start = truncated_text.find('[', kpis_start)
        if bracket_start == -1:
            return salvaged_kpis
        
        # Extract individual KPI objects using brace matching
        depth = 0
        obj_start = -1
        i = bracket_start + 1
        
        while i < len(truncated_text):
            char = truncated_text[i]
            
            if char == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and obj_start != -1:
                    # Found a complete object
                    obj_text = truncated_text[obj_start:i + 1]
                    try:
                        kpi = json.loads(obj_text)
                        # Validate it has required KPI fields
                        if isinstance(kpi, dict) and 'name' in kpi and 'value' in kpi:
                            salvaged_kpis.append(kpi)
                    except json.JSONDecodeError:
                        pass  # Skip malformed objects
                    obj_start = -1
            elif char == ']' and depth == 0:
                break  # End of array
            
            i += 1
        
        return salvaged_kpis
    
    def _get_last_complete_kpi_context(self, kpis: List[Dict[str, Any]], max_kpis: int = 3) -> str:
        """
        Get a string representation of the last few KPIs for continuation context.
        
        Args:
            kpis: List of already extracted KPIs
            max_kpis: Maximum number of trailing KPIs to include
            
        Returns:
            JSON string of the last few KPIs
        """
        if not kpis:
            return "No KPIs extracted yet."
        
        tail = kpis[-max_kpis:]
        return json.dumps(tail, indent=2, ensure_ascii=False)
    
    def _continue_and_merge_output(
        self,
        image_path: str,
        truncated_text: str,
        partial_kpis: List[Dict[str, Any]],
        max_continuations: int = 2
    ) -> Dict[str, Any]:
        """
        Continue generating from a truncated output and merge results.
        
        Similar to ChatGPT's "Continue generating" button: detects where the
        output was cut off, prompts the model to continue from that point,
        and merges the partial KPIs with the continuation KPIs.
        
        Handles deduplication by matching on (row_idx, col_idx, year) tuples
        to avoid counting the same cell twice.
        
        Args:
            image_path: Path to the table image
            truncated_text: The truncated raw output text
            partial_kpis: KPIs already salvaged from the truncated output
            max_continuations: Maximum number of continuation attempts
            
        Returns:
            Dictionary with merged KPIs and metadata, or error dict
        """
        merged_kpis = list(partial_kpis)  # Start with what we already have
        total_continuation_time = 0.0
        continuations_used = 0
        
        for attempt in range(1, max_continuations + 1):
            logger.info(f"    → Continuation {attempt}/{max_continuations}...")
            
            # Build context showing the last few KPIs so the model knows where to resume
            last_kpis_context = self._get_last_complete_kpi_context(merged_kpis)
            
            # Determine last row_idx to tell the model where to continue
            last_row_idx = max((kpi.get("row_idx", 0) for kpi in merged_kpis), default=-1) if merged_kpis else -1
            
            continuation_prompt = f"""You were extracting KPIs from a financial table image but the output was truncated.

## WHAT WAS ALREADY EXTRACTED

{len(merged_kpis)} KPIs were successfully extracted. Here are the LAST few KPIs extracted:

```json
{last_kpis_context}
```

The last extracted row_idx was {last_row_idx}.

## YOUR TASK

CONTINUE extracting the REMAINING KPIs from the table image, starting from where the previous extraction stopped.

CRITICAL RULES:
1. Do NOT re-extract KPIs that were already extracted (row_idx <= {last_row_idx} for already-covered columns)
2. Continue with the NEXT rows/columns that haven't been extracted yet
3. Maintain the same field format and naming conventions
4. If there are year columns not yet extracted for existing rows, include those too

OUTPUT FORMAT - Return ONLY the remaining KPIs as valid JSON:
{{
  "kpis": [
    {{
      "name": "string",
      "key": "string",
      "country": "string",
      "value": number or null,
      "year": integer or null,
      "units": "string",
      "row_idx": integer,
      "col_idx": integer
    }}
  ],
  "is_complete": true
}}

Set "is_complete" to true if you have extracted ALL remaining KPIs from the table, or false if there are still more."""
            
            try:
                cont_start = time.time()
                continuation_text = self.model_manager.generate_vlm_output(
                    image_path=image_path,
                    prompt=continuation_prompt
                )
                cont_time = time.time() - cont_start
                total_continuation_time += cont_time
                continuations_used += 1
                
                # Clean and parse
                continuation_cleaned = clean_json_response(continuation_text)
                continuation_result = json.loads(continuation_cleaned)
                
                if "kpis" in continuation_result and isinstance(continuation_result["kpis"], list):
                    new_kpis = continuation_result["kpis"]
                    logger.info(f"    ✓ Continuation {attempt}: got {len(new_kpis)} additional KPIs in {cont_time:.2f}s")
                    
                    # Deduplicate: build set of (row_idx, col_idx, year) from existing KPIs
                    existing_keys = set()
                    for kpi in merged_kpis:
                        key = (kpi.get("row_idx"), kpi.get("col_idx"), kpi.get("year"))
                        existing_keys.add(key)
                    
                    # Add only truly new KPIs
                    added_count = 0
                    for kpi in new_kpis:
                        key = (kpi.get("row_idx"), kpi.get("col_idx"), kpi.get("year"))
                        if key not in existing_keys:
                            merged_kpis.append(kpi)
                            existing_keys.add(key)
                            added_count += 1
                    
                    duplicate_count = len(new_kpis) - added_count
                    if duplicate_count > 0:
                        logger.info(f"      Skipped {duplicate_count} duplicate KPIs, added {added_count} new")
                    
                    # Check if extraction is complete
                    is_complete = continuation_result.get("is_complete", True)
                    if is_complete:
                        logger.info(f"    ✓ Model reports extraction is complete")
                        break
                    elif len(new_kpis) == 0:
                        logger.info(f"    ✓ No new KPIs returned, assuming extraction is complete")
                        break
                else:
                    logger.warning(f"    ⚠ Continuation {attempt}: invalid JSON structure")
                    break
                    
            except json.JSONDecodeError as cont_error:
                logger.warning(f"    ⚠ Continuation {attempt}: JSON parsing failed - {str(cont_error)}")
                # Try to salvage partial KPIs from this continuation too
                try:
                    cont_cleaned = clean_json_response(continuation_text)
                    salvaged = self._extract_partial_kpis(cont_cleaned)
                    if salvaged:
                        existing_keys = set()
                        for kpi in merged_kpis:
                            key = (kpi.get("row_idx"), kpi.get("col_idx"), kpi.get("year"))
                            existing_keys.add(key)
                        
                        added = 0
                        for kpi in salvaged:
                            key = (kpi.get("row_idx"), kpi.get("col_idx"), kpi.get("year"))
                            if key not in existing_keys:
                                merged_kpis.append(kpi)
                                existing_keys.add(key)
                                added += 1
                        logger.info(f"      Salvaged {added} additional KPIs from truncated continuation")
                except Exception:
                    pass
                break
                
            except Exception as cont_exc:
                logger.warning(f"    ⚠ Continuation {attempt}: failed - {str(cont_exc)}")
                break
        
        # Sort merged KPIs by (row_idx, col_idx) for consistent ordering
        merged_kpis.sort(key=lambda k: (k.get("row_idx", 0), k.get("col_idx", 0)))
        
        return {
            "kpis": merged_kpis,
            "continuation_time_seconds": round(total_continuation_time, 2),
            "continuations_used": continuations_used,
            "total_kpis_merged": len(merged_kpis),
            "partial_kpis_salvaged": len(partial_kpis),
        }

    def extract_kpis_from_image(
        self,
        image_path: str,
        context: Optional[str] = None,
        title: Optional[str] = None,
        max_correction_iterations: int = 0,
        db_path: Optional[str] = None,
        validate_results: bool = True,
        page: Optional[int] = None,
        year: Optional[int] = None,
        table_idx: Optional[int] = None,
        bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract KPIs from a single table image.
        
        Args:
            image_path: Path to the table image file
            context: Optional additional context about the table
            page_markdown_path: Path to page markdown file for context
            max_correction_iterations: Maximum validation/correction attempts (0 = no validation)
            db_path: Optional path to SQLite database for validation
            validate_results: Whether to validate extracted KPIs (requires db_path)
            page: Optional page number for validation filtering
            year: Optional year for validation filtering
            
        Returns:
            Dictionary with extracted KPIs, metadata, and validation statistics
        """
        if self.model_manager.current_llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Load and validate image
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                logger.error(f"  ✗ Image file not found: {image_path}")
                return {
                    "kpis": [],
                    "image_path": str(image_path),
                    "error": "Image file not found"
                }
            
            logger.info(f"  Processing image: {image_path_obj.name}")
            
            # Load image to check size
            image = Image.open(image_path).convert("RGB")
            logger.info(f"    Image size: {image.size[0]}x{image.size[1]}")
            
            # Prepare the prompt
            user_prompt = VLM_SYSTEM_PROMPT
            if context:
                user_prompt += f"\n\n## ADDITIONAL CONTEXT\n\n{context}"

            # Add table title to the prompt if provided
            if title:
                user_prompt += f"\n\n## TABLE TITLE\n\nTable title: {title}\n"
            
            # Start timing for inference
            start_time = time.time()
            
            # Generate response using ModelManager's generate_vlm_output
            output_text = self.model_manager.generate_vlm_output(
                image_path=str(image_path),
                prompt=user_prompt
            )
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            logger.info(f"    → Generation complete in {inference_time:.2f}s. Parsing JSON...")
            
            # Clean and parse response
            cleaned_text = clean_json_response(output_text)
            
            try:
                result = json.loads(cleaned_text)
                
                # Validate structure
                if "kpis" in result and isinstance(result["kpis"], list):
                    # Add source information to each KPI
                    for kpi in result["kpis"]:
                        kpi["source_model"] = self.model_name
                        kpi["source_image"] = str(image_path_obj.name)
                    
                    result["model"] = self.model_name
                    result["image_path"] = str(image_path)
                    result["page_markdown_used"] = False
                    result["num_kpis"] = len(result["kpis"])
                    result["inference_time_seconds"] = round(inference_time, 2)
                    logger.info(f"    ✓ Extracted {len(result['kpis'])} KPIs from image in {inference_time:.2f}s")
                    
                    # Run validation if db_path is provided
                    if validate_results and db_path and result["kpis"]:
                        logger.info(f"    → Running validation against database...")
                        try:
                            validation_output = validate_kpis(
                                kpis=result["kpis"],
                                db_path=db_path,
                                year=year,
                                page=page,
                                bucket=bucket,
                                table_idx=table_idx,
                            )
                            
                            # Add validation statistics to result
                            result["validation_statistics"] = validation_output["statistics"]
                            result["validation_summary"] = {
                                "total_kpis": validation_output["statistics"]["total_kpis"],
                                "valid_kpis": validation_output["statistics"]["valid_kpis"],
                                "invalid_kpis": validation_output["statistics"]["invalid_kpis"],
                                "accuracy": validation_output["statistics"]["accuracy"],
                                "confidence_avg": validation_output["statistics"]["confidence_avg"]
                            }
                            
                            # Add detailed validation results (valid and invalid KPIs with error details)
                            result["valid_kpis"] = validation_output["valid_kpis"]
                            result["invalid_kpis"] = validation_output["invalid_kpis"]
                            
                            logger.info(
                                f"    ✓ Validation complete: "
                                f"{validation_output['statistics']['valid_kpis']}/{validation_output['statistics']['total_kpis']} valid "
                                f"(Accuracy: {validation_output['statistics']['accuracy']:.1f}%)"
                            )
                            
                        except Exception as e:
                            logger.warning(f"    ⚠ Validation failed: {str(e)}")
                            result["validation_error"] = str(e)
                    
                    return result
                else:
                    logger.warning(f"  Invalid JSON structure from VLM")
                    return {
                        "kpis": [],
                        "image_path": str(image_path),
                        "model": self.model_name,
                        "inference_time_seconds": round(inference_time, 2),
                        "error": "Invalid JSON structure"
                    }
                    
            except json.JSONDecodeError as e:
                logger.warning(f"  JSON parsing failed: {str(e)}")
                
                # Step 1: Salvage any complete KPIs from the truncated output
                partial_kpis = self._extract_partial_kpis(cleaned_text)
                logger.info(f"    → Salvaged {len(partial_kpis)} complete KPIs from truncated output")
                
                # Step 2: Continue generation to get remaining KPIs
                if partial_kpis:
                    # We have partial data - this was likely a max_tokens truncation
                    logger.info(f"    → Attempting continuation (like ChatGPT 'Continue generating')...")
                    
                    merge_result = self._continue_and_merge_output(
                        image_path=str(image_path),
                        truncated_text=cleaned_text,
                        partial_kpis=partial_kpis,
                        max_continuations=2
                    )
                    
                    merged_kpis = merge_result["kpis"]
                    inference_time += merge_result["continuation_time_seconds"]
                    
                    if merged_kpis:
                        logger.info(
                            f"    ✓ Merged result: {len(merged_kpis)} KPIs "
                            f"({merge_result['partial_kpis_salvaged']} salvaged + "
                            f"{len(merged_kpis) - merge_result['partial_kpis_salvaged']} from continuation)"
                        )
                        
                        # Add source information
                        for kpi in merged_kpis:
                            kpi["source_model"] = self.model_name
                            kpi["source_image"] = str(image_path_obj.name)
                        
                        result = {
                            "kpis": merged_kpis,
                            "model": self.model_name,
                            "image_path": str(image_path),
                            "page_markdown_used": False,
                            "num_kpis": len(merged_kpis),
                            "inference_time_seconds": round(inference_time, 2),
                            "recovery_method": "continue_and_merge",
                            "continuations_used": merge_result["continuations_used"],
                        }
                        
                        # Run validation if db_path is provided
                        if validate_results and db_path and merged_kpis:
                            logger.info(f"    → Running validation against database...")
                            try:
                                validation_output = validate_kpis(
                                    kpis=merged_kpis,
                                    db_path=db_path,
                                    year=year,
                                    page=page,
                                    bucket=bucket,
                                    table_idx=table_idx,
                                )
                                result["validation_statistics"] = validation_output["statistics"]
                                result["validation_summary"] = {
                                    "total_kpis": validation_output["statistics"]["total_kpis"],
                                    "valid_kpis": validation_output["statistics"]["valid_kpis"],
                                    "invalid_kpis": validation_output["statistics"]["invalid_kpis"],
                                    "accuracy": validation_output["statistics"]["accuracy"],
                                    "confidence_avg": validation_output["statistics"]["confidence_avg"]
                                }
                                result["valid_kpis"] = validation_output["valid_kpis"]
                                result["invalid_kpis"] = validation_output["invalid_kpis"]
                                logger.info(
                                    f"    ✓ Validation complete: "
                                    f"{validation_output['statistics']['valid_kpis']}/{validation_output['statistics']['total_kpis']} valid "
                                    f"(Accuracy: {validation_output['statistics']['accuracy']:.1f}%)"
                                )
                            except Exception as val_e:
                                logger.warning(f"    ⚠ Validation failed: {str(val_e)}")
                                result["validation_error"] = str(val_e)
                        
                        return result
                
                # Step 3: No partial KPIs salvaged - try a full retry from scratch
                logger.info(f"    → No partial KPIs salvaged, attempting full retry...")
                retry_count = 0
                max_retries = 2
                
                while retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"    → Full retry {retry_count}/{max_retries}...")
                    
                    try:
                        retry_start = time.time()
                        recovery_text = self.model_manager.generate_vlm_output(
                            image_path=str(image_path),
                            prompt=user_prompt  # Use original prompt for clean retry
                        )
                        retry_time = time.time() - retry_start
                        inference_time += retry_time
                        
                        recovery_cleaned = clean_json_response(recovery_text)
                        result = json.loads(recovery_cleaned)
                        
                        if "kpis" in result and isinstance(result["kpis"], list):
                            logger.info(f"    ✓ Full retry successful! Extracted {len(result['kpis'])} KPIs")
                            
                            for kpi in result["kpis"]:
                                kpi["source_model"] = self.model_name
                                kpi["source_image"] = str(image_path_obj.name)
                            
                            result["model"] = self.model_name
                            result["image_path"] = str(image_path)
                            result["page_markdown_used"] = False
                            result["num_kpis"] = len(result["kpis"])
                            result["inference_time_seconds"] = round(inference_time, 2)
                            result["recovery_method"] = "full_retry"
                            result["recovery_attempt"] = retry_count
                            
                            return result
                        else:
                            logger.warning(f"    ⚠ Retry {retry_count}: Invalid JSON structure")
                    
                    except json.JSONDecodeError as retry_error:
                        logger.warning(f"    ⚠ Retry {retry_count}: JSON parsing still failed - {str(retry_error)}")
                    except Exception as retry_exception:
                        logger.warning(f"    ⚠ Retry {retry_count}: Recovery failed - {str(retry_exception)}")
                
                # Save raw output to file for debugging
                logger.info(f"    → Saving raw output for debugging...")
                raw_output_file = image_path_obj.parent / f"{image_path_obj.stem}_raw_output.txt"
                try:
                    with open(raw_output_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== RAW MODEL OUTPUT ===\n")
                        f.write(output_text)
                        f.write(f"\n\n=== CLEANED OUTPUT ===\n")
                        f.write(cleaned_text)
                        f.write(f"\n\n=== ERROR ===\n")
                        f.write(str(e))
                    logger.info(f"    → Raw output saved to: {raw_output_file.name}")
                except Exception as save_error:
                    logger.warning(f"    ⚠ Failed to save raw output: {str(save_error)}")
                
                # All recovery methods failed
                return {
                    "kpis": [],
                    "image_path": str(image_path),
                    "model": self.model_name,
                    "inference_time_seconds": round(inference_time, 2),
                    "raw_output_file": str(raw_output_file.name) if 'raw_output_file' in locals() else None,
                    "error": f"JSON decode error after all recovery attempts: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"  ✗ Error processing image: {str(e)}")
            return {
                "kpis": [],
                "image_path": str(image_path),
                "model": self.model_name,
                "error": str(e)
            }
    
    def extract_kpis_from_images(
        self,
        image_paths: List[str],
        context: Optional[str] = None,
        output_dir: Optional[str] = None,
        page_markdown_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        validate_results: bool = True,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract KPIs from multiple table images.
        
        Args:
            image_paths: List of paths to table image files
            context: Optional context about the tables
            output_dir: Optional directory to save individual results
            page_markdown_dir: Optional directory containing page markdown files
            db_path: Optional path to SQLite database for validation
            validate_results: Whether to validate extracted KPIs (requires db_path)
            year: Optional year for validation filtering
            
        Returns:
            Dictionary with aggregated results from all images
        """
        if self.model_manager.current_llm is None:
            if not self.load_model():
                return {
                    "error": "Failed to load model",
                    "results": []
                }
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EXTRACTING KPIs FROM {len(image_paths)} IMAGES")
        logger.info(f"{'=' * 70}\n")
        
        all_results = []
        all_kpis = []
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"\n[{idx}/{len(image_paths)}] Processing: {Path(image_path).name}")

            page_markdown_path = None
            if page_markdown_dir:
                md_dir = Path(page_markdown_dir)
                image_parent = Path(image_path).parent.name
                if image_parent.startswith("page_") and "_table_" in image_parent:
                    page_prefix = image_parent.split("_table_")[0]
                    candidate_md = md_dir / f"{page_prefix}.md"
                    candidate_markdown = md_dir / f"{page_prefix}.markdown"
                    if candidate_md.exists():
                        page_markdown_path = str(candidate_md)
                    elif candidate_markdown.exists():
                        page_markdown_path = str(candidate_markdown)

            result = self.extract_kpis_from_image(
                image_path, 
                context, 
                page_markdown_path,
                db_path=db_path,
                validate_results=validate_results,
                year=year
            )
            all_results.append(result)
            
            if "kpis" in result:

                all_kpis.extend(result["kpis"])
            
            # Save individual result if output directory specified
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                image_name = Path(image_path).stem
                output_file = output_dir_path / f"{image_name}_kpis.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"    → Saved to: {output_file}")
                
                image_name = Path(image_path).stem
                output_file = output_dir_path / f"{image_name}_kpis.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"    → Saved to: {output_file}")
        
        # Compile summary
        summary = {
            "model": self.model_name,
            "total_images": len(image_paths),
            "total_kpis": len(all_kpis),
            "extraction_date": datetime.now().isoformat(),
            "results": all_results,
            "all_kpis": all_kpis
        }
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EXTRACTION COMPLETE")
        logger.info(f"  Total images processed: {len(image_paths)}")
        logger.info(f"  Total KPIs extracted: {len(all_kpis)}")
        logger.info(f"{'=' * 70}\n")
        
        return summary
    
    def extract_kpis_from_tables_json(
        self,
        tables_json_path: str,
        output_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        validate_results: bool = True,
        year: Optional[int] = None,
        bucket: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract KPIs from tables using a tables.json file from detect_tables.py.
        
        Args:
            tables_json_path: Path to tables.json file containing table metadata
            output_dir: Optional directory to save individual results
            db_path: Optional path to SQLite database for validation
            validate_results: Whether to validate extracted KPIs (requires db_path)
            year: Optional year for validation filtering (e.g., 2023)
            
        Returns:
            Dictionary with aggregated results from all tables
        """
        tables_json_path_obj = Path(tables_json_path)
        if not tables_json_path_obj.exists():
            logger.error(f"Tables JSON file not found: {tables_json_path}")
            return {
                "error": "Tables JSON file not found",
                "tables_json_path": str(tables_json_path)
            }
        
        # Load tables.json
        try:
            with open(tables_json_path, 'r', encoding='utf-8') as f:
                tables_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load tables JSON: {str(e)}")
            return {
                "error": f"Failed to load tables JSON: {str(e)}",
                "tables_json_path": str(tables_json_path)
            }
        
        tables = tables_data.get("tables", [])
        if not tables:
            logger.warning("No tables found in JSON file")
            return {
                "model": self.model_name,
                "total_tables": 0,
                "total_kpis": 0,
                "tables_json_path": str(tables_json_path),
                "results": [],
                "all_kpis": []
            }
        
        if self.model_manager.current_llm is None:
            if not self.load_model():
                return {
                    "error": "Failed to load model",
                    "results": []
                }
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EXTRACTING KPIs FROM {len(tables)} TABLES")
        logger.info(f"Source: {tables_json_path}")
        logger.info(f"Context: Volkswagen end-of-year financial report")
        logger.info(f"{'=' * 70}\n")
        
        all_results = []
        all_kpis = []
        
        for idx, table_entry in enumerate(tables, 1):
            image_path = table_entry.get("image_path")
            title = table_entry.get("title", "")
            page_num = table_entry.get("page")
            table_idx = table_entry.get("table_index")
            pdf_filename = table_entry.get("pdf_file", "")
            section_context = table_entry.get("section_context", "Unknown Section")
            
            # Use year from filename extraction if available, otherwise fallback to provided year
            table_year = table_entry.get("year")
            effective_year = table_year if table_year is not None else year
            
            # Use bucket from filename extraction if available, otherwise fallback to provided bucket
            table_bucket = table_entry.get("bucket")
            effective_bucket = table_bucket if table_bucket is not None else bucket
            
            if not image_path:
                logger.warning(f"[{idx}/{len(tables)}] Skipping entry: no image path")
                continue
            
            logger.info(f"\n[{idx}/{len(tables)}] Processing: {pdf_filename} - Page {page_num}, Table {table_idx}")
            logger.info(f"  Image: {Path(image_path).name}")
            logger.info(f"  Section: {section_context}")
            logger.info(f"  Year: {effective_year} (from {'filename' if table_year else 'parameter'})")
            logger.info(f"  Bucket: {effective_bucket} (from {'filename' if table_bucket else 'parameter'})")
            
            # Build context with company-specific information and section context
            context_parts = [
                "Financial report table from Volkswagen Group.",
            ]
            
            # Add bucket/report type context if available (divisions or management)
            if effective_bucket:
                if effective_bucket.lower() == 'divisions':
                    context_parts.append("This is from the DIVISIONS financial report (brand-specific performance metrics).")
                elif effective_bucket.lower() == 'management':
                    context_parts.append("This is from the MANAGEMENT REPORT (consolidated group-level metrics).")
                else:
                    context_parts.append(f"Report type: {effective_bucket}")
            
            # Add section context
            context_parts.append(f"Document section: '{section_context}'")
            context_parts.append(f"This table appears in the '{section_context}' section of the annual report.")
            
            # Add year context if available
            if effective_year:
                context_parts.append(f"Report year: {effective_year}")
                
            context = " ".join(context_parts)
            
            result = self.extract_kpis_from_image(
                image_path,
                context=context,
                title=title,
                db_path=db_path,
                validate_results=validate_results,
                page=page_num,
                year=effective_year,  # Use extracted year from filename
                bucket=effective_bucket,  # Use extracted bucket from filename
                table_idx=table_idx
            )
            
            # Add table metadata to result
            result["page"] = page_num
            result["table_index"] = table_idx
            result["bbox"] = table_entry.get("bbox")
            
            all_results.append(result)
            
            if "kpis" in result:
                all_kpis.extend(result["kpis"])
            
            # Save individual result if output directory specified
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                
                # Include PDF filename to make output files distinct
                pdf_name = Path(pdf_filename).stem if pdf_filename else "unknown"
                output_file = output_dir_path / f"{pdf_name}_page_{page_num:03d}_table_{table_idx:02d}_kpis.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"    → Saved to: {output_file.name}")
        
        # Compile summary
        summary = {
            "model": self.model_name,
            "total_tables": len(tables),
            "total_kpis": len(all_kpis),
            "extraction_date": datetime.now().isoformat(),
            "tables_json_path": str(tables_json_path),
            "context": "Financial report - Volkswagen Group",
            "results": all_results,
            "all_kpis": all_kpis
        }
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EXTRACTION COMPLETE")
        logger.info(f"  Total tables processed: {len(tables)}")
        logger.info(f"  Total KPIs extracted: {len(all_kpis)}")
        logger.info(f"{'=' * 70}\n")
        
        return summary

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for the VLM KPI extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract KPIs from financial table images using Vision-Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from tables.json (recommended)
  python extract_kpis_vlm.py --tables-json detected_tables_test/tables.json --output-dir ./results/
  
  # Extract from a single image
  python extract_kpis_vlm.py --image table_2015.png --output results.json
  
  # Extract from multiple images
  python extract_kpis_vlm.py --images table_*.png --output-dir ./results/
        """
    )
    
    parser.add_argument(
        '--tables-json',
        type=str,
        help='Path to tables.json file from detect_tables.py (recommended method)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single table image file'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Paths to multiple table image files (supports glob patterns)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for single image extraction'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for multiple image extraction'
    )
    
    parser.add_argument(
        '--context',
        type=str,
        help='Additional context about the tables (e.g., "Annual Report 2015")'
    )

    parser.add_argument(
        '--page-markdown',
        type=str,
        help='Path to markdown file for the full page (used as additional context)'
    )

    parser.add_argument(
        '--page-markdown-dir',
        type=str,
        help='Directory containing page markdown files (matched by page_###.md)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default=VLM_MODEL_NAME,
        help=f'Model name to use (default: {VLM_MODEL_NAME})'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (0.0 = deterministic, default: 0.1)'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to SQLite database for KPI validation'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='Year for validation filtering (e.g., 2023)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable automatic validation of extracted KPIs'
    )

    parser.add_argument(
        '--bucket',
        type=str,
        help='bucket for validation filtering'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.tables_json and not args.image and not args.images:
        parser.error("Must specify either --tables-json, --image, or --images")
    
    if sum([bool(args.tables_json), bool(args.image), bool(args.images)]) > 1:
        parser.error("Can only specify one of: --tables-json, --image, or --images")
    
    # Initialize extractor
    extractor = VLMKPIExtractor(
        temperature=args.temperature,
        model_name=args.model_name
    )
    
    try:
        # Process from tables.json (recommended)
        if args.tables_json:
            summary = extractor.extract_kpis_from_tables_json(
                args.tables_json,
                output_dir=args.output_dir,
                db_path=args.db_path,
                validate_results=not args.no_validation,
                year=args.year,
                bucket=args.bucket
            )
            
            # Save summary if output directory specified
            if args.output_dir:
                output_dir_path = Path(args.output_dir)
                summary_file = output_dir_path / "extraction_summary.json"
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                logger.info(f"\n✓ Summary saved to: {summary_file}")
            else:
                # Print summary to stdout
                print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # Process single image
        elif args.image:
            result = extractor.extract_kpis_from_image(
                args.image,
                args.context,
                page_markdown_path=args.page_markdown,
                db_path=args.db_path,
                validate_results=not args.no_validation,
                year=args.year
            )
            
            # Save results
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"\n✓ Results saved to: {output_path}")
            else:
                # Print to stdout
                print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Process multiple images
        else:
            summary = extractor.extract_kpis_from_images(
                args.images,
                context=args.context,
                output_dir=args.output_dir,
                page_markdown_dir=args.page_markdown_dir,
                db_path=args.db_path,
                validate_results=not args.no_validation,
                year=args.year
            )
            
            # Save summary if output directory specified
            if args.output_dir:
                output_dir_path = Path(args.output_dir)
                summary_file = output_dir_path / "extraction_summary.json"
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                logger.info(f"\n✓ Summary saved to: {summary_file}")
            else:
                # Print summary to stdout
                print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    finally:
        # Clean up
        extractor.unload_model()
        logger.info("\n✓ Extraction complete")

if __name__ == "__main__":
    main()
