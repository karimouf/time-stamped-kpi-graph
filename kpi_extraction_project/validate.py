#!/usr/bin/env python3
"""
KPI Extraction Validation (Index-Based)
========================================

Validates extracted KPIs from Seventh-Trial using row_idx and col_idx directly.
No string matching needed - uses indices provided by LLM.

Author: Karim Ouf
Date: November 2025
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from loader import load_tables_from_db_with_filters


def parse_numeric_value(text: str) -> Optional[float]:
    """
    Parse numeric value from table cell text.
    Handles: commas, parentheses (negatives), currency, footnotes, null indicators.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Null indicators
    if text in ['–', '—', '', 'x', 'X', 'n/a', 'N/A']:
        return None
    
    # Handle special minus signs (Unicode minus, en-dash, em-dash)
    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    
    # Detect European decimal format (comma as decimal separator)
    # Pattern: digits, optional comma, digits (no period as thousands separator)
    # Examples: "2,0524" → 2.0524, "1.234,56" → 1234.56, "−1,4864" → -1.4864
    # Check if comma appears after digits and is followed by 1-4 digits (decimal part)
    # Also handle cases with both period and comma: period=thousands, comma=decimal
    european_decimal_pattern = r'^-?[\d\.\s]*,\d{1,4}$'
    if re.match(european_decimal_pattern, text.replace(' ', '')):
        # European format: comma is decimal separator, period/space is thousands separator
        text = text.replace('.', '').replace(' ', '').replace(',', '.')
    else:
        # US format: comma is thousands separator, period is decimal separator
        text = text.replace(',', '').replace(' ', '')
    
    # Parentheses = negative
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Remove currency and footnotes
    text = re.sub(r'[€$£¥]', '', text)
    text = re.sub(r'\^[\d]+\.?\d*', '', text)  # Updated to handle ^7.0, ^4.0 etc
    
    # Abbreviations (K, M, B, T)
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    for suffix, mult in multipliers.items():
        if text.upper().endswith(suffix):
            try:
                return float(text[:-1]) * mult
            except ValueError:
                pass
    
    # Convert to float
    try:
        return float(text)
    except ValueError:
        return None


def _year_row_offset(rows: list) -> int:
    """Return 1 if rows[0] appears to be a year-sub-header row (all non-empty cells
    are 4-digit years in the range 2000-2040), otherwise return 0.

    The DB sometimes stores the column-year labels as the very first row of `rows[]`
    (e.g. ["", "2015", "2014", "2013", ...]).  The VLM never counts that row because
    it sees it as part of the column headers, so its row_idx values are offset by 1
    relative to the raw DB rows.
    """
    if not rows:
        return 0
    first_row = rows[0]
    non_empty = [str(cell).strip() for cell in first_row if str(cell).strip()]
    if not non_empty:
        return 0
    # Normalise float-formatted years like "2014.0" → "2014" before matching
    def _norm(v: str) -> str:
        if v.endswith(".0") and v[:-2].isdigit():
            return v[:-2]
        return v
    normalised = [_norm(v) for v in non_empty]
    year_like = sum(1 for v in normalised if re.fullmatch(r'20[0-3]\d', v))
    # Consider it a year row if at least half the non-empty cells are years
    if year_like / len(normalised) >= 0.5:
        return 1
    return 0


def validate_kpi_indexed(
    kpi: Dict[str, Any],
    table_data: Dict[str, Any],
    seen_nodes: Optional[set] = None
) -> Dict[str, Any]:
    """
    Validate one KPI using row_idx and col_idx directly.
    
    PRIMARY: Uses row_idx and col_idx directly (no string matching)
    Verifies that rows[row_idx][col_idx] matches the extracted value.
    """
    result = {
        "is_valid": True,
        "confidence": 1.0,
        "errors": [],
        "fix_instructions": [],  # Clear instructions for fixing this KPI
        "source_cell_value": None,
        "source_cell_text": None,
        "extracted_value": kpi.get("value"),
        "row_idx": None,
        "col_idx": None,
        "row_name_match": None,
        "col_name_match": None
    }
    # Check required fields are present
    required_fields = ["name", "key", "units", "value", "year", "row_idx", "col_idx"]
    missing_fields = [field for field in required_fields if field not in kpi]
    
    if missing_fields:
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append(f"Missing required fields: {', '.join(missing_fields)}")
        return result
    
    # Extract fields
    row_idx = kpi.get("row_idx")
    col_idx = kpi.get("col_idx")
    extracted_value = kpi.get("value")
    kpi_name = kpi.get("name", "")
    kpi_key = kpi.get("key", "")
    kpi_year = kpi.get("year", "")
    
    # Check that name and key are not empty (required for grouping/linking)
    kpi_name_stripped = kpi_name.strip() if kpi_name else ""
    kpi_key_stripped = kpi_key.strip() if kpi_key else ""
    
    # Check for duplicate KPIs using seen_nodes set
    if seen_nodes is not None:
        kpi_units = str(kpi.get("units", "")).strip()
        kpi_identifier = (kpi_name_stripped, kpi_key_stripped, kpi_units, kpi_year)
        
        if kpi_identifier in seen_nodes:
            # Duplicate KPI detected
            result["is_valid"] = False
            result["confidence"] = 0.0
            result["errors"].append(f"DUPLICATE KPI: name='{kpi_name}', key='{kpi_key}', units='{kpi_units}', year={kpi_year}")
            result["errors"].append("This exact KPI (same name, key, units, and year) has already been extracted")
            result["is_duplicate"] = True
            
            return result
        else:
            # Add to seen_nodes to track this KPI
            seen_nodes.add(kpi_identifier)
    
    if not kpi_name_stripped:
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append("Empty 'name' field - cannot be grouped/linked")
        result["fix_instructions"].append("FIX: 'name' is empty or whitespace-only")
        result["fix_instructions"].append("SOLUTION: Extract the KPI metric name from merged_headers, title, or stub_col")
        return result
    
    if not kpi_key_stripped:
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append("Empty 'key' field - cannot be grouped/linked")
        result["fix_instructions"].append("FIX: 'key' is empty or whitespace-only")
        result["fix_instructions"].append("SOLUTION: Extract the entity/segment from stub_col, section_name, or context")
        return result
    
    # Check that key != name when both are non-empty
    if kpi_name_stripped and kpi_key_stripped:
        # Normalize for comparison
        name_norm = str(kpi_name).strip()
        key_norm = str(kpi_key).strip()
        
        if name_norm == key_norm:
            result["is_valid"] = False
            result["confidence"] = 0.5
            result["errors"].append(f"❌ key and name are identical: '{kpi_key}' (they must be different when both are non-empty)")
            
            # Add clear fix instructions
            table_title = table_data.get('title', '')
            section_name = table_data.get('section_name', '')
            result["fix_instructions"].append(f"FIX: key='{kpi_key}' and name='{kpi_name}' are identical. Look at table context:")
            result["fix_instructions"].append(f"  - Table title: '{table_title}'")
            result["fix_instructions"].append(f"  - Section: '{section_name}'")
            result["fix_instructions"].append(f"  - Current row: '{kpi.get('row_name', '')}'")
            result["fix_instructions"].append(f"SOLUTION: Determine which is the metric (name) vs entity (key):")
            result["fix_instructions"].append(f"  - If table measures production/sales of vehicles: name='Production' or 'Sales', key='{kpi_key}'")
            result["fix_instructions"].append(f"  - If table shows KPIs for a company/brand: name='{kpi_name}', key='Company'/'Brand'")
            result["fix_instructions"].append(f"  - Check title and section to determine the correct interpretation")
            return result
    
    # Check required fields
    if row_idx is None:
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append("Missing row_idx in KPI")
        return result
    
    if col_idx is None:
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append("Missing col_idx in KPI")
        return result
    
    # Get table structure
    rows = table_data.get('rows', [])
    stub_col = table_data.get('stub_col', [])
    merged_headers = table_data.get('merged_headers', [])
    
    if not rows:
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append("No rows in table data")
        return result

    # Detect if the DB stored a year-sub-header as rows[0] and offset accordingly
    row_offset = _year_row_offset(rows)
    db_row_idx = row_idx + row_offset  # adjusted index into the raw DB rows array

    result["row_idx"] = row_idx
    result["col_idx"] = col_idx

    # Validate row index bounds
    if db_row_idx < 0 or db_row_idx >= len(rows):
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append(f"row_idx {row_idx} out of bounds (table has {len(rows)} rows, offset={row_offset})")
        return result

    # Validate adjusted column index bounds
    if col_idx < 0 or col_idx >= len(rows[db_row_idx]):
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append(f"col_idx {col_idx} out of bounds (row has {len(rows[db_row_idx])} cols)")
        return result

    # Cross-validate with stub_col[db_row_idx] for reference (informational only)
    if stub_col and db_row_idx < len(stub_col):
        expected_row_name = stub_col[db_row_idx]
        result["row_name_match"] = expected_row_name

    # Cross-validate with merged_headers[col_idx] for reference (informational only)
    if merged_headers and col_idx < len(merged_headers):
        expected_col_name = merged_headers[col_idx]
        result["col_name_match"] = expected_col_name

    # Extract cell value using adjusted indices
    cell_text = rows[db_row_idx][col_idx]
    result["source_cell_text"] = cell_text
    
    # Parse numeric value
    source_value = parse_numeric_value(str(cell_text))
    result["source_cell_value"] = source_value
    
    # Compare extracted vs source values
    if extracted_value is None and source_value is None:
        # Both null - valid but lower confidence
        result["confidence"] *= 0.95
        result["errors"].append(f"Both null (cell: '{cell_text}')")
    elif extracted_value is None:
        # Extracted null but source has value
        result["is_valid"] = False
        result["confidence"] = 0.2
        result["errors"].append(f"Extracted null but source={source_value} (cell: '{cell_text}')")
    elif source_value is None:
        # Source null but extracted has value
        result["is_valid"] = False
        result["confidence"] = 0.2
        result["errors"].append(f"Source null but extracted={extracted_value} (cell: '{cell_text}')")
    else:
        # Both should be numeric - ensure types are correct
        try:
            extracted_numeric = float(extracted_value) if not isinstance(extracted_value, (int, float)) else extracted_value
            source_numeric = float(source_value) if not isinstance(source_value, (int, float)) else source_value
        except (ValueError, TypeError) as e:
            result["is_valid"] = False
            result["confidence"] = 0.0
            result["errors"].append(f"Type conversion error: extracted={extracted_value} (type={type(extracted_value).__name__}), source={source_value} (type={type(source_value).__name__})")
            return result
        
        # Compare with tolerance - only accept exact matches
        diff = abs(source_numeric - extracted_numeric)
        
        if diff <= 1e-6:
            # Perfect match - valid
            pass  # confidence already 1.0
        else:
            # Any difference is invalid
            result["is_valid"] = False
            result["confidence"] = 0.0
            result["errors"].append(f"Value mismatch: extracted={extracted_numeric}, source={source_numeric}, diff={diff:.6f}")
            
            # Add fix instructions
            result["fix_instructions"].append(f"FIX: Value mismatch")
            result["fix_instructions"].append(f"  - Extracted: {extracted_numeric}")
            result["fix_instructions"].append(f"  - Source at db_row={db_row_idx} (row_idx={row_idx} + offset={row_offset}), col={col_idx}: {source_numeric} (text: '{cell_text}')")
            result["fix_instructions"].append(f"SOLUTION: Verify the correct row_idx and col_idx for this value")
    
    return result


def validate_extraction_file(extraction_file: Path, tables_dir: Path) -> Dict[str, Any]:
    """
    Validate one extraction file using index-based lookup.
    
    Steps:
    1. Extract year from filename
    2. Load corresponding tables file
    3. Loop through all tables in extraction file
    4. For each table, use table_id to fetch source table
    5. Loop through all KPIs
    6. For each KPI, use row_idx and col_idx DIRECTLY (no string matching)
    7. Access cell value: rows[row_idx][col_idx]
    8. Compare extracted value vs source value
    """
    # Step 1: Extract year from filename
    match = re.search(r'linked_tables\((\d{4})\)', extraction_file.name)
    if not match:
        print(f"⚠️  Could not extract year from {extraction_file.name}")
        return None
    
    year = match.group(1)
    tables_file = tables_dir / f'linked_tables({year}).jsonl'
    
    if not tables_file.exists():
        print(f"❌ Tables file not found: {tables_file.name}")
        return None
    
    # Step 2: Load extraction data
    with open(extraction_file, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    
    # Step 3: Load source tables into dictionary keyed by table_id
    tables = {}
    with open(tables_file, 'r', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line)
            table_id = table.get('table_id')
            if table_id:
                tables[table_id] = table
    
    print(f"   Loaded {len(tables)} source tables from {tables_file.name}")
    
    # Validation results
    invalid_kpis = []
    valid_kpis = []
    seen_nodes = set()  # Track seen (name, key, year) combinations
    stats = {
        "total_kpis": 0,
        "valid_kpis": 0,
        "invalid_kpis": 0,
        "tables_processed": 0,
        "row_name_mismatches": 0,
        "col_name_mismatches": 0,
        "row_name_verified": 0,
        "col_name_verified": 0,
        "name_mismatches": 0,  # Total (row + col)
        "duplicate_kpis": 0  # Count of duplicate KPIs
    }
    
    # Step 4: Loop through all tables in extraction file
    for table_result in extraction_data.get('tables', []):
        table_id = table_result.get('table_id')
        kpis = table_result.get('extraction_result', {}).get('kpis', [])
        
        if not table_id or not kpis:
            continue
        
        # Use table_id to fetch source table
        source_table = tables.get(table_id)
        if not source_table:
            print(f"   ⚠️  Source table {table_id} not found in tables file")
            continue
        
        stats["tables_processed"] += 1
        
        # Step 5: Loop through all KPIs in this table
        for kpi in kpis:
            stats["total_kpis"] += 1
            
            # Step 6-8: Validate using indices directly, passing seen_nodes to detect duplicates
            validation = validate_kpi_indexed(kpi, source_table, seen_nodes)
            
            # Count name verification results
            for err in validation["errors"]:
                if "DUPLICATE KPI:" in err:
                    stats["duplicate_kpis"] += 1
                
                if "✓ row_name verified" in err:
                    stats["row_name_verified"] += 1
                elif "✗ row_name MISMATCH" in err or "row_name mismatch" in err:
                    stats["row_name_mismatches"] += 1
                    stats["name_mismatches"] += 1
                
                if "✓ col_name verified" in err:
                    stats["col_name_verified"] += 1
                elif "✗ col_name MISMATCH" in err or "col_name mismatch" in err:
                    stats["col_name_mismatches"] += 1
                    stats["name_mismatches"] += 1
            
            if validation["is_valid"]:
                stats["valid_kpis"] += 1
                # Save valid KPI with context
                valid_kpis.append({
                    "name": kpi.get("name"),
                    "key": kpi.get("key"),
                    "units": kpi.get("units"),
                    "value": kpi.get("value"),
                    "year": kpi.get("year"),
                    "evidence": {
                        "table_id": table_id,
                        "row_idx": kpi.get("row_idx"),
                        "col_idx": kpi.get("col_idx"),
                        "row_name": kpi.get("row_name"),
                        "col_name": kpi.get("col_name")
                    },           
                })
            else:
                stats["invalid_kpis"] += 1
                # Save invalid KPI with context
                invalid_kpis.append({
                    "table_id": table_id,
                    "kpi": {
                        "name": kpi.get("name"),
                        "key": kpi.get("key"),
                        "units": kpi.get("units"),
                        "year": kpi.get("year"),
                        "row_idx": kpi.get("row_idx"),
                        "col_idx": kpi.get("col_idx"),
                        "row_name": kpi.get("row_name"),
                        "col_name": kpi.get("col_name")
                    },
                    "validation": validation
                })
    
    # Calculate accuracy
    if stats["total_kpis"] > 0:
        stats["accuracy"] = (stats["valid_kpis"] / stats["total_kpis"]) * 100
    else:
        stats["accuracy"] = 0.0
    
    return {
        "file": extraction_file.name,
        "year": year,
        "stats": stats,
        "invalid_kpis": invalid_kpis,
        "valid_kpis": valid_kpis
    }

def _validate_order_fallback(
    kpis: List[Dict[str, Any]],
    table_data: Dict[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """
    Order-based fallback validation.

    Flattens DB rows into a reading-order sequence (row-by-row, left-to-right)
    of numeric cells, then checks that the extracted KPI values — sorted by
    (row_idx, col_idx) — appear as a forward subsequence in that sequence.

    Returns a dict mapping each KPI's original list index to:
        {"order_valid": bool, "matched_at": (row_i, col_j) | None, "reason": str}
    """
    rows = table_data.get("rows", [])

    # Build flat reading-order list of numeric cells
    flat = []  # list of (row_i, col_j, numeric_val)
    for row_i, row in enumerate(rows):
        for col_j, cell in enumerate(row):
            val = parse_numeric_value(str(cell) if cell is not None else "")
            if val is not None:
                flat.append((row_i, col_j, val))

    # Sort KPIs by (row_idx, col_idx), keeping original index
    sorted_kpis = sorted(
        enumerate(kpis),
        key=lambda x: (x[1].get("row_idx") or 0, x[1].get("col_idx") or 0)
    )

    results: Dict[int, Dict[str, Any]] = {}
    pointer = 0  # forward cursor in flat list

    for orig_idx, kpi in sorted_kpis:
        extracted_val = parse_numeric_value(
            str(kpi.get("value", "")) if kpi.get("value") is not None else ""
        )

        if extracted_val is None:
            results[orig_idx] = {
                "order_valid": False,
                "matched_at": None,
                "reason": "non-numeric extracted value"
            }
            continue

        # Search forward from pointer
        matched = False
        for pos in range(pointer, len(flat)):
            row_i, col_j, db_val = flat[pos]
            if abs(db_val - extracted_val) <= 1e-6:
                results[orig_idx] = {
                    "order_valid": True,
                    "matched_at": (row_i, col_j),
                    "reason": f"matched DB cell at row={row_i}, col={col_j}"
                }
                pointer = pos + 1
                matched = True
                break

        if not matched:
            results[orig_idx] = {
                "order_valid": False,
                "matched_at": None,
                "reason": "value not found in remaining DB reading-order cells"
            }

    return results


def validate_kpis(
    kpis: List[Dict[str, Any]],
    db_path: str,
    table_idx: int,
    year: Optional[int] = None,
    page: Optional[int] = None,
    bucket: Optional[str] = None,
    max_tables: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate a list of KPIs against the provided table data using index-based lookup.
    
    Args:
        kpis: List of KPI dictionaries to validate
        db_path: Path to the SQLite database containing tables
        table_idx: Table index for direct table access (0-based, required)
        year: Optional year filter (e.g., 2023)
        page: Optional page filter (e.g., 3)
        bucket: Optional bucket filter (e.g., "financial_reports")
        max_tables: Optional maximum number of tables to load
        
    Returns:
        Dictionary containing validation results and statistics:
        {
            "validation_results": [...],
            "statistics": {
                "total_kpis": int,
                "valid_kpis": int,
                "invalid_kpis": int,
                "accuracy": float,
                "precision": float,
                "confidence_avg": float,
                "duplicate_kpis": int,
                "missing_tables": int
            },
            "valid_kpis": [...],
            "invalid_kpis": [...]
        }
        
    Note:
        All KPIs must belong to the same table specified by table_idx.
        The table is accessed once and reused for all KPI validations.
        This approach is efficient when validating multiple KPIs from the same table.
    """
    results = []
    seen_nodes = set()  # Track seen (name, key, units, year) combinations to detect duplicates
    duplicate_kpis_list: list = []  # Collect each duplicate KPI for the retry prompt
    
    # Statistics tracking
    stats = {
        "total_kpis": 0,
        "valid_kpis": 0,
        "invalid_kpis": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "confidence_avg": 0.0,
        "confidence_sum": 0.0,
        "duplicate_kpis": 0,
        "missing_tables": 0,
        "row_name_verified": 0,
        "col_name_verified": 0,
        "row_name_mismatches": 0,
        "col_name_mismatches": 0,
        "valid_by_primary": 0,
        "valid_by_order_fallback": 0,
        "invalid_both_failed": 0
    }
    
    valid_kpis = []
    invalid_kpis = []
    
    # Build filters dictionary
    filters = {}
    if year is not None:
        filters['year'] = year
    if page is not None:
        filters['page'] = page
    if bucket is not None:
        normalized_bucket = bucket.strip().lower()
        if normalized_bucket == 'management':
            normalized_bucket = 'management_report'
        filters['bucket'] = normalized_bucket
    
    # Load tables from database with filters
    tables = load_tables_from_db_with_filters(
        db_path=db_path,
        filters=filters if filters else None,
        max_tables=max_tables
    )
    
    # Create a lookup dictionary by table_id
    tables_by_id = {table['table_id']: table for table in tables}
    
    # AUTO-CORRECTION: Detect and fix incorrect index starting points
    # Check first few KPIs to detect if indices start at wrong values
    if kpis:
        # Find minimum indices across first 5 KPIs (or all if less than 5)
        sample_size = min(5, len(kpis))
        min_row_idx = min((kpi.get('row_idx', 0) for kpi in kpis[:sample_size] if kpi.get('row_idx') is not None), default=0)
        min_col_idx = min((kpi.get('col_idx', 1) for kpi in kpis[:sample_size] if kpi.get('col_idx') is not None), default=1)
        row_offset = 0
        col_offset = 0
        
        # If row_idx starts at 1, we need to subtract 1 from all row indices (should start at 0)
        if min_row_idx == 1:
            row_offset = -1
            print(f"⚠️  Auto-correction: row_idx starts at 1, adjusting to start at 0 (subtracting 1)")
        
        # If col_idx starts at 0, we need to add 1 to all col indices (should start at 1)
        if min_col_idx == 0:
            col_offset = 1
            print(f"⚠️  Auto-correction: col_idx starts at 0, adjusting to start at 1 (adding 1)")
        
        # Apply corrections if needed
        if row_offset != 0 or col_offset != 0:
            for kpi in kpis:
                if row_offset != 0 and kpi.get('row_idx') is not None:
                    kpi['row_idx'] = kpi['row_idx'] + row_offset
                if col_offset != 0 and kpi.get('col_idx') is not None:
                    kpi['col_idx'] = kpi['col_idx'] + col_offset
    # Validate each KPI using provided table_idx parameter
    # Verify table_idx is in bounds
    if not (0 <= table_idx < len(tables)):
        # All KPIs are invalid if table_idx is out of bounds
        for kpi in kpis:
            stats["total_kpis"] += 1
            stats["missing_tables"] += 1
            stats["invalid_kpis"] += 1
            validation_result = {
                "is_valid": False,
                "confidence": 0.0,
                "errors": [f"table_idx {table_idx} out of bounds (loaded {len(tables)} tables matching filters)"]
            }
            results.append({
                "kpi": kpi,
                "validation": validation_result
            })
            invalid_kpis.append({
                "kpi": kpi,
                "validation": validation_result
            })
        return {
            "validation_results": results,
            "statistics": stats,
            "valid_kpis": valid_kpis,
            "invalid_kpis": invalid_kpis
        }
    
    # Get the table once using table_idx
    table_data = tables[table_idx]

    # ------------------------------------------------------------------
    # Phase 1: Primary index-based validation (row_idx / col_idx exact)
    # ------------------------------------------------------------------
    primary_results = []
    for kpi in kpis:
        stats["total_kpis"] += 1
        validation_result = validate_kpi_indexed(kpi, table_data, seen_nodes)
        validation_result["used_table_index"] = table_idx
        validation_result["selected_table_id"] = table_data.get("table_id")
        primary_results.append((kpi, validation_result))

    # ------------------------------------------------------------------
    # Phase 2: Order-based fallback (reading-order subsequence in rows)
    # ------------------------------------------------------------------
    order_results = _validate_order_fallback(kpis, table_data)

    # ------------------------------------------------------------------
    # Phase 3: Combine — take the higher result of both checks
    # ------------------------------------------------------------------
    for i, (kpi, validation_result) in enumerate(primary_results):
        primary_valid = validation_result.get("is_valid", False)
        order = order_results.get(i, {"order_valid": False, "matched_at": None, "reason": "not checked"})
        order_valid = order.get("order_valid", False)

        if primary_valid:
            validation_result["validation_method"] = "primary"
        elif order_valid:
            # Fallback upgrades this KPI to valid
            validation_result["is_valid"] = True
            validation_result["confidence"] = 0.8
            validation_result["validation_method"] = "order_fallback"
            validation_result["order_match"] = order.get("matched_at")
            validation_result["errors"] = [
                f"[PRIMARY FAILED] {e}" for e in validation_result.get("errors", [])
            ]
            validation_result["errors"].append(
                f"[ORDER FALLBACK PASSED] value matched DB cell at {order.get('matched_at')}"
            )
        else:
            validation_result["validation_method"] = "both_failed"
            validation_result["errors"].append(
                f"[ORDER FALLBACK FAILED] {order.get('reason', 'value not found in reading-order sequence')}"
            )

        # Update statistics
        stats["confidence_sum"] += validation_result.get("confidence", 0.0)

        if validation_result.get("is_valid", False):
            stats["valid_kpis"] += 1
            valid_kpis.append({"kpi": kpi, "validation": validation_result})
        else:
            stats["invalid_kpis"] += 1
            invalid_kpis.append({"kpi": kpi, "validation": validation_result})

        # Track row/col name validation
        if validation_result.get("row_name_match") is not None:
            if "row_name mismatch" not in str(validation_result.get("errors", [])):
                stats["row_name_verified"] += 1
            else:
                stats["row_name_mismatches"] += 1

        if validation_result.get("col_name_match") is not None:
            if "col_name mismatch" not in str(validation_result.get("errors", [])):
                stats["col_name_verified"] += 1
            else:
                stats["col_name_mismatches"] += 1

        # Track duplicates
        if validation_result.get("is_duplicate", False):
            stats["duplicate_kpis"] += 1
            duplicate_kpis_list.append(kpi)

        # Track which method validated the KPI
        method = validation_result.get("validation_method", "")
        if method == "primary":
            stats["valid_by_primary"] += 1
        elif method == "order_fallback":
            stats["valid_by_order_fallback"] += 1
        else:
            stats["invalid_both_failed"] += 1

        results.append({"kpi": kpi, "validation": validation_result})

    # Calculate final metrics
    if stats["total_kpis"] > 0:
        stats["accuracy"] = (stats["valid_kpis"] / stats["total_kpis"]) * 100
        stats["confidence_avg"] = stats["confidence_sum"] / stats["total_kpis"]
        stats["precision"] = stats["accuracy"]
    
    # Remove intermediate calculation field
    del stats["confidence_sum"]
    
    return {
        "validation_results": results,
        "statistics": stats,
        "valid_kpis": valid_kpis,
        "invalid_kpis": invalid_kpis,
        "duplicate_kpis_list": duplicate_kpis_list,
        "has_duplicates": len(duplicate_kpis_list) > 0,
    }


def validate_trial27_with_annotations(
    annotation_file: Path,
    factgenie_outputs_dir: Path,
    extraction_dir: Path,
    output_dir: Path,
) -> None:
    """
    Validate trial-27 extraction results using human annotation as ground truth.

    For each annotated table split in the annotation JSONL:
      1. Map split → factgenie output JSONL → metadata.source_file
      2. Load the corresponding KPI extraction file from extraction_dir
      3. Apply annotation row_checks as labels (True=valid, False=invalid)
      4. Write per-file annotated validation results to output_dir
      5. Print aggregate summary

    Args:
        annotation_file:        Path to the annotator JSONL (one line per table).
        factgenie_outputs_dir:  Directory containing {split}-manual-kpi.jsonl files.
        extraction_dir:         Directory containing the *_kpis.json extraction files.
        output_dir:             Directory to write per-file validation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("ANNOTATION-BASED KPI VALIDATION  —  trial-27 / vlm_qwen_72b")
    print("=" * 70)
    print(f"Annotation : {annotation_file}")
    print(f"Factgenie  : {factgenie_outputs_dir}")
    print(f"Extraction : {extraction_dir}")
    print(f"Output     : {output_dir}")
    print("=" * 70)

    total_kpis = 0
    total_valid = 0
    total_invalid = 0
    files_processed = 0
    skipped = 0

    with open(annotation_file, 'r', encoding='utf-8', errors='replace') as af:
        for line in af:
            line = line.strip()
            if not line:
                continue
            annotation = json.loads(line)

            split = annotation.get("split")          # e.g. "table-017"
            row_checks = annotation.get("row_checks", [])  # [{rowIndex: N, value: bool}]
            annotator_id = annotation.get("annotator_id", "unknown")

            # Collect free-text notes from the annotator
            annotator_notes = ""
            for tf in annotation.get("text_fields", []):
                if tf.get("label") == "Remaining kpis":
                    annotator_notes = tf.get("value", "")

            # Step 1: Map split → factgenie output JSONL → source_file name
            factgenie_output_file = factgenie_outputs_dir / f"{split}-manual-kpi.jsonl"
            if not factgenie_output_file.exists():
                print(f"  ⚠  Factgenie output not found: {factgenie_output_file.name}")
                skipped += 1
                continue

            with open(factgenie_output_file, 'r', encoding='utf-8') as ff:
                factgenie_entry = json.loads(ff.readline())
            source_file_name = factgenie_entry.get("metadata", {}).get("source_file")
            if not source_file_name:
                print(f"  ⚠  No source_file in factgenie output for {split}")
                skipped += 1
                continue

            # Step 2: Load the source KPI extraction file
            extraction_file = extraction_dir / source_file_name
            if not extraction_file.exists():
                print(f"  ⚠  Extraction file not found: {source_file_name}")
                skipped += 1
                continue

            with open(extraction_file, 'r', encoding='utf-8') as ef:
                extraction_data = json.load(ef)

            kpis = extraction_data.get("kpis", [])
            if not kpis:
                print(f"  ⚠  No KPIs in extraction file: {source_file_name}")
                skipped += 1
                continue

            # Step 3: Build row-level validity map from annotation row_checks
            # row_checks format: [{"rowIndex": N, "value": bool}, ...]
            # True = annotator confirmed valid, False = annotator marked invalid
            row_validity: Dict[int, bool] = {}
            for check in row_checks:
                row_idx = check.get("rowIndex")
                is_valid = check.get("value", True)
                if row_idx is not None:
                    row_validity[row_idx] = is_valid

            # Step 4: Apply annotation labels to KPIs
            valid_kpis = []
            invalid_kpis = []
            for kpi_idx, kpi in enumerate(kpis):
                # Default to valid if the annotator did not explicitly review this row
                is_valid = row_validity.get(kpi_idx, True)
                entry = {
                    "kpi": kpi,
                    "validation": {
                        "is_valid": is_valid,
                        "confidence": 1.0,
                        "method": "human_annotation",
                        "annotator": annotator_id,
                        "row_index": kpi_idx,
                    },
                }
                if is_valid:
                    valid_kpis.append(entry)
                else:
                    entry["validation"]["errors"] = ["Marked as invalid by human annotator"]
                    if annotator_notes:
                        entry["validation"]["annotator_notes"] = annotator_notes
                    invalid_kpis.append(entry)

            n_total = len(kpis)
            n_valid = len(valid_kpis)
            n_invalid = len(invalid_kpis)
            accuracy = (n_valid / n_total * 100) if n_total > 0 else 0.0

            stats = {
                "total_kpis": n_total,
                "valid_kpis": n_valid,
                "invalid_kpis": n_invalid,
                "accuracy": accuracy,
                "annotated_rows": len(row_checks),
                "validation_method": "human_annotation",
            }

            print(
                f"  {source_file_name}  ({split})  ->  "
                f"{n_valid}/{n_total} valid  (acc {accuracy:.1f}%)"
            )

            # Step 5: Write per-file result
            out_file = output_dir / source_file_name
            out_data = {
                "source_file": source_file_name,
                "split": split,
                "annotation_file": annotation_file.name,
                "annotator_id": annotator_id,
                "annotator_notes": annotator_notes,
                "validation_method": "human_annotation",
                "statistics": stats,
                "valid_kpis": valid_kpis,
                "invalid_kpis": invalid_kpis,
            }
            with open(out_file, 'w', encoding='utf-8') as of:
                json.dump(out_data, of, indent=2, ensure_ascii=False)

            total_kpis += n_total
            total_valid += n_valid
            total_invalid += n_invalid
            files_processed += 1

    # Summary
    overall_acc = (total_valid / total_kpis * 100) if total_kpis > 0 else 0.0

    print("\n" + "=" * 70)
    print("ANNOTATION VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Tables processed         : {files_processed}  (skipped: {skipped})")
    print(f"  Total KPIs               : {total_kpis}")
    print(f"  Valid KPIs  (annotated)  : {total_valid}")
    print(f"  Invalid KPIs (annotated) : {total_invalid}")
    print(f"  Accuracy                 : {overall_acc:.2f}%")
    print("=" * 70)
    print(f"\n  Per-file results saved to: {output_dir}")


def validate_against_ground_truth(
    kpis: List[Dict[str, Any]],
    source_file: str,
    ground_truth_file: Path,
    value_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Validate extracted KPIs against a ground-truth JSONL file.

    Pre-filter: drop any extracted KPI where name, key, value, year, units,
    row_idx, or col_idx is None — these are tracked as 'filtered'.

    For each remaining extracted KPI:
      Step 1 — anchor by (row_idx, col_idx):
        - No GT KPI at that position → hallucinated
      Step 2 — compare name, key, value, year, units (all must match):
        - All match → valid
        - Any mismatch → invalid (with mismatched_fields list)

    For each GT KPI:
      - Not covered by any valid extracted KPI → missed

    Args:
        kpis:               The extracted KPI list to evaluate.
        source_file:        Filename key used to look up the record in the
                            ground-truth JSONL.
        ground_truth_file:  Path to the ground-truth JSONL file.
        value_tolerance:    Absolute tolerance for numeric value comparison.

    Returns:
        {
            "source_file":       str,
            "ground_truth_file": str,
            "gt_found":          bool,
            "statistics": {
                "total_extracted": int,
                "filtered":        int,
                "total_gt":        int,
                "valid":           int,
                "invalid":         int,
                "hallucinated":    int,
                "missed":          int,
                "valid_rate":      float,  # valid / (valid + invalid + hallucinated)
                "coverage":        float,  # valid / total_gt
            },
            "valid":        [ {extracted_kpi, gt_kpi} ],
            "invalid":      [ {extracted_kpi, gt_kpi, mismatched_fields} ],
            "hallucinated": [ extracted_kpi ],
            "missed":       [ gt_kpi ],
            "filtered":     [ extracted_kpi ],
        }
    """
    _REQUIRED_FIELDS = ("name", "key", "value", "year", "units", "row_idx", "col_idx")

    # ------------------------------------------------------------------
    # 1. Load ground-truth record for this source_file
    # ------------------------------------------------------------------
    gt_record: Optional[Dict[str, Any]] = None
    gt_path = Path(ground_truth_file)
    if gt_path.exists():
        with open(gt_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("source_file") == source_file:
                    gt_record = rec
                    break

    # ------------------------------------------------------------------
    # 2. Pre-filter: drop KPIs with any null required field
    # ------------------------------------------------------------------
    filtered: List[Dict[str, Any]] = []
    clean_kpis: List[Dict[str, Any]] = []
    for kpi in kpis:
        if any(kpi.get(f) is None for f in _REQUIRED_FIELDS):
            filtered.append(kpi)
        else:
            clean_kpis.append(kpi)

    if gt_record is None:
        return {
            "source_file": source_file,
            "ground_truth_file": str(ground_truth_file),
            "gt_found": False,
            "statistics": {
                "total_extracted": len(kpis),
                "filtered": len(filtered),
                "total_gt": 0,
                "valid": 0,
                "invalid": 0,
                "hallucinated": len(clean_kpis),
                "missed": 0,
                "valid_rate": 0.0,
                "coverage": 0.0,
            },
            "valid": [],
            "invalid": [],
            "hallucinated": clean_kpis,
            "missed": [],
            "filtered": filtered,
        }

    gt_valid_kpis: List[Dict[str, Any]] = gt_record.get("valid_kpis", [])

    # ------------------------------------------------------------------
    # 3. Build (row_idx, col_idx) → gt_kpi lookup
    # ------------------------------------------------------------------
    gt_by_pos: Dict[tuple, Dict[str, Any]] = {}
    for gt_kpi in gt_valid_kpis:
        pos = (gt_kpi.get("row_idx"), gt_kpi.get("col_idx"))
        gt_by_pos[pos] = gt_kpi

    # ------------------------------------------------------------------
    # 4. Classify each extracted KPI
    # ------------------------------------------------------------------
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    hallucinated: List[Dict[str, Any]] = []
    valid_gt_positions: set = set()

    def _values_match(extracted_val, gt_val) -> bool:
        try:
            return abs(float(extracted_val) - float(gt_val)) <= value_tolerance
        except (TypeError, ValueError):
            return str(extracted_val).strip() == str(gt_val).strip()

    def _str_match(a, b) -> bool:
        return str(a).strip().lower() == str(b).strip().lower()

    for kpi in clean_kpis:
        pos = (kpi.get("row_idx"), kpi.get("col_idx"))
        gt_kpi = gt_by_pos.get(pos)

        # Step 1: position not in GT → hallucinated
        if gt_kpi is None:
            hallucinated.append(kpi)
            continue

        # Step 2: compare all fields
        mismatched: List[str] = []

        if not _str_match(kpi.get("name"), gt_kpi.get("name")):
            mismatched.append(f"name: extracted={kpi.get('name')!r} vs gt={gt_kpi.get('name')!r}")

        if not _str_match(kpi.get("key"), gt_kpi.get("key")):
            mismatched.append(f"key: extracted={kpi.get('key')!r} vs gt={gt_kpi.get('key')!r}")

        if not _values_match(kpi.get("value"), gt_kpi.get("value")):
            mismatched.append(f"value: extracted={kpi.get('value')!r} vs gt={gt_kpi.get('value')!r}")

        if kpi.get("year") != gt_kpi.get("year"):
            mismatched.append(f"year: extracted={kpi.get('year')!r} vs gt={gt_kpi.get('year')!r}")

        if not _str_match(kpi.get("units"), gt_kpi.get("units")):
            mismatched.append(f"units: extracted={kpi.get('units')!r} vs gt={gt_kpi.get('units')!r}")

        if mismatched:
            invalid.append({
                "extracted_kpi": kpi,
                "gt_kpi": gt_kpi,
                "mismatched_fields": mismatched,
            })
        else:
            valid.append({"extracted_kpi": kpi, "gt_kpi": gt_kpi})
            valid_gt_positions.add(pos)

    # ------------------------------------------------------------------
    # 5. Missed: GT KPIs not covered by any valid extracted KPI
    # ------------------------------------------------------------------
    missed: List[Dict[str, Any]] = [
        gt_kpi for pos, gt_kpi in gt_by_pos.items()
        if pos not in valid_gt_positions
    ]

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    n_valid = len(valid)
    n_invalid = len(invalid)
    n_hallucinated = len(hallucinated)
    n_missed = len(missed)
    n_evaluated = n_valid + n_invalid + n_hallucinated
    valid_rate = n_valid / n_evaluated if n_evaluated > 0 else 0.0
    coverage = n_valid / len(gt_valid_kpis) if gt_valid_kpis else 0.0

    return {
        "source_file": source_file,
        "ground_truth_file": str(ground_truth_file),
        "gt_found": True,
        "statistics": {
            "total_extracted": len(kpis),
            "filtered": len(filtered),
            "total_gt": len(gt_valid_kpis),
            "valid": n_valid,
            "invalid": n_invalid,
            "hallucinated": n_hallucinated,
            "missed": n_missed,
            "valid_rate": round(valid_rate * 100, 2),
            "coverage": round(coverage * 100, 2),
        },
        "valid": valid,
        "invalid": invalid,
        "hallucinated": hallucinated,
        "missed": missed,
        "filtered": filtered,
    }


def validate_folder_against_ground_truth(
    extraction_dir: Path,
    ground_truth_file: Path,
    output_dir: Optional[Path] = None,
    value_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Run ``validate_against_ground_truth`` for every ``*_kpis.json`` file in
    ``extraction_dir`` that has a matching record in the ground-truth JSONL.

    Args:
        extraction_dir:     Folder containing ``*_kpis.json`` extraction files.
        ground_truth_file:  Path to the ground-truth JSONL.
        output_dir:         Optional folder to write per-file JSON results.
        value_tolerance:    Passed through to ``validate_against_ground_truth``.

    Returns:
        {
            "ground_truth_file": str,
            "extraction_dir":    str,
            "files_evaluated":   int,
            "files_skipped":     int,
            "aggregate": {
                "total_extracted": int,
                "filtered":        int,
                "total_gt":        int,
                "valid":           int,
                "invalid":         int,
                "hallucinated":    int,
                "missed":          int,
                "valid_rate":      float,
                "coverage":        float,
            },
            "per_file": [ {result dict per file} ],
        }
    """
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    extraction_files = sorted([
        f for f in Path(extraction_dir).glob("*_kpis.json")
        if f.name not in ("extraction_statistics.json", "extraction_summary.json")
    ])

    print("\n" + "=" * 70)
    print("GROUND-TRUTH KPI VALIDATION")
    print("=" * 70)
    print(f"Extraction dir  : {extraction_dir}")
    print(f"Ground truth    : {ground_truth_file}")
    if output_dir:
        print(f"Output dir      : {output_dir}")
    print("=" * 70)

    per_file_results: List[Dict[str, Any]] = []
    skipped = 0
    agg = {
        "total_extracted": 0, "filtered": 0, "total_gt": 0,
        "valid": 0, "invalid": 0, "hallucinated": 0, "missed": 0,
    }

    for ext_file in extraction_files:
        with open(ext_file, encoding="utf-8") as fh:
            data = json.load(fh)
        kpis = data.get("kpis", [])

        result = validate_against_ground_truth(
            kpis=kpis,
            source_file=ext_file.name,
            ground_truth_file=ground_truth_file,
            value_tolerance=value_tolerance,
        )

        if not result["gt_found"]:
            skipped += 1
            continue

        stats = result["statistics"]
        for key in ("total_extracted", "filtered", "total_gt",
                    "valid", "invalid", "hallucinated", "missed"):
            agg[key] += stats[key]

        per_file_results.append(result)

        print(
            f"  {ext_file.name}  ->  "
            f"valid={stats['valid']}  invalid={stats['invalid']}  "
            f"hallucinated={stats['hallucinated']}  missed={stats['missed']}  "
            f"filtered={stats['filtered']}  "
            f"valid_rate={stats['valid_rate']:.1f}%  coverage={stats['coverage']:.1f}%"
        )

        if output_dir is not None:
            out_path = Path(output_dir) / ext_file.name
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)

    # Aggregate rates
    n_evaluated = agg["valid"] + agg["invalid"] + agg["hallucinated"]
    valid_rate = agg["valid"] / n_evaluated if n_evaluated > 0 else 0.0
    coverage = agg["valid"] / agg["total_gt"] if agg["total_gt"] > 0 else 0.0

    summary = {
        "ground_truth_file": str(ground_truth_file),
        "extraction_dir": str(extraction_dir),
        "files_evaluated": len(per_file_results),
        "files_skipped": skipped,
        "aggregate": {
            **agg,
            "valid_rate": round(valid_rate * 100, 2),
            "coverage": round(coverage * 100, 2),
        },
        "per_file": per_file_results,
    }

    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    print(f"  Files evaluated : {len(per_file_results)}  (skipped / no GT: {skipped})")
    print(f"  Extracted KPIs  : {agg['total_extracted']}")
    print(f"  Filtered (null) : {agg['filtered']}")
    print(f"  Ground-truth    : {agg['total_gt']}")
    print(f"  Valid           : {agg['valid']}")
    print(f"  Invalid         : {agg['invalid']}")
    print(f"  Hallucinated    : {agg['hallucinated']}")
    print(f"  Missed          : {agg['missed']}")
    print(f"  Valid rate      : {valid_rate*100:.2f}%")
    print(f"  Coverage        : {coverage*100:.2f}%")
    print("=" * 70)
    if output_dir:
        print(f"\n  Per-file results saved to: {output_dir}")

    return summary


def _parse_year_from_filename(stem: str) -> Optional[int]:
    """Extract 4-digit document year from a KPI filename stem.
    Supports patterns: ar22, ar15, ..."""
    m = re.search(r'ar(\d{2})', stem)
    if m:
        suffix = int(m.group(1))
        return 2000 + suffix if suffix <= 30 else 1900 + suffix
    return None


def _parse_bucket_from_filename(stem: str) -> Optional[str]:
    """Extract bucket ('management' or 'divisions') from a KPI filename stem."""
    lower = stem.lower()
    if 'management' in lower:
        return 'management'
    if 'division' in lower:
        return 'divisions'
    return None


def main():
    """
    Run index-based validation on trial-27/vlm_qwen_72b extraction files.

    For each *_kpis.json file:
      - Year and bucket are parsed from the filename.
      - Page and table_index are taken from the top-level fields.
      - KPIs are validated via validate_kpis() against pack_context.db.
      - Per-file results are written to the output folder.
      - Files where ALL KPIs have no DB match are filtered out of results.
      - Aggregate accuracy/precision is printed at the end.
    """
    base_dir = Path(__file__).parent.parent
    extraction_dir = base_dir / 'data' / 'output' / 'trial-27' / 'vlm_qwen_72b'
    db_path = base_dir / 'kpi_extraction_project' / 'data' / 'pack_context.db'
    output_dir = base_dir / 'data' / 'output' / 'trial-27-validation'
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("INDEX-BASED KPI VALIDATION  —  trial-27 / vlm_qwen_72b")
    print("=" * 70)
    print(f"Input  : {extraction_dir}")
    print(f"DB     : {db_path}")
    print(f"Output : {output_dir}")
    print("=" * 70)

    extraction_files = sorted([
        f for f in extraction_dir.glob('*_kpis.json')
        if f.name not in ('extraction_statistics.json', 'extraction_summary.json')
    ])

    if not extraction_files:
        print("❌  No KPI files found in extraction folder.")
        return

    print(f"\n  Found {len(extraction_files)} file(s) to validate\n")

    all_results = []
    skipped = 0
    no_db_match = 0

    for extraction_file in extraction_files:
        stem = extraction_file.stem  # e.g. "divisions-vw-ar20_page_003_table_00_kpis"

        # --- metadata from filename ---
        doc_year = _parse_year_from_filename(stem)
        bucket   = _parse_bucket_from_filename(stem)

        # --- load file ---
        with open(extraction_file, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        kpis = data.get("kpis", [])
        if not kpis:
            skipped += 1
            continue

        page        = data.get("page")
        table_index = data.get("table_index", 0)

        if bucket == "management":
            if page is not None and doc_year == 2019:
                page -= 4
            if page is not None and doc_year == 2016:
                page -= 5
            if page is not None and doc_year == 2017:
                page -= 2
            if page is not None and doc_year == 2018:
                page -= 4

        if doc_year is None or page is None:
            print(f"  ⚠  Skipping {extraction_file.name}: could not determine year ({doc_year}) or page ({page})")
            skipped += 1
            continue

        # --- validate ---
        validation_output = validate_kpis(
            kpis=kpis,
            db_path=str(db_path),
            table_idx=table_index,
            year=doc_year,
            page=page,
            bucket=bucket,
        )

        stats        = validation_output["statistics"]
        valid_list   = validation_output["valid_kpis"]
        invalid_list = validation_output["invalid_kpis"]

        # Filter out tables with no DB match at all
        if stats["missing_tables"] >= stats["total_kpis"]:
            print(f"  [NO DB MATCH] {extraction_file.name}  ->  skipped ({stats['total_kpis']} KPIs, no matching table in DB)")
            no_db_match += 1
            continue

        print(
            f"  {extraction_file.name}  ->  "
            f"{stats['valid_kpis']}/{stats['total_kpis']} valid  "
            f"(acc {stats['accuracy']:.1f}%,  "
            f"primary={stats['valid_by_primary']},  "
            f"fallback={stats['valid_by_order_fallback']},  "
            f"missing_tables={stats['missing_tables']},  "
            f"dupes={stats['duplicate_kpis']})"
        )

        record = {
            "file": extraction_file.name,
            "year": doc_year,
            "page": page,
            "table_index": table_index,
            "bucket": bucket,
            "statistics": stats,
        }
        all_results.append(record)

        # --- persist per-file results ---
        out_file = output_dir / extraction_file.name
        out_data = {
            "source_file": extraction_file.name,
            "year": doc_year,
            "page": page,
            "table_index": table_index,
            "bucket": bucket,
            "validation_method": "index-based (row_idx / col_idx)",
            "statistics": stats,
            "valid_kpis": valid_list,
            "invalid_kpis": invalid_list,
        }
        with open(out_file, 'w', encoding='utf-8') as fh:
            json.dump(out_data, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Overall summary (only tables with at least one DB match)
    # ------------------------------------------------------------------
    validated = [r for r in all_results]
    if not validated:
        print("\n⚠  No files were validated.")
        return

    total_kpis       = sum(r["statistics"]["total_kpis"]             for r in validated)
    total_valid      = sum(r["statistics"]["valid_kpis"]             for r in validated)
    total_missing    = sum(r["statistics"]["missing_tables"]          for r in validated)
    total_dupes      = sum(r["statistics"]["duplicate_kpis"]          for r in validated)
    total_primary    = sum(r["statistics"]["valid_by_primary"]        for r in validated)
    total_fallback   = sum(r["statistics"]["valid_by_order_fallback"] for r in validated)

    # Fully-matched accuracy: also exclude partial-miss KPIs within matched tables
    fully_matched = [r for r in validated if r["statistics"]["missing_tables"] == 0]
    fm_kpis  = sum(r["statistics"]["total_kpis"]  for r in fully_matched)
    fm_valid = sum(r["statistics"]["valid_kpis"]  for r in fully_matched)

    overall_acc     = (total_valid / total_kpis * 100) if total_kpis > 0 else 0.0
    fully_match_acc = (fm_valid    / fm_kpis    * 100) if fm_kpis   > 0 else 0.0

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Files processed         : {len(validated)}  (skipped: {skipped},  no DB match: {no_db_match})")
    print(f"  Total KPIs              : {total_kpis}")
    print(f"  Valid KPIs              : {total_valid}")
    print(f"    -> by primary check   : {total_primary}")
    print(f"    -> by order fallback  : {total_fallback}")
    print(f"  Accuracy (w/ DB match)  : {overall_acc:.2f}%")
    print(f"  Accuracy (full match)   : {fully_match_acc:.2f}%  (excl. {total_missing} partial-miss KPIs, {len(fully_matched)} tables)")
    print(f"  Duplicate KPIs          : {total_dupes}")
    print("=" * 70)
    print(f"\n  Per-file results saved to: {output_dir}")


def main_annotation_validation():
    """
    Run annotation-based validation on trial-27/vlm_qwen_72b using the
    human-annotated JSONL from the vw-management factgenie campaign.
    """
    import sys
    base_dir = Path(__file__).parent.parent
    annotation_file   = base_dir / 'factgenie' / 'factgenie' / 'campaigns' / 'vw-management' / 'files' / '2-1-annotater_2-1776933375.jsonl'
    factgenie_outputs = base_dir / 'factgenie' / 'factgenie' / 'data' / 'outputs' / 'trial27'
    extraction_dir    = base_dir / 'data' / 'output' / 'trial-27' / 'vlm_qwen_72b'
    output_dir        = base_dir / 'data' / 'output' / 'trial-27-annotation-validation'

    validate_trial27_with_annotations(
        annotation_file=annotation_file,
        factgenie_outputs_dir=factgenie_outputs,
        extraction_dir=extraction_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import sys
    if '--annotate' in sys.argv:
        main_annotation_validation()
    else:
        main()
