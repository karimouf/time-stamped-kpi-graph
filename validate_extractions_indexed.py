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
    # Examples: "2,0524" → 2.0524, "1.234,56" → 1234.56
    # Check if comma appears after digits and is followed by exactly 1-4 digits (decimal part)
    european_decimal_pattern = r'^-?[\d\s]*,\d{1,4}$'
    if re.match(european_decimal_pattern, text.replace(' ', '')):
        # European format: comma is decimal separator, space/period is thousands separator
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


def validate_kpi_indexed(
    kpi: Dict[str, Any],
    table_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate one KPI using BOTH indices and names for robust verification.
    
    PRIMARY: Uses row_idx and col_idx directly (no string matching)
    SECONDARY: Cross-validates with row_name and col_name for accuracy
    
    This dual approach catches:
    - Index errors (off-by-one, wrong convention)
    - Name mismatches (LLM extraction errors)
    - Inconsistent extraction (indices don't match names)
    """
    result = {
        "is_valid": True,
        "confidence": 1.0,
        "errors": [],
        "source_cell_value": None,
        "source_cell_text": None,
        "extracted_value": kpi.get("value"),
        "row_idx": None,
        "col_idx": None,
        "row_name_match": None,
        "col_name_match": None
    }
    
    # Extract fields
    row_idx = kpi.get("row_idx")
    col_idx = kpi.get("col_idx")
    row_name = kpi.get("row_name")
    col_name = kpi.get("col_name")
    extracted_value = kpi.get("value")
    
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
    
    result["row_idx"] = row_idx
    result["col_idx"] = col_idx
    
    # Validate row index bounds
    if row_idx < 0 or row_idx >= len(rows):
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append(f"row_idx {row_idx} out of bounds (table has {len(rows)} rows)")
        return result
    
    # Validate column index bounds
    if col_idx < 0 or col_idx >= len(rows[row_idx]):
        result["is_valid"] = False
        result["confidence"] = 0.0
        result["errors"].append(f"col_idx {col_idx} out of bounds (row has {len(rows[row_idx])} cols)")
        return result
    
    # Cross-validate row_name with stub_col[row_idx]
    if stub_col and row_idx < len(stub_col):
        expected_row_name = stub_col[row_idx]
        result["row_name_match"] = expected_row_name
        
        if row_name:
            # Normalize for comparison (strip whitespace, case-insensitive)
            row_name_norm = str(row_name).strip()
            expected_norm = str(expected_row_name).strip()
            
            if row_name_norm == expected_norm:
                # Perfect match - boost confidence
                result["confidence"] *= 1.0
                result["errors"].append(f"✓ row_name verified: '{row_name}'")
            elif row_name_norm.lower() == expected_norm.lower():
                # Case-insensitive match - acceptable
                result["confidence"] *= 0.98
                result["errors"].append(f"⚠ row_name case mismatch: KPI='{row_name}', expected='{expected_row_name}'")
            else:
                # Complete mismatch - likely extraction error
                result["is_valid"] = False
                result["confidence"] = 0.5
                result["errors"].append(f"❌ row_name MISMATCH: KPI='{row_name}', stub_col[{row_idx}]='{expected_row_name}'")
        else:
            # row_name missing but we have index - warning only
            result["confidence"] *= 0.95
            result["errors"].append(f"⚠ row_name missing, using stub_col[{row_idx}]='{expected_row_name}'")
    
    # Cross-validate col_name with merged_headers[col_idx]
    if merged_headers and col_idx < len(merged_headers):
        expected_col_name = merged_headers[col_idx]
        result["col_name_match"] = expected_col_name
        
        if col_name:
            # Normalize for comparison (strip whitespace, case-insensitive)
            col_name_norm = str(col_name).strip()
            expected_norm = str(expected_col_name).strip()
            
            if col_name_norm == expected_norm:
                # Perfect match - boost confidence
                result["confidence"] *= 1.0
                result["errors"].append(f"✓ col_name verified: '{col_name}'")
            elif col_name_norm.lower() == expected_norm.lower():
                # Case-insensitive match - acceptable
                result["confidence"] *= 0.98
                result["errors"].append(f"⚠ col_name case mismatch: KPI='{col_name}', expected='{expected_col_name}'")
            else:
                # Complete mismatch - likely extraction error
                result["is_valid"] = False
                result["confidence"] = 0.5
                result["errors"].append(f"❌ col_name MISMATCH: KPI='{col_name}', merged_headers[{col_idx}]='{expected_col_name}'")
        else:
            # col_name missing but we have index - warning only
            result["confidence"] *= 0.95
            result["errors"].append(f"⚠ col_name missing, using merged_headers[{col_idx}]='{expected_col_name}'")
    
    # Extract cell value using indices directly
    cell_text = rows[row_idx][col_idx]
    result["source_cell_text"] = cell_text
    
    # Parse numeric value
    source_value = parse_numeric_value(str(cell_text))
    result["source_cell_value"] = source_value
    
    # FALLBACK: If col_idx=0 has no numeric value, try col_idx+1
    # This handles cases where LLM incorrectly used col_idx=0 (row label column)
    if source_value is None and col_idx == 0 and col_idx + 1 < len(rows[row_idx]):
        alt_cell_text = rows[row_idx][col_idx + 1]
        alt_source_value = parse_numeric_value(str(alt_cell_text))
        
        if alt_source_value is not None:
            # Check if col_name matches the next column header
            if merged_headers and col_idx + 1 < len(merged_headers):
                expected_alt_col = merged_headers[col_idx + 1]
                # If col_name matches col_idx+1 better than col_idx, use it
                if col_name and str(col_name).strip() == str(expected_alt_col).strip():
                    result["errors"].append(f"⚠ Auto-corrected: col_idx=0 (row label) → col_idx=1, found value={alt_source_value}")
                    result["col_idx"] = col_idx + 1
                    result["source_cell_text"] = alt_cell_text
                    source_value = alt_source_value
                    result["source_cell_value"] = source_value
                    result["col_name_match"] = expected_alt_col
                    result["confidence"] *= 0.95  # Slight penalty for auto-correction
    
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
        
        # Compare with tolerance
        diff = abs(source_numeric - extracted_numeric)
        rel_diff = diff / max(abs(source_numeric), 1e-9)
        
        if diff <= 1e-6:
            # Perfect match
            pass  # confidence already 1.0
        elif rel_diff < 0.01:
            # Within 1% - acceptable
            result["confidence"] *= 0.98
            result["errors"].append(f"Small diff: {diff:.6f}")
        elif rel_diff < 0.05:
            # Within 5% - moderate error
            result["is_valid"] = False
            result["confidence"] = 0.7
            result["errors"].append(f"Moderate diff: extracted={extracted_numeric}, source={source_numeric}, diff={diff:.2f}")
        else:
            # Large difference - try col_idx+1 as fallback before marking invalid
            # This catches cases where LLM used col_idx when it should be col_idx+1
            tried_fallback = False
            if col_idx + 1 < len(rows[row_idx]):
                alt_cell_text = rows[row_idx][col_idx + 1]
                alt_source_value = parse_numeric_value(str(alt_cell_text))
                
                if alt_source_value is not None:
                    try:
                        alt_source_numeric = float(alt_source_value) if not isinstance(alt_source_value, (int, float)) else alt_source_value
                        alt_diff = abs(alt_source_numeric - extracted_numeric)
                        alt_rel_diff = alt_diff / max(abs(alt_source_numeric), 1e-9)
                        
                        # If col_idx+1 is a much better match, use it
                        if alt_diff <= 1e-6 or (alt_rel_diff < 0.01 and alt_rel_diff < rel_diff * 0.5):
                            tried_fallback = True
                            result["errors"].append(f"⚠ Auto-corrected: col_idx={col_idx} → col_idx={col_idx+1}, value match improved ({source_numeric} → {alt_source_numeric})")
                            result["col_idx"] = col_idx + 1
                            result["source_cell_text"] = alt_cell_text
                            result["source_cell_value"] = alt_source_value
                            result["confidence"] *= 0.95
                            
                            # Update col_name_match if available
                            if merged_headers and col_idx + 1 < len(merged_headers):
                                result["col_name_match"] = merged_headers[col_idx + 1]
                            
                            # Re-validate with corrected value
                            if alt_diff <= 1e-6:
                                result["is_valid"] = True
                            elif alt_rel_diff < 0.01:
                                result["is_valid"] = True
                                result["confidence"] *= 0.98
                                result["errors"].append(f"Small diff after correction: {alt_diff:.6f}")
                            elif alt_rel_diff < 0.05:
                                result["is_valid"] = False
                                result["confidence"] = 0.7
                                result["errors"].append(f"Moderate diff after correction: {alt_diff:.2f}")
                            else:
                                result["is_valid"] = False
                                result["confidence"] = 0.3
                                result["errors"].append(f"Large diff even after correction: {alt_diff:.2f}")
                            
                            return result
                    except (ValueError, TypeError):
                        pass
            
            # If fallback didn't help or wasn't available, mark as invalid
            if not tried_fallback:
                result["is_valid"] = False
                result["confidence"] = 0.3
                result["errors"].append(f"Large diff: extracted={extracted_numeric}, source={source_numeric}, diff={diff:.2f}")
    
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
    stats = {
        "total_kpis": 0,
        "valid_kpis": 0,
        "invalid_kpis": 0,
        "tables_processed": 0,
        "row_name_mismatches": 0,
        "col_name_mismatches": 0,
        "row_name_verified": 0,
        "col_name_verified": 0,
        "name_mismatches": 0  # Total (row + col)
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
            
            # Step 6-8: Validate using indices directly
            validation = validate_kpi_indexed(kpi, source_table)
            
            # Count name verification results
            for err in validation["errors"]:
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
        "invalid_kpis": invalid_kpis
    }


def main():
    """
    Run index-based validation on Seventh-trial-deepseek folder.
    
    LOGIC FLOW (INDEX-BASED):
    ==========================
    1. Loop through all extraction files in Seventh-trial-deepseek folder
    2. For each file:
       a. Loop through all tables in the file
       b. Use table_id to locate the corresponding source table
       c. Loop through all KPIs in each table
       d. For each KPI:
          - Use row_idx DIRECTLY (no string matching)
          - Use col_idx DIRECTLY (no fuzzy matching)
          - Access cell: rows[row_idx][col_idx]
          - Optional: Verify row_name matches stub_col[row_idx]
          - Optional: Verify col_name matches merged_headers[col_idx]
          - Compare extracted value vs source value
       e. Save only invalid KPIs to output file
    """
    # Paths
    base_dir = Path(__file__).parent
    extraction_dir = base_dir / 'data' / 'output' / 'Eighth-Trial-deepseek'
    tables_dir = base_dir / 'data' / 'tables'
    output_dir = extraction_dir / 'validation_reports_eighth'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("KPI EXTRACTION VALIDATION - SEVENTH TRIAL (INDEX-BASED)")
    print("=" * 70)
    print(f"Extraction folder: {extraction_dir}")
    print(f"Tables folder: {tables_dir}")
    print("=" * 70)
    
    # DEMO MODE: Only process 2020 file for testing
    DEMO_MODE = False
    
    if DEMO_MODE:
        print("\n🔍 DEMO MODE: Processing 2020 only\n")
        extraction_files = list(extraction_dir.glob('*linked_tables(2020).json'))
    else:
        print("\n📂 Processing ALL extraction files\n")
        extraction_files = sorted(extraction_dir.glob('*.json'))
    
    if not extraction_files:
        print("❌ No extraction files found!")
        return
    
    print(f"Found {len(extraction_files)} file(s) to process\n")
    
    # Step 1: Loop through all files
    results = []
    for extraction_file in extraction_files:
        print(f"\n{'=' * 70}")
        print(f"Processing: {extraction_file.name}")
        print(f"{'=' * 70}")
        
        # Validate this file (steps 2-4 happen inside this function)
        result = validate_extraction_file(extraction_file, tables_dir)
        
        if result:
            results.append(result)
            
            # Print summary
            stats = result["stats"]
            print(f"\n✅ Validation complete:")
            print(f"   Tables processed: {stats['tables_processed']}")
            print(f"   Total KPIs: {stats['total_kpis']}")
            print(f"   Valid: {stats['valid_kpis']}")
            print(f"   Invalid: {stats['invalid_kpis']}")
            print(f"   Accuracy: {stats['accuracy']:.2f}%")
            print(f"\n   Name Validation:")
            print(f"   ✓ row_name verified: {stats['row_name_verified']}")
            print(f"   ✓ col_name verified: {stats['col_name_verified']}")
            print(f"   ✗ row_name mismatches: {stats['row_name_mismatches']}")
            print(f"   ✗ col_name mismatches: {stats['col_name_mismatches']}")
            print(f"   Total name issues: {stats['name_mismatches']}")
            
            # Save only invalid KPIs
            if result["invalid_kpis"]:
                output_file = output_dir / f"invalid_{extraction_file.stem}.json"
                invalid_report = {
                    "source_file": extraction_file.name,
                    "year": result["year"],
                    "validation_method": "index-based (row_idx, col_idx)",
                    "total_invalid": len(result["invalid_kpis"]),
                    "invalid_kpis": result["invalid_kpis"],
                    "statistics": stats
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(invalid_report, f, indent=2, ensure_ascii=False)
                
                print(f"   📄 Invalid KPIs saved to: {output_file.name}")
            else:
                print(f"   🎉 No invalid KPIs - 100% accuracy!")
    
    # Overall summary
    if results:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY (INDEX-BASED VALIDATION)")
        print("=" * 70)
        total_kpis = sum(r["stats"]["total_kpis"] for r in results)
        total_valid = sum(r["stats"]["valid_kpis"] for r in results)
        total_invalid = sum(r["stats"]["invalid_kpis"] for r in results)
        total_row_verified = sum(r["stats"]["row_name_verified"] for r in results)
        total_col_verified = sum(r["stats"]["col_name_verified"] for r in results)
        total_row_mismatches = sum(r["stats"]["row_name_mismatches"] for r in results)
        total_col_mismatches = sum(r["stats"]["col_name_mismatches"] for r in results)
        total_mismatches = sum(r["stats"]["name_mismatches"] for r in results)
        overall_accuracy = (total_valid / total_kpis * 100) if total_kpis > 0 else 0
        
        print(f"Files processed: {len(results)}")
        print(f"Total KPIs: {total_kpis}")
        print(f"Valid: {total_valid}")
        print(f"Invalid: {total_invalid}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"\nName Validation Summary:")
        print(f"✓ row_name verified: {total_row_verified}")
        print(f"✓ col_name verified: {total_col_verified}")
        print(f"✗ row_name mismatches: {total_row_mismatches}")
        print(f"✗ col_name mismatches: {total_col_mismatches}")
        print(f"Total name issues: {total_mismatches}")
        print("=" * 70)
        print("\n💡 DUAL VALIDATION APPROACH:")
        print("   PRIMARY: Uses row_idx and col_idx directly (instant cell access)")
        print("   SECONDARY: Cross-validates with row_name and col_name")
        print("   - Catches index errors (off-by-one, wrong convention)")
        print("   - Catches name mismatches (LLM extraction errors)")
        print("   - Ensures consistency between indices and names")
        
        if DEMO_MODE:
            print("\n💡 To run on all files, set DEMO_MODE = False in the script")
    
    print(f"\n✅ Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
