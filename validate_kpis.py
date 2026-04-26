#!/usr/bin/env python3
"""
KPI Validation Script

Validates extracted KPIs from individual files against ground truth data.
random_15_tables.json is the ground truth (corrected).
Extracted files are in vlm_qwen_32b/*.json

Usage:
    python validate_kpis.py --ground-truth random_15_tables.json --extracted-folder data/output/trial-21/output/vlm_qwen_32b --output validation_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import difflib


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_kpis_by_table(kpis: List[Dict]) -> Dict[str, List[Dict]]:
    """Group KPIs by their source table."""
    tables = defaultdict(list)
    for kpi in kpis:
        source = kpi.get('source', '')
        tables[source].append(kpi)
    return dict(tables)


def find_matching_kpi(
    extracted_kpi: Dict,
    ground_truth_kpis: List[Dict],
    tolerance: float = 0.01
) -> Tuple[Optional[Dict], str]:
    """
    Find matching KPI in ground truth.
    
    Strategy:
    1. Primary: Match by value (99.9% accurate)
    2. Secondary: For duplicate values, use row_idx/col_idx proximity
    
    Returns:
        (matched_kpi, match_reason)
    """
    value = extracted_kpi.get('value')
    row_idx = extracted_kpi.get('row_idx')
    col_idx = extracted_kpi.get('col_idx')
    
    # Find all KPIs with matching value
    value_matches = []
    for gt_kpi in ground_truth_kpis:
        gt_value = gt_kpi.get('value')
        
        # Handle numeric comparison with tolerance
        if isinstance(value, (int, float)) and isinstance(gt_value, (int, float)):
            if abs(value - gt_value) <= tolerance:
                value_matches.append(gt_kpi)
        # Exact match for strings/other types
        elif value == gt_value:
            value_matches.append(gt_kpi)
    
    # No value match found
    if not value_matches:
        return None, "no_value_match"
    
    # Single value match - perfect
    if len(value_matches) == 1:
        return value_matches[0], "unique_value_match"
    
    # Multiple value matches - use row_idx/col_idx proximity
    if row_idx is not None and col_idx is not None:
        best_match = None
        best_distance = float('inf')
        
        for gt_kpi in value_matches:
            gt_row = gt_kpi.get('row_idx')
            gt_col = gt_kpi.get('col_idx')
            
            if gt_row is not None and gt_col is not None:
                # Manhattan distance
                distance = abs(row_idx - gt_row) + abs(col_idx - gt_col)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = gt_kpi
        
        if best_match and best_distance == 0:
            return best_match, "exact_position_match"
        elif best_match:
            return best_match, f"closest_position_match (distance={best_distance})"
    
    # Fallback: return first match
    return value_matches[0], "ambiguous_match (multiple_values)"


def compare_kpi_fields(extracted: Dict, ground_truth: Dict) -> Dict[str, bool]:
    """Compare individual fields between extracted and ground truth KPI."""
    comparison = {}
    
    fields_to_check = ['name', 'key', 'country', 'value', 'year', 'units', 'row_idx', 'col_idx']
    
    for field in fields_to_check:
        ext_val = extracted.get(field)
        gt_val = ground_truth.get(field)
        
        # Handle numeric comparison with small tolerance
        if field == 'value' and isinstance(ext_val, (int, float)) and isinstance(gt_val, (int, float)):
            comparison[field] = abs(ext_val - gt_val) <= 0.01
        else:
            comparison[field] = ext_val == gt_val
    
    return comparison


def validate_table_kpis(
    extracted_kpis: List[Dict],
    ground_truth_kpis: List[Dict],
    table_source: str
) -> Dict:
    """
    Validate all KPIs for a single table.
    
    Returns:
        Validation report for the table
    """
    report = {
        'table_source': table_source,
        'extracted_count': len(extracted_kpis),
        'ground_truth_count': len(ground_truth_kpis),
        'matched_kpis': [],
        'unmatched_kpis': [],
        'extra_kpis': [],
        'missing_kpis': [],
        'field_mismatches': defaultdict(int),
        'match_quality': {
            'unique_value_match': 0,
            'exact_position_match': 0,
            'closest_position_match': 0,
            'ambiguous_match': 0,
            'no_match': 0
        }
    }
    
    # Track which ground truth KPIs have been matched
    matched_gt_indices = set()
    # Track which extracted KPIs have been matched
    matched_ext_indices = set()
    
    # Validate each extracted KPI
    for ext_idx, ext_kpi in enumerate(extracted_kpis):
        # Find matching ground truth KPI
        matched_gt, match_reason = find_matching_kpi(ext_kpi, ground_truth_kpis)
        
        if matched_gt:
            # Compare fields
            field_comparison = compare_kpi_fields(ext_kpi, matched_gt)
            
            match_info = {
                'extracted': ext_kpi,
                'ground_truth': matched_gt,
                'match_reason': match_reason,
                'field_comparison': field_comparison,
                'all_fields_match': all(field_comparison.values())
            }
            
            report['matched_kpis'].append(match_info)
            matched_ext_indices.add(ext_idx)
            
            # Track match quality
            match_type = match_reason.split('(')[0].strip()
            if match_type in report['match_quality']:
                report['match_quality'][match_type] += 1
            else:
                report['match_quality']['closest_position_match'] += 1
            
            # Track field mismatches
            for field, matches in field_comparison.items():
                if not matches:
                    report['field_mismatches'][field] += 1
            
            # Mark ground truth as matched
            try:
                gt_index = ground_truth_kpis.index(matched_gt)
                matched_gt_indices.add(gt_index)
            except ValueError:
                pass
        else:
            report['unmatched_kpis'].append({
                'kpi': ext_kpi,
                'reason': 'No matching value found in ground truth'
            })
            report['match_quality']['no_match'] += 1
    
    # Find extra KPIs (in extracted but not properly matched - these are false positives)
    for ext_idx, ext_kpi in enumerate(extracted_kpis):
        if ext_idx not in matched_ext_indices:
            report['extra_kpis'].append(ext_kpi)
    
    # Find missing KPIs (in ground truth but not matched)
    for idx, gt_kpi in enumerate(ground_truth_kpis):
        if idx not in matched_gt_indices:
            report['missing_kpis'].append(gt_kpi)
    
    # Calculate metrics
    total_matched = len(report['matched_kpis'])
    perfect_matches = sum(1 for m in report['matched_kpis'] if m['all_fields_match'])
    
    report['metrics'] = {
        'match_rate': total_matched / len(extracted_kpis) if extracted_kpis else 0,
        'perfect_match_rate': perfect_matches / total_matched if total_matched else 0,
        'recall': total_matched / len(ground_truth_kpis) if ground_truth_kpis else 0,
        'count_match': len(extracted_kpis) == len(ground_truth_kpis)
    }
    
    return report


def generate_summary_report(table_reports: List[Dict]) -> Dict:
    """Generate overall summary from all table reports."""
    summary = {
        'total_tables': len(table_reports),
        'total_extracted_kpis': sum(r['extracted_count'] for r in table_reports),
        'total_ground_truth_kpis': sum(r['ground_truth_count'] for r in table_reports),
        'total_matched': sum(len(r['matched_kpis']) for r in table_reports),
        'total_perfect_matches': sum(
            sum(1 for m in r['matched_kpis'] if m['all_fields_match'])
            for r in table_reports
        ),
        'total_unmatched': sum(len(r['unmatched_kpis']) for r in table_reports),
        'total_missing': sum(len(r['missing_kpis']) for r in table_reports),
        'total_extra': sum(len(r['extra_kpis']) for r in table_reports),
        'tables_with_issues': [],
        'field_mismatch_summary': defaultdict(int),
        'match_quality_summary': defaultdict(int)
    }
    
    # Aggregate field mismatches and match quality
    for report in table_reports:
        for field, count in report['field_mismatches'].items():
            summary['field_mismatch_summary'][field] += count
        
        for match_type, count in report['match_quality'].items():
            summary['match_quality_summary'][match_type] += count
        
        # Track tables with issues
        if (report['unmatched_kpis'] or 
            report['missing_kpis'] or 
            report['extra_kpis'] or 
            not report['metrics']['count_match']):
            summary['tables_with_issues'].append({
                'table': report['table_source'],
                'extracted': report['extracted_count'],
                'ground_truth': report['ground_truth_count'],
                'unmatched': len(report['unmatched_kpis']),
                'missing': len(report['missing_kpis']),
                'extra': len(report['extra_kpis'])
            })
    
    # Calculate overall metrics
    if summary['total_extracted_kpis'] > 0:
        summary['overall_match_rate'] = summary['total_matched'] / summary['total_extracted_kpis']
    else:
        summary['overall_match_rate'] = 0
    
    if summary['total_matched'] > 0:
        summary['overall_perfect_rate'] = summary['total_perfect_matches'] / summary['total_matched']
    else:
        summary['overall_perfect_rate'] = 0
    
    if summary['total_ground_truth_kpis'] > 0:
        summary['overall_recall'] = summary['total_matched'] / summary['total_ground_truth_kpis']
    else:
        summary['overall_recall'] = 0
    
    return summary


def calculate_correct_kv_percentage(table_reports: List[Dict]) -> float:
    fields = ['name', 'key', 'country', 'value', 'year', 'units', 'row_idx', 'col_idx']
    total_pairs = 0
    correct_pairs = 0
    for report in table_reports:
        for match in report['matched_kpis']:
            field_comparison = match.get('field_comparison', {})
            for field in fields:
                total_pairs += 1
                if field_comparison.get(field, False):
                    correct_pairs += 1
    if total_pairs == 0:
        return 0.0
    return correct_pairs / total_pairs


def load_extracted_kpis_from_folder(folder_path: Path) -> Dict[str, List[Dict]]:
    """Load all extracted KPI files from a folder."""
    extracted = {}
    for json_file in folder_path.glob("*_kpis.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            kpis = data.get('kpis', [])
            # Use just the filename as key to match new ground truth source format
            extracted[json_file.name] = kpis
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Validate extracted KPIs against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        required=True,
        help='Path to ground truth JSON file (e.g., random_15_tables.json)'
    )
    
    parser.add_argument(
        '--extracted-folder', '-e',
        required=True,
        help='Path to folder with extracted KPI JSON files'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='validation_report.json',
        help='Output validation report path (default: validation_report.json)'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.01,
        help='Numeric comparison tolerance (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    print("Loading ground truth...")
    ground_truth_data = load_json(args.ground_truth)
    ground_truth_kpis = ground_truth_data.get('kpis', [])
    
    print(f"Ground Truth KPIs: {len(ground_truth_kpis)}")
    
    print("\nLoading extracted files...")
    extracted_folder = Path(args.extracted_folder)
    if not extracted_folder.exists():
        print(f"Error: Extracted folder not found: {args.extracted_folder}")
        return 1
    
    extracted_files = load_extracted_kpis_from_folder(extracted_folder)
    total_extracted = sum(len(kpis) for kpis in extracted_files.values())
    
    print(f"Extracted files: {len(extracted_files)}")
    print(f"Total Extracted KPIs: {total_extracted}")
    
    # Group ground truth by source
    print("\nGrouping ground truth by table...")
    ground_truth_by_table = group_kpis_by_table(ground_truth_kpis)
    print("\nGrouping ground truth by table...")
    ground_truth_by_table = group_kpis_by_table(ground_truth_kpis)
    
    print(f"Ground truth tables: {len(ground_truth_by_table)}")
    print(f"Extracted tables: {len(extracted_files)}")
    
    # Validate each table
    print("\nValidating tables...")
    table_reports = []
    
    for table_source, gt_kpis in ground_truth_by_table.items():
        print(f"  Validating: {table_source}")
        
        ext_kpis = extracted_files.get(table_source, [])
        
        if not ext_kpis:
            print(f"    ⚠ No extracted file found for {table_source}")
            continue
        
        report = validate_table_kpis(ext_kpis, gt_kpis, table_source)
        table_reports.append(report)
        
        print(f"    Extracted: {report['extracted_count']}, "
              f"Ground Truth: {report['ground_truth_count']}, "
              f"Matched: {len(report['matched_kpis'])}")
    
    # Generate summary
    print("\nGenerating summary report...")
    summary = generate_summary_report(table_reports)
    # Calculate correct key-value pair percentage
    correct_kv_percentage = calculate_correct_kv_percentage(table_reports)
    summary['correct_kv_percentage'] = correct_kv_percentage
    
    # Create full report
    full_report = {
        'summary': summary,
        'table_reports': table_reports
    }
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Validation report saved to: {args.output}")
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total Tables: {summary['total_tables']}")
    print(f"Total Extracted KPIs: {summary['total_extracted_kpis']}")
    print(f"Total Ground Truth KPIs: {summary['total_ground_truth_kpis']}")
    print(f"Total Matched: {summary['total_matched']}")
    print(f"Total Perfect Matches: {summary['total_perfect_matches']}")
    print(f"Total Unmatched: {summary['total_unmatched']}")
    print(f"Total Missing: {summary['total_missing']}")
    print(f"Total Extra: {summary['total_extra']}")
    print(f"\nOverall Match Rate: {summary['overall_match_rate']:.1%}")
    print(f"Overall Perfect Rate: {summary['overall_perfect_rate']:.1%}")
    print(f"Overall Recall: {summary['overall_recall']:.1%}")
    print(f"Correct Key-Value Pairs: {correct_kv_percentage:.1%}")
    
    if summary['tables_with_issues']:
        print(f"\n⚠ Tables with issues: {len(summary['tables_with_issues'])}")
        for issue in summary['tables_with_issues'][:5]:
            print(f"  - {issue['table']}: "
                  f"Extracted={issue['extracted']}, "
                  f"GT={issue['ground_truth']}, "
                  f"Unmatched={issue['unmatched']}, "
                  f"Missing={issue['missing']}, "
                  f"Extra={issue['extra']}")
        if len(summary['tables_with_issues']) > 5:
            print(f"  ... and {len(summary['tables_with_issues']) - 5} more")
    
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
