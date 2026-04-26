#!/usr/bin/env python3
"""
Extraction Statistics Analyzer
==============================

Analyzes extraction summary files from VLM KPI extraction to generate comprehensive
statistics and insights about the extraction process performance.

This script processes extraction_summary.json files and generates detailed statistics
including extraction success rates, KPI distribution, validation results, and more.

Author: Karim Ouf
Date: February 2026
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime

from logger import logger


class ExtractionStatsAnalyzer:
    """Analyzes extraction summary files to generate comprehensive statistics."""
    
    def __init__(self):
        self.summary_data = None
        self.stats = {}
    
    def load_summary(self, summary_path: str) -> bool:
        """Load extraction summary from JSON file."""
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                self.summary_data = json.load(f)
            
            logger.info(f"✓ Loaded extraction summary: {summary_path}")
            logger.info(f"  Model: {self.summary_data.get('model', 'Unknown')}")
            logger.info(f"  Tables: {self.summary_data.get('total_tables', 0)}")
            logger.info(f"  KPIs: {self.summary_data.get('total_kpis', 0)}")
            logger.info(f"  Date: {self.summary_data.get('extraction_date', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load summary: {str(e)}")
            return False
    
    def analyze_extraction_performance(self) -> Dict[str, Any]:
        """Analyze overall extraction performance metrics."""
        if not self.summary_data:
            return {}
        
        results = self.summary_data.get('results', [])
        total_tables = len(results)
        
        # Count successful vs failed extractions
        successful_tables = 0
        failed_tables = 0
        tables_with_kpis = 0
        tables_without_kpis = 0
        
        for result in results:
            if 'error' in result:
                failed_tables += 1
            else:
                successful_tables += 1
                
                kpis = result.get('kpis', [])
                if kpis:
                    tables_with_kpis += 1
                else:
                    tables_without_kpis += 1
        
        # Calculate rates
        success_rate = (successful_tables / total_tables * 100) if total_tables > 0 else 0
        kpi_yield_rate = (tables_with_kpis / total_tables * 100) if total_tables > 0 else 0
        
        performance_stats = {
            'total_tables_processed': total_tables,
            'successful_extractions': successful_tables,
            'failed_extractions': failed_tables,
            'tables_with_kpis': tables_with_kpis,
            'tables_without_kpis': tables_without_kpis,
            'success_rate_percent': round(success_rate, 2),
            'kpi_yield_rate_percent': round(kpi_yield_rate, 2)
        }
        
        return performance_stats
    
    def analyze_kpi_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of extracted KPIs."""
        if not self.summary_data:
            return {}
        
        all_kpis = self.summary_data.get('all_kpis', [])
        
        # Basic counts
        total_kpis = len(all_kpis)
        
        # Analyze by fields
        names_counter = Counter()
        keys_counter = Counter()
        countries_counter = Counter()
        units_counter = Counter()
        years_counter = Counter()
        
        # Value statistics
        values = []
        null_values = 0
        
        for kpi in all_kpis:
            # Count field values
            names_counter[kpi.get('name', 'Unknown')] += 1
            keys_counter[kpi.get('key', 'Unknown')] += 1
            countries_counter[kpi.get('country', 'Unknown')] += 1
            units_counter[kpi.get('units', 'Unknown')] += 1
            
            year = kpi.get('year')
            if year is not None:
                years_counter[year] += 1
            else:
                years_counter['null'] += 1
            
            # Collect values for statistics
            value = kpi.get('value')
            if value is not None:
                try:
                    # Handle string values like '+2.7%', '-1.5%', etc.
                    if isinstance(value, str):
                        cleaned = value.strip().replace(',', '').replace('%', '').replace('+', '')
                        values.append(float(cleaned))
                    else:
                        values.append(float(value))
                except (ValueError, TypeError):
                    null_values += 1
            else:
                null_values += 1
        
        # Calculate value statistics
        value_stats = {}
        if values:
            values.sort()
            value_stats = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': values[len(values)//2],
                'null_count': null_values,
                'null_percentage': round(null_values / total_kpis * 100, 2)
            }
        
        kpi_distribution = {
            'total_kpis': total_kpis,
            'unique_names': len(names_counter),
            'unique_keys': len(keys_counter),
            'unique_countries': len(countries_counter),
            'unique_units': len(units_counter),
            'unique_years': len([y for y in years_counter.keys() if y != 'null']),
            'top_names': dict(names_counter.most_common(10)),
            'top_keys': dict(keys_counter.most_common(10)),
            'top_countries': dict(countries_counter.most_common(10)),
            'top_units': dict(units_counter.most_common(10)),
            'years_distribution': dict(years_counter),
            'value_statistics': value_stats
        }
        
        return kpi_distribution
    
    def analyze_table_performance(self) -> Dict[str, Any]:
        """Analyze per-table extraction performance."""
        if not self.summary_data:
            return {}
        
        results = self.summary_data.get('results', [])
        
        # Per-table statistics
        table_stats = []
        kpis_per_table = []
        
        for i, result in enumerate(results):
            page = result.get('page', 'Unknown')
            table_idx = result.get('table_index', 'Unknown')
            kpis = result.get('kpis', [])
            kpi_count = len(kpis)
            
            table_stat = {
                'table_id': f"page_{page}_table_{table_idx}",
                'page': page,
                'table_index': table_idx,
                'kpi_count': kpi_count,
                'has_error': 'error' in result,
                'error_message': result.get('error', None)
            }
            
            # Add validation stats if available
            if 'validation_stats' in result:
                val_stats = result['validation_stats']
                table_stat.update({
                    'valid_kpis': val_stats.get('valid_kpis', 0),
                    'invalid_kpis': val_stats.get('invalid_kpis', 0),
                    'validation_rate': val_stats.get('validation_rate', 0)
                })
            
            table_stats.append(table_stat)
            if kpi_count > 0:
                kpis_per_table.append(kpi_count)
        
        # Summary statistics
        performance_summary = {
            'tables_analyzed': len(table_stats),
            'average_kpis_per_table': round(sum(kpis_per_table) / len(kpis_per_table), 2) if kpis_per_table else 0,
            'max_kpis_per_table': max(kpis_per_table) if kpis_per_table else 0,
            'min_kpis_per_table': min(kpis_per_table) if kpis_per_table else 0,
            'tables_with_errors': len([t for t in table_stats if t['has_error']]),
            'detailed_table_stats': table_stats
        }
        
        return performance_summary
    
    def analyze_validation_performance(self) -> Dict[str, Any]:
        """Analyze validation performance using validation_statistics from individual results."""
        if not self.summary_data:
            return {}
        
        results = self.summary_data.get('results', [])
        
        # Initialize aggregation variables
        validation_stats = {
            'validation_performed': False,
            'total_validated_tables': 0,
            'total_kpis': 0,
            'total_valid_kpis': 0,
            'total_invalid_kpis': 0,
            'total_duplicate_kpis': 0,
            'total_missing_tables': 0,
            'total_row_name_verified': 0,
            'total_col_name_verified': 0,
            'total_row_name_mismatches': 0,
            'total_col_name_mismatches': 0,
            'overall_accuracy': 0.0,
            'overall_precision': 0.0,
            'average_confidence': 0.0,
            'validation_rates_by_table': []
        }
        
        # Aggregate validation statistics from all results
        total_accuracy_sum = 0.0
        total_precision_sum = 0.0
        total_confidence_sum = 0.0
        validated_tables = 0
        
        for result in results:
            val_stats = result.get('validation_statistics')
            if val_stats:
                validation_stats['validation_performed'] = True
                validated_tables += 1
                
                # Aggregate counts
                validation_stats['total_kpis'] += val_stats.get('total_kpis', 0)
                validation_stats['total_valid_kpis'] += val_stats.get('valid_kpis', 0)
                validation_stats['total_invalid_kpis'] += val_stats.get('invalid_kpis', 0)
                validation_stats['total_duplicate_kpis'] += val_stats.get('duplicate_kpis', 0)
                validation_stats['total_missing_tables'] += val_stats.get('missing_tables', 0)
                validation_stats['total_row_name_verified'] += val_stats.get('row_name_verified', 0)
                validation_stats['total_col_name_verified'] += val_stats.get('col_name_verified', 0)
                validation_stats['total_row_name_mismatches'] += val_stats.get('row_name_mismatches', 0)
                validation_stats['total_col_name_mismatches'] += val_stats.get('col_name_mismatches', 0)
                
                # Aggregate rates for averaging
                accuracy = val_stats.get('accuracy', 0.0)
                precision = val_stats.get('precision', 0.0)
                confidence = val_stats.get('confidence_avg', 0.0)
                
                total_accuracy_sum += accuracy
                total_precision_sum += precision
                total_confidence_sum += confidence
                
                # Store per-table validation rate
                table_info = {
                    'page': result.get('page', 'Unknown'),
                    'table_index': result.get('table_index', 'Unknown'),
                    'total_kpis': val_stats.get('total_kpis', 0),
                    'valid_kpis': val_stats.get('valid_kpis', 0),
                    'invalid_kpis': val_stats.get('invalid_kpis', 0),
                    'accuracy': accuracy,
                    'precision': precision,
                    'confidence': confidence
                }
                validation_stats['validation_rates_by_table'].append(table_info)
        
        if validation_stats['validation_performed'] and validated_tables > 0:
            # Calculate overall rates
            validation_stats['total_validated_tables'] = validated_tables
            validation_stats['overall_accuracy'] = round(total_accuracy_sum / validated_tables, 2)
            validation_stats['overall_precision'] = round(total_precision_sum / validated_tables, 2)
            validation_stats['average_confidence'] = round(total_confidence_sum / validated_tables, 2)
            
            # Calculate overall validation rate based on total KPIs
            total_validated_kpis = validation_stats['total_kpis']
            if total_validated_kpis > 0:
                validation_stats['overall_validation_rate'] = round(
                    validation_stats['total_valid_kpis'] / total_validated_kpis * 100, 2
                )
            else:
                validation_stats['overall_validation_rate'] = 0.0
            
            # Calculate verification rates
            if validation_stats['total_kpis'] > 0:
                validation_stats['row_name_verification_rate'] = round(
                    validation_stats['total_row_name_verified'] / validation_stats['total_kpis'] * 100, 2
                )
                validation_stats['col_name_verification_rate'] = round(
                    validation_stats['total_col_name_verified'] / validation_stats['total_kpis'] * 100, 2
                )
            else:
                validation_stats['row_name_verification_rate'] = 0.0
                validation_stats['col_name_verification_rate'] = 0.0
        
        return validation_stats
    
    def generate_comprehensive_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics from the extraction summary."""
        if not self.summary_data:
            logger.error("No summary data loaded")
            return {}
        
        logger.info("Analyzing extraction statistics...")
        
        # Generate all analysis sections
        extraction_performance = self.analyze_extraction_performance()
        kpi_distribution = self.analyze_kpi_distribution()
        table_performance = self.analyze_table_performance()
        validation_performance = self.analyze_validation_performance()
        
        # Meta information
        meta_info = {
            'analysis_date': datetime.now().isoformat(),
            'extraction_date': self.summary_data.get('extraction_date'),
            'model_used': self.summary_data.get('model'),
            'source_file': self.summary_data.get('tables_json_path'),
            'context': self.summary_data.get('context', 'Unknown')
        }
        
        comprehensive_stats = {
            'meta_information': meta_info,
            'extraction_performance': extraction_performance,
            'kpi_distribution': kpi_distribution,
            'table_performance': table_performance,
            'validation_performance': validation_performance
        }
        
        self.stats = comprehensive_stats
        return comprehensive_stats
    
    def print_summary(self):
        """Print a human-readable summary of the statistics."""
        if not self.stats:
            logger.error("No statistics generated. Run generate_comprehensive_stats() first.")
            return
        
        print("\n" + "=" * 80)
        print("EXTRACTION STATISTICS SUMMARY")
        print("=" * 80)
        
        # Meta information
        meta = self.stats['meta_information']
        print(f"\nExtraction Details:")
        print(f"  Model: {meta['model_used']}")
        print(f"  Context: {meta['context']}")
        print(f"  Extraction Date: {meta['extraction_date']}")
        print(f"  Analysis Date: {meta['analysis_date']}")
        
        # Performance overview
        perf = self.stats['extraction_performance']
        print(f"\nExtraction Performance:")
        print(f"  Total Tables: {perf['total_tables_processed']}")
        print(f"  Successful Extractions: {perf['successful_extractions']} ({perf['success_rate_percent']}%)")
        print(f"  Tables with KPIs: {perf['tables_with_kpis']} ({perf['kpi_yield_rate_percent']}%)")
        print(f"  Failed Extractions: {perf['failed_extractions']}")
        
        # KPI overview
        kpi = self.stats['kpi_distribution']
        print(f"\nKPI Distribution:")
        print(f"  Total KPIs: {kpi['total_kpis']}")
        print(f"  Unique Metric Names: {kpi['unique_names']}")
        print(f"  Unique Entity Keys: {kpi['unique_keys']}")
        print(f"  Unique Countries: {kpi['unique_countries']}")
        print(f"  Unique Units: {kpi['unique_units']}")
        print(f"  Years Covered: {kpi['unique_years']}")
        
        # Value statistics
        if kpi['value_statistics']:
            val_stats = kpi['value_statistics']
            print(f"\nValue Statistics:")
            print(f"  Values with Data: {val_stats['count']}")
            print(f"  Null Values: {val_stats['null_count']} ({val_stats['null_percentage']}%)")
            print(f"  Range: {val_stats['min']:.2f} to {val_stats['max']:.2f}")
            print(f"  Average: {val_stats['mean']:.2f}")
        
        # Table performance
        table_perf = self.stats['table_performance']
        print(f"\nTable Performance:")
        print(f"  Average KPIs per Table: {table_perf['average_kpis_per_table']}")
        print(f"  Max KPIs in Single Table: {table_perf['max_kpis_per_table']}")
        print(f"  Tables with Errors: {table_perf['tables_with_errors']}")
        
        # Validation performance
        val_perf = self.stats['validation_performance']
        if val_perf['validation_performed']:
            print(f"\nValidation Performance:")
            print(f"  Validated Tables: {val_perf['total_validated_tables']}")
            print(f"  Total KPIs Validated: {val_perf['total_kpis']}")
            print(f"  Valid KPIs: {val_perf['total_valid_kpis']} ({val_perf['overall_validation_rate']}%)")
            print(f"  Invalid KPIs: {val_perf['total_invalid_kpis']}")
            print(f"  Overall Accuracy: {val_perf['overall_accuracy']}%")
            print(f"  Overall Precision: {val_perf['overall_precision']}%")
            print(f"  Average Confidence: {val_perf['average_confidence']}%")
            print(f"  Duplicate KPIs Found: {val_perf['total_duplicate_kpis']}")
            print(f"  Row Name Verification: {val_perf['row_name_verification_rate']}%")
            print(f"  Column Name Verification: {val_perf['col_name_verification_rate']}%")
        else:
            print(f"\nValidation: Not performed")
        
        # Top metrics
        print(f"\nTop Metric Names:")
        for name, count in list(kpi['top_names'].items())[:5]:
            print(f"  {name}: {count}")
        
        print(f"\nTop Entity Keys:")
        for key, count in list(kpi['top_keys'].items())[:5]:
            print(f"  {key}: {count}")
        
        print(f"\nTop Countries:")
        for country, count in list(kpi['top_countries'].items())[:5]:
            print(f"  {country}: {count}")
        
        print("\n" + "=" * 80)
    
    def save_stats(self, output_path: str) -> bool:
        """Save comprehensive statistics to JSON file."""
        if not self.stats:
            logger.error("No statistics generated. Run generate_comprehensive_stats() first.")
            return False
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Statistics saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save statistics: {str(e)}")
            return False


def main():
    """Main entry point for the statistics analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze extraction summary files and generate comprehensive statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single summary file
  python analyze_extraction_stats.py data/output/trial-19/extraction_summary.json
  
  # Analyze and save detailed stats
  python analyze_extraction_stats.py summary.json --output-json stats.json
  
  # Just print summary without saving
  python analyze_extraction_stats.py summary.json --no-save
        """
    )
    
    parser.add_argument(
        'summary_file',
        help='Path to extraction_summary.json file'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        help='Output path for detailed statistics JSON file'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Only print summary, do not save files'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExtractionStatsAnalyzer()
    
    # Load summary
    if not analyzer.load_summary(args.summary_file):
        return 1
    
    # Generate statistics
    logger.info("Generating comprehensive statistics...")
    stats = analyzer.generate_comprehensive_stats()
    
    if not stats:
        logger.error("Failed to generate statistics")
        return 1
    
    # Print summary
    analyzer.print_summary()
    
    # Save outputs if requested
    if not args.no_save:
        summary_path = Path(args.summary_file)
        
        # Save JSON
        json_output = args.output_json or (summary_path.parent / "extraction_statistics.json")
        analyzer.save_stats(str(json_output))
    
    logger.info("Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())