"""
Database loader utilities for KPI extraction.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from logger import logger


def load_tables_from_db(
    db_path: str,
    year_filter: Optional[str] = None,
    max_tables: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load tables directly from the database.
    
    Args:
        db_path: Path to the SQLite database
        year_filter: Optional year to filter tables (e.g., "2019")
        max_tables: Maximum number of tables to load
        
    Returns:
        List of table dictionaries
    """
    logger.info(f"Loading tables from database: {db_path}")
    if year_filter:
        logger.info(f"  Year filter: {year_filter}")
    
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT table_id, section_name, title, headers, merged_headers, rows, stub_col FROM context_packs")
        
        tables = []
        for row in cur.fetchall():
            table_id, section_name, title, headers, merged_headers, rows, stub_col = row
            
            # Apply year filter if specified
            if year_filter:
                # Extract year from table_id (e.g., VW2019_T4e9153 -> 2019)
                parts = table_id.split('_')
                if parts:
                    prefix = parts[0]
                    if len(prefix) >= 6:
                        table_year = prefix[2:6]
                        if table_year != year_filter:
                            continue
            
            table_data = {
                "table_id": table_id,
                "section_name": section_name if section_name else "",
                "title": title if title else "",
                "headers": json.loads(headers) if headers else [],
                "merged_headers": json.loads(merged_headers) if merged_headers else None,
                "rows": json.loads(rows) if rows else [],
                "stub_col": json.loads(stub_col) if stub_col else None,
            }
            tables.append(table_data)
            
            # Check if we've hit the limit
            if max_tables and len(tables) >= max_tables:
                break
        
        logger.info(f"  Loaded {len(tables)} tables from database")
        return tables
        
    finally:
        conn.close()


def get_years_from_db(db_path: str) -> List[str]:
    """
    Get all unique years available in the database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        List of year strings (e.g., ["2015", "2016", "2019"])
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT table_id FROM context_packs")
        
        years = set()
        for row in cur.fetchall():
            table_id = row[0]
            parts = table_id.split('_')
            if parts:
                prefix = parts[0]
                if len(prefix) >= 6 and prefix[:2].isalpha():
                    year = prefix[2:6]
                    if year.isdigit():
                        years.add(year)
        
        return sorted(list(years))
        
    finally:
        conn.close()


def get_table_count_by_year(db_path: str) -> Dict[str, int]:
    """
    Get count of tables per year in the database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dictionary mapping year to table count
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT table_id FROM context_packs")
        
        year_counts = {}
        for row in cur.fetchall():
            table_id = row[0]
            parts = table_id.split('_')
            if parts:
                prefix = parts[0]
                if len(prefix) >= 6 and prefix[:2].isalpha():
                    year = prefix[2:6]
                    if year.isdigit():
                        year_counts[year] = year_counts.get(year, 0) + 1
        
        return year_counts
        
    finally:
        conn.close()


def load_existing_results(output_file: Path) -> Tuple[List[Dict[str, Any]], set]:
    """
    Load existing results from checkpoint file.
    
    Args:
        output_file: Path to output JSON file
        
    Returns:
        Tuple of (existing_results_list, set_of_processed_table_ids)
    """
    if not output_file.exists():
        return [], set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        if isinstance(existing_data, list):
            results = existing_data
        elif isinstance(existing_data, dict) and 'results' in existing_data:
            results = existing_data['results']
        else:
            logger.warning(f"Unexpected format in {output_file}, starting fresh")
            return [], set()
        
        processed_ids = {r.get('table_id') for r in results if r.get('table_id')}
        logger.info(f"  Found {len(processed_ids)} already processed tables in checkpoint file")
        return results, processed_ids
        
    except Exception as e:
        logger.warning(f"Error loading checkpoint file: {e}, starting fresh")
        return [], set()


def save_checkpoint(output_file: Path, results: List[Dict[str, Any]], 
                   model_name: str, year_filter: Optional[str]) -> None:
    """
    Save current results to checkpoint file.
    
    Args:
        output_file: Path to output JSON file
        results: List of results to save
        model_name: Model name for metadata
        year_filter: Year filter for metadata
    """
    checkpoint_data = {
        "metadata": {
            "model": model_name,
            "year_filter": year_filter,
            "last_updated": datetime.now().isoformat(),
            "total_tables_processed": len(results),
            "checkpoint": True
        },
        "results": results
    }
    
    # Write to temporary file first, then rename (atomic operation)
    temp_file = output_file.with_suffix('.tmp')
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    # Atomic rename
    temp_file.replace(output_file)
