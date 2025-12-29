from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_all_tables_from_db(db_path: str) -> List[Dict[str, Any]]:
    """Load all tables from the database."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT table_id, section_name, title, headers, merged_headers, rows, stub_col, paragraphs FROM context_packs")
        
        tables = []
        for row in cur.fetchall():
            table_id, section_name, title, headers, merged_headers, rows, stub_col, paragraphs = row
            
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
        
        return tables
    finally:
        conn.close()


def extract_year_from_table_id(table_id: str) -> str | None:
    """Extract year from table_id (e.g., VW2019_T4e9153 -> 2019)."""
    parts = table_id.split('_')
    if len(parts) >= 1:
        # Extract year from format like VW2019
        prefix = parts[0]
        if len(prefix) >= 6 and prefix[:2].isalpha():
            year = prefix[2:]
            if year.isdigit() and len(year) == 4:
                return year
    return None


def group_tables_by_year(tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group tables by year extracted from table_id."""
    grouped = defaultdict(list)
    
    for table in tables:
        table_id = table.get("table_id", "")
        year = extract_year_from_table_id(table_id)
        
        if year:
            grouped[year].append(table)
        else:
            # If year cannot be extracted, put in 'unknown' category
            grouped["unknown"].append(table)
    
    return dict(grouped)


def export_tables_to_json_files(db_path: str, output_dir: str) -> None:
    """Export tables from database to JSON files grouped by year."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading tables from database...")
    tables = load_all_tables_from_db(db_path)
    print(f"Loaded {len(tables)} tables")
    
    print("\nGrouping tables by year...")
    grouped_tables = group_tables_by_year(tables)
    
    print(f"Found {len(grouped_tables)} year groups")
    
    for year, year_tables in sorted(grouped_tables.items()):
        output_file = output_path / f"tables_{year}.json"
        
        print(f"\nExporting {len(year_tables)} tables for year {year} to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(year_tables, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved to {output_file}")
        
        # Show sample table IDs
        sample_ids = [t['table_id'] for t in year_tables[:5]]
        print(f"  Sample table IDs: {', '.join(sample_ids)}")
        if len(year_tables) > 5:
            print(f"  ... and {len(year_tables) - 5} more tables")
    
    print(f"\n{'='*80}")
    print("Export complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    DB_PATH = "data/pack_context.db"
    OUTPUT_DIR = "data/output/exported_tables"
    
    try:
        export_tables_to_json_files(DB_PATH, OUTPUT_DIR)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
