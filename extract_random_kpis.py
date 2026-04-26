#!/usr/bin/env python3
"""
Random Table Extractor

This script extracts random full tables (with all their KPIs) from JSON files
in a given folder, ensuring at least one table from each year is included.

Usage:
    python extract_random_kpis.py --folder trial-20 --num-tables 15 --output random_tables.json
"""

import argparse
import csv
import json
import html
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_kpi_file(file_path: Path) -> Optional[Dict]:
    """Load and parse a KPI JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename (e.g., divisions-vw-ar20.pdf -> 2020)."""
    # Look for patterns like ar20, ar2020, 2020, etc.
    patterns = [
        r'ar(\d{2})',      # ar20 -> 20
        r'ar(\d{4})',      # ar2020 -> 2020
        r'_(\d{4})',       # _2020
        r'-(\d{4})',       # -2020
        r'(\d{4})'         # 2020
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year_str = match.group(1)
            year = int(year_str)
            # Convert 2-digit to 4-digit year
            if year < 100:
                year = 2000 + year
            # Sanity check: year should be reasonable
            if 2000 <= year <= 2030:
                return year
    
    return None


def extract_full_table(data: Dict, source_file: str, document_name: str,
                       from_validation: bool = False,
                       filter_null_kpis: bool = False) -> Optional[Dict]:
    """Extract full table with all KPIs and metadata.

    Args:
        from_validation: If True, read KPIs from ``valid_kpis[i]['kpi']``
            instead of the top-level ``kpis`` list (validation output format).
        filter_null_kpis: If True, discard any KPI whose ``year`` or ``value``
            field is None/null before returning.
    """
    if from_validation:
        raw_entries = data.get('valid_kpis', [])
        raw_kpis = [entry['kpi'] for entry in raw_entries if isinstance(entry.get('kpi'), dict)]
    else:
        raw_kpis = data.get('kpis', [])

    if not raw_kpis:
        return None

    # Add consistent source field to each KPI
    kpis = []
    for kpi in raw_kpis:
        if filter_null_kpis:
            if kpi.get('year') is None or kpi.get('value') is None:
                continue
        kpi_copy = kpi.copy()
        # Remove source_table field if it exists and replace with source
        if 'source_table' in kpi_copy:
            del kpi_copy['source_table']
        kpi_copy['source'] = f"{document_name}/{source_file}"
        kpis.append(kpi_copy)

    if not kpis:
        return None

    # Extract year from filename if available
    year = extract_year_from_filename(source_file)

    return {
        'source_file': source_file,
        'document_name': document_name,
        'year': year,
        'num_kpis': len(kpis),
        'kpis': kpis
    }


def flatten_dict(data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary using dot notation for keys."""
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def build_headers(rows: List[Dict[str, Any]]) -> List[str]:
    """Build CSV headers preserving first-seen key order across rows."""
    headers: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers


def write_table_csv(table: Dict[str, Any], csv_output_path: Path, csv_encoding: str = 'utf-8-sig') -> int:
    """Write one table's KPI list into a CSV file with all keys as headers."""
    kpis = table.get('kpis', [])
    flattened_rows = [flatten_dict(kpi) for kpi in kpis]

    if not flattened_rows:
        # Still create an empty file with no headers for traceability.
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_output_path.write_text('', encoding=csv_encoding)
        return 0

    headers = build_headers(flattened_rows)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_output_path, 'w', newline='', encoding=csv_encoding) as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(flattened_rows)

    return len(flattened_rows)


def resolve_table_image_path(table: Dict[str, Any], images_root: Optional[str] = None) -> Optional[Path]:
    """Resolve the extracted table image path for a selected table record."""
    image_path = table.get('image_path')

    if image_path:
        image_path_obj = Path(image_path)
        if image_path_obj.exists():
            return image_path_obj

        if images_root and "detected_tables_test/" in image_path:
            rel = image_path.split("detected_tables_test/", 1)[1].replace("\\", "/")
            remapped = Path(images_root) / Path(rel)
            if remapped.exists():
                return remapped

    source_file = table.get("source_file", "")
    match = re.match(r"(.+)_page_(\d{3})_table_(\d{2})_kpis\.json$", source_file)
    if match and images_root:
        doc_id, page_idx, table_idx = match.groups()
        candidate = Path(images_root) / doc_id / f"page_{page_idx}" / f"table_{table_idx}.png"
        if candidate.exists():
            return candidate

    return None


def render_kpi_table_html(table: Dict[str, Any]) -> str:
    """Render KPIs of one table as an HTML table."""
    kpis = table.get('kpis', [])
    rows = [flatten_dict(kpi) for kpi in kpis]
    if not rows:
        return "<p>No KPI rows found.</p>"

    headers = build_headers(rows)

    thead = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body_rows.append(f"<tr>{tds}</tr>")

    return (
        "<div class=\"table-wrap\">"
        "<table>"
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def write_tables_html(
        selected_tables: List[Dict[str, Any]],
        html_output_path: Path,
        images_root: Optional[str] = None,
) -> Dict[str, Any]:
        """Write one HTML file per selected table containing image + KPI HTML table."""
        html_output_path.parent.mkdir(parents=True, exist_ok=True)

        if html_output_path.suffix.lower() == ".html":
                html_dir = html_output_path.parent / f"{html_output_path.stem}_tables"
                index_file = html_output_path
        else:
                html_dir = html_output_path
                index_file = html_dir / "index.html"

        html_dir.mkdir(parents=True, exist_ok=True)

        index_links = []
        images_found = 0

        for idx, table in enumerate(selected_tables, start=1):
                source_file = table.get("source_file", f"table_{idx}")
                source_stem = Path(source_file).stem
                file_name = f"{idx:03d}_{source_stem}.html"
                file_path = html_dir / file_name

                image_path = resolve_table_image_path(table, images_root=images_root)
                image_block = "<p class=\"missing\">Image not found. Provide --images-root.</p>"
                if image_path:
                        images_found += 1
                        rel_image = os.path.relpath(image_path, html_dir).replace("\\", "/")
                        image_block = (
                                f"<img src=\"{html.escape(rel_image)}\" alt=\"{html.escape(source_file)}\" loading=\"lazy\" />"
                        )

                table_html = render_kpi_table_html(table)

                page = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{html.escape(source_file)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f8; }}
        .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
        h1 {{ font-size: 18px; margin: 0 0 12px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 12px; }}
        .missing {{ color: #a94442; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }}
        th {{ background: #f2f2f2; }}
        .meta {{ margin: 8px 0 14px 0; color: #555; font-size: 13px; }}
    </style>
</head>
<body>
    <div class=\"card\">
        <h1>{idx:03d} - {html.escape(source_file)}</h1>
        <div class=\"meta\">Year: {html.escape(str(table.get('year')))} | KPIs: {html.escape(str(table.get('num_kpis')))}</div>
        {image_block}
        {table_html}
    </div>
</body>
</html>
""".strip()

                file_path.write_text(page, encoding="utf-8")
                index_links.append(f"<li><a href=\"{html.escape(file_name)}\">{html.escape(file_name)}</a></li>")

        index_page = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Extracted Tables - Index</title>
</head>
<body>
    <h1>Extracted Tables</h1>
    <p>Total selected tables: {len(selected_tables)} | Images resolved: {images_found}</p>
    <ul>
        {''.join(index_links)}
    </ul>
</body>
</html>
""".strip()
        index_file.write_text(index_page, encoding="utf-8")

        return {
                "html_output": str(index_file),
                "html_output_dir": str(html_dir),
                "html_files_count": len(selected_tables),
                "images_found": images_found,
                "images_missing": len(selected_tables) - images_found,
        }


def get_kpi_files(folder_path: Path, recursive: bool = True, bucket: Optional[str] = None) -> List[Path]:
    """Get all KPI JSON files from the specified folder.
    
    Args:
        bucket: Filter by document type. 'management' keeps only files whose
                name contains 'management'. 'divisions' keeps only files whose
                name contains 'division'. None / 'all' returns everything.
    """
    pattern = "*_kpis.json"
    if recursive:
        kpi_files = list(folder_path.rglob(pattern))
    else:
        kpi_files = list(folder_path.glob(pattern))
    
    if bucket and bucket != 'all':
        kpi_files = [f for f in kpi_files if bucket.lower() in f.name.lower()]
    
    kpi_files.sort()  # Sort for consistent ordering
    return kpi_files


def extract_random_tables_from_folder(
    folder_path: str, 
    num_tables: int, 
    output_file: str,
    csv_output_dir: Optional[str] = None,
    csv_encoding: str = 'utf-8-sig',
    html_output_file: Optional[str] = None,
    images_root: Optional[str] = None,
    document_name: Optional[str] = None,
    seed: Optional[int] = None,
    recursive: bool = True,
    bucket: Optional[str] = None,
    require_gt: bool = True,
    from_validation: bool = False,
    filter_null_kpis: bool = False,
    require_full_valid: bool = False,
) -> Dict:
    """
    Extract random full tables from JSON files in a folder.
    Ensures at least one table from each year is included.
    
    Args:
        folder_path: Path to folder containing KPI JSON files
        num_tables: Number of tables to extract (minimum 15)
        output_file: Path for output JSON file
        document_name: Name of source document (auto-detected if None)
        seed: Random seed for reproducible results
        require_gt: If True (default), skip tables where all KPIs have no
                    matching ground-truth table (missing_tables == total_kpis).
        from_validation: If True, read KPIs from the ``valid_kpis`` list of
                         validation-output files (``valid_kpis[i]['kpi']``)
                         instead of the raw ``kpis`` list.  Tables with 0
                         valid KPIs after null-filtering are excluded.
        filter_null_kpis: If True, discard any KPI whose ``year`` or ``value``
                          is None before adding it to the selected table.
        require_full_valid: If True, only include tables where every KPI is
                            valid (valid_kpis == total_kpis, i.e. 100% accuracy).
        
    Returns:
        Dictionary containing the extracted tables and metadata
    """
    if seed is not None:
        random.seed(seed)
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found or not a directory: {folder_path}")
    
    # Auto-detect document name if not provided
    if document_name is None:
        document_name = extract_document_name_from_folder(folder)
    
    # Get all KPI files
    kpi_files = get_kpi_files(folder, recursive=recursive, bucket=bucket)
    if not kpi_files:
        bucket_label = bucket if bucket and bucket != 'all' else 'any'
        raise ValueError(f"No KPI files found in {folder_path} (bucket filter: {bucket_label})")
    
    bucket_label = f" [{bucket}]" if bucket and bucket != 'all' else ""
    print(f"Found {len(kpi_files)} KPI files in {folder_path}{bucket_label}")
    
    # Load all tables with metadata
    all_tables = []
    tables_by_year = {}
    
    skipped_no_gt = 0
    skipped_partial = 0
    for kpi_file in kpi_files:
        data = load_kpi_file(kpi_file)
        if data is None:
            continue

        # Skip tables with no ground-truth match in the database.
        # For validation output files use statistics.missing_tables;
        # for raw extraction files use validation_statistics.missing_tables.
        if require_gt:
            if from_validation:
                stats = data.get('statistics', {})
                total   = stats.get('total_kpis', 0)
                missing = stats.get('missing_tables', 0)
                valid_n = stats.get('valid_kpis', 0)
                if total > 0 and (missing >= total or valid_n == 0):
                    skipped_no_gt += 1
                    continue
            else:
                val_stats = data.get('validation_statistics', {})
                total = val_stats.get('total_kpis', 0)
                missing = val_stats.get('missing_tables', 0)
                if total > 0 and missing >= total:
                    skipped_no_gt += 1
                    continue

        # Skip tables that are not 100% valid (when require_full_valid is set).
        if require_full_valid and from_validation:
            stats = data.get('statistics', {})
            total   = stats.get('total_kpis', 0)
            valid_n = stats.get('valid_kpis', 0)
            if total > 0 and valid_n < total:
                skipped_partial += 1
                continue

        table = extract_full_table(data, kpi_file.name, document_name,
                                   from_validation=from_validation,
                                   filter_null_kpis=filter_null_kpis)
        if table and table['num_kpis'] > 0:
            all_tables.append(table)
            
            # Group by year
            year = table['year']
            if year:
                if year not in tables_by_year:
                    tables_by_year[year] = []
                tables_by_year[year].append(table)

    if require_gt and skipped_no_gt:
        print(f"Skipped {skipped_no_gt} tables with no ground-truth table match")
    if require_full_valid and skipped_partial:
        print(f"Skipped {skipped_partial} tables with partial validation (not 100% valid)")

    if not all_tables:
        raise ValueError(f"No valid tables found in {folder_path}")
    
    print(f"Loaded {len(all_tables)} tables with KPIs")
    print(f"Years found: {sorted(tables_by_year.keys())}")
    
    # Select tables: at least one from each year, then random
    selected_tables = []
    remaining_tables = all_tables.copy()
    
    # First, select one table from each year
    for year in sorted(tables_by_year.keys()):
        year_tables = tables_by_year[year]
        selected = random.choice(year_tables)
        selected_tables.append(selected)
        remaining_tables.remove(selected)
        print(f"  Selected 1 table from year {year}")
    
    # Calculate how many more tables we need
    tables_needed = num_tables - len(selected_tables)
    
    if tables_needed > 0 and remaining_tables:
        # Randomly select additional tables
        additional = random.sample(
            remaining_tables, 
            min(tables_needed, len(remaining_tables))
        )
        selected_tables.extend(additional)
        print(f"  Selected {len(additional)} additional random tables")
    
    # Flatten all KPIs from selected tables
    all_kpis = []
    for table in selected_tables:
        all_kpis.extend(table['kpis'])
    
    # Create output structure
    result = {
        "source_document": document_name,
        "extraction_folder": folder.name,
        "extraction_date": datetime.now().strftime("%Y-%m-%d"),
        "description": f"Random sample of {len(selected_tables)} full tables from {len(kpi_files)} files",
        "total_tables_available": len(all_tables),
        "total_tables_selected": len(selected_tables),
        "num_tables_requested": num_tables,
        "tables_by_year": {year: len(tables) for year, tables in tables_by_year.items()},
        "total_kpis": len(all_kpis),
        "random_seed": seed,
        "selected_tables": [
            {
                "source_file": t['source_file'],
                "year": t['year'],
                "num_kpis": t['num_kpis']
            } for t in selected_tables
        ],
        "kpis": all_kpis
    }

    # Save one CSV per selected table
    if csv_output_dir:
        csv_dir_path = Path(csv_output_dir)
        csv_dir_path.mkdir(parents=True, exist_ok=True)
        csv_outputs = []

        for idx, table in enumerate(selected_tables, start=1):
            source_stem = Path(table['source_file']).stem
            csv_name = f"{idx:03d}_{source_stem}.csv"
            csv_path = csv_dir_path / csv_name
            row_count = write_table_csv(table, csv_path, csv_encoding=csv_encoding)

            csv_outputs.append(
                {
                    "source_file": table['source_file'],
                    "csv_file": str(csv_path),
                    "rows_written": row_count,
                }
            )

        result["csv_output_dir"] = str(csv_dir_path)
        result["csv_encoding"] = csv_encoding
        result["csv_files"] = csv_outputs

    # Save optional HTML preview with only extracted table images
    if html_output_file:
        html_meta = write_tables_html(
            selected_tables=selected_tables,
            html_output_path=Path(html_output_file),
            images_root=images_root,
        )
        result.update(html_meta)
    
    # Save to output file
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_kpis)} KPIs from {len(selected_tables)} tables to {output_file}")

    if csv_output_dir:
        print(f"Saved {len(selected_tables)} table CSV files to {csv_output_dir}")

    if html_output_file:
        print(
            f"Saved {result.get('html_files_count', 0)} per-table HTML files to "
            f"{result.get('html_output_dir', html_output_file)}"
        )
    
    print(f"\nExtraction Summary:")
    print(f"  Tables selected: {len(selected_tables)}")
    print(f"  Years covered: {len(tables_by_year)}")
    print(f"  Total KPIs extracted: {len(all_kpis)}")
    print(f"  Average KPIs per table: {len(all_kpis) / len(selected_tables):.1f}")
    
    return result


def extract_document_name_from_folder(folder_path: Path) -> str:
    """Extract document name from folder structure or use folder name."""
    folder_name = folder_path.name
    
    # Try to infer document name from common patterns
    if 'trial-' in folder_name:
        # Look for parent folder or use a default pattern
        parent = folder_path.parent.name
        if parent and parent != 'output':
            return parent
        return "divisions-vw-ar23"  # Default based on the example
    
    return folder_name


def main():
    parser = argparse.ArgumentParser(
        description="Extract random full tables from JSON files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_random_kpis.py --folder data/output/trial-20 --num-tables 15 --output random_tables.json
  python extract_random_kpis.py --folder trial-20 --num-tables 20 --document "divisions-vw-ar23" --seed 42
        """
    )
    
    parser.add_argument(
        '--folder', '-f',
        required=True,
        help='Path to folder containing KPI JSON files'
    )
    
    parser.add_argument(
        '--num-tables', '-n',
        type=int,
        default=15,
        help='Number of tables to extract (default: 15)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='random_tables_output.json',
        help='Output JSON file path (default: random_tables_output.json)'
    )

    parser.add_argument(
        '--csv-dir',
        default='random_tables_csv',
        help='Directory to save one CSV per selected table (default: random_tables_csv)'
    )

    parser.add_argument(
        '--csv-encoding',
        default='utf-8-sig',
        help='Encoding for CSV export (default: utf-8-sig; recommended for Windows/Excel)'
    )

    parser.add_argument(
        '--html-output',
        help='Optional HTML output path or directory. Creates one HTML file per selected table (image + KPI table) plus an index page.'
    )

    parser.add_argument(
        '--images-root',
        help='Optional local root of extracted table images (e.g., data/detected_tables_test) for resolving image paths'
    )
    
    parser.add_argument(
        '--document', '-d',
        help='Source document name (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        help='Random seed for reproducible results'
    )

    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Search only the top-level folder for *_kpis.json files (default: recursive search)'
    )

    parser.add_argument(
        '--bucket', '-b',
        choices=['management', 'divisions', 'all'],
        default='all',
        help='Filter files by document type: management, divisions, or all (default: all)'
    )

    parser.add_argument(
        '--no-require-gt',
        action='store_true',
        help='Include tables even if they have no ground-truth table match in validation (default: such tables are skipped)'
    )

    parser.add_argument(
        '--from-validation',
        action='store_true',
        help='Read KPIs from validation output files (valid_kpis[i][\'kpi\']) instead of raw extraction kpis list'
    )

    parser.add_argument(
        '--filter-null-kpis',
        action='store_true',
        help='Discard KPIs whose year or value is null/None before including them in the output'
    )

    parser.add_argument(
        '--require-full-valid',
        action='store_true',
        help='Only include tables where every KPI is valid (100%% accuracy). Requires --from-validation.'
    )
    
    args = parser.parse_args()
    
    try:
        result = extract_random_tables_from_folder(
            folder_path=args.folder,
            num_tables=args.num_tables,
            output_file=args.output,
            csv_output_dir=args.csv_dir,
            csv_encoding=args.csv_encoding,
            html_output_file=args.html_output,
            images_root=args.images_root,
            document_name=args.document,
            seed=args.seed,
            recursive=not args.no_recursive,
            bucket=args.bucket,
            require_gt=not args.no_require_gt,
            from_validation=args.from_validation,
            filter_null_kpis=args.filter_null_kpis,
            require_full_valid=args.require_full_valid,
        )
        
        print(f"\n✓ Successfully extracted {result['total_tables_selected']} tables!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())