import argparse
import json
import re
from pathlib import Path


def parse_kpi_line(line: str) -> dict:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 6:
        return {"raw_line": line}

    def extract_value(token: str, key: str) -> str:
        prefix = f"{key}="
        return token[len(prefix) :].strip() if token.startswith(prefix) else token

    return {
        "name": parts[0],
        "key": parts[1],
        "country": parts[2],
        "value": extract_value(parts[3], "value"),
        "year": extract_value(parts[4], "year"),
        "units": extract_value(parts[5], "units"),
        "raw_line": line,
    }


def parse_row_checks(row_checks):
    checks = {}
    for item in row_checks or []:
        row_index = item.get("rowIndex")
        if row_index is None:
            continue

        checked = item.get("value")
        if checked is None:
            checked = item.get("checked", False)

        checks[int(row_index)] = bool(checked)
    return checks


def split_output_lines(record: dict):
    output = record.get("output", "")
    return [line.strip() for line in str(output).splitlines() if line.strip()]


def is_percent_row(row: dict) -> bool:
    units = str(row.get("units", "")).strip()
    return units == "%" or "%" in units


def extract_checked_unchecked(record: dict, filter_unchecked_percent: bool = False):
    lines = split_output_lines(record)
    checks = parse_row_checks(record.get("row_checks", []))

    base = {
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "setup_id": record.get("setup_id"),
        "example_idx": record.get("example_idx"),
        "annotator_id": (record.get("metadata") or {}).get("annotator_id"),
    }

    checked_rows = []
    not_checked_rows = []

    for idx, line in enumerate(lines):
        row = {
            **base,
            "line_index": idx,
            **parse_kpi_line(line),
        }

        if checks.get(idx, False):
            checked_rows.append(row)
        else:
            not_checked_rows.append(row)

    if filter_unchecked_percent:
        not_checked_rows = [row for row in not_checked_rows if not is_percent_row(row)]

    return checked_rows, not_checked_rows


def extract_manual_annotations(record: dict):
    spans = record.get("annotations") or []
    text_fields = record.get("text_fields") or []

    non_empty_text_fields = []
    for item in text_fields:
        value = str(item.get("value", "")).strip()
        if value:
            non_empty_text_fields.append(
                {
                    "label": item.get("label"),
                    "value": value,
                }
            )

    return spans, non_empty_text_fields


def load_jsonl_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace(": NaN", ": null")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def split_sort_key(split_name: str):
    match = re.search(r"(\d+)$", str(split_name))
    return int(match.group(1)) if match else 10**9


def parse_source_file_metadata(source_file: str) -> dict:
    page_match = re.search(r"_page_(\d+)", source_file)
    table_match = re.search(r"_table_(\d+)", source_file)
    report_match = re.match(r"^(.*?)_page_\d+_table_\d+_kpis\.json$", source_file)

    return {
        "source_file": source_file,
        "report_id": report_match.group(1) if report_match else None,
        "page": int(page_match.group(1)) if page_match else None,
        "table_index": int(table_match.group(1)) if table_match else None,
    }


def build_table_source_mapping(records: list, selected_tables_path: Path | None):
    if not selected_tables_path or not selected_tables_path.exists():
        return {}

    try:
        payload = json.loads(selected_tables_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    selected_tables = payload.get("selected_tables") or []
    if not selected_tables:
        return {}

    split_names = sorted({(r.get("split") or "unknown") for r in records}, key=split_sort_key)
    mapping = {}

    for idx, split_name in enumerate(split_names):
        if idx >= len(selected_tables):
            break
        item = selected_tables[idx]
        source_file = item.get("source_file")
        if not source_file:
            continue

        source_meta = parse_source_file_metadata(source_file)
        source_meta["year"] = item.get("year")
        source_meta["num_kpis"] = item.get("num_kpis")
        mapping[split_name] = source_meta

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Extract checked and not-checked KPI rows from Factgenie annotation JSONL"
    )
    parser.add_argument("--input", required=True, help="Path to annotation JSONL file")
    parser.add_argument("--output", default="checked_unchecked_kpis.json", help="Output JSON file path")
    parser.add_argument(
        "--filter-unchecked-percent",
        action="store_true",
        help="Exclude unchecked rows whose units contain %%",
    )
    parser.add_argument(
        "--selected-tables-json",
        default="random_tables_trial24.json",
        help="Path to random_tables_*.json used to map split to source file/page/table",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_jsonl_records(input_path)

    checked_kpis = []
    not_checked_kpis = []
    by_table = {}
    tables_with_annotations = {}
    all_span_annotations = []
    table_source_mapping = build_table_source_mapping(records, Path(args.selected_tables_json))

    for record in records:
        checked_rows, not_checked_rows = extract_checked_unchecked(
            record, filter_unchecked_percent=args.filter_unchecked_percent
        )
        spans, non_empty_text_fields = extract_manual_annotations(record)
        checked_kpis.extend(checked_rows)
        not_checked_kpis.extend(not_checked_rows)

        split = record.get("split") or "unknown"
        table_entry = {
            "dataset": record.get("dataset"),
            "split": split,
            "source_table": table_source_mapping.get(split),
            "setup_id": record.get("setup_id"),
            "example_idx": record.get("example_idx"),
            "annotator_id": (record.get("metadata") or {}).get("annotator_id"),
            "annotations": record.get("annotations", []),
            "text_fields": record.get("text_fields", []),
            "checked_count": len(checked_rows),
            "not_checked_count": len(not_checked_rows),
            "checked_rows": checked_rows,
            "not_checked_rows": not_checked_rows,
        }
        by_table[split] = table_entry

        if spans or non_empty_text_fields:
            tables_with_annotations[split] = {
                "dataset": record.get("dataset"),
                "split": split,
                "setup_id": record.get("setup_id"),
                "example_idx": record.get("example_idx"),
                "annotator_id": (record.get("metadata") or {}).get("annotator_id"),
                "annotation_count": len(spans),
                "annotations": spans,
                "non_empty_text_fields": non_empty_text_fields,
            }

            for ann in spans:
                all_span_annotations.append(
                    {
                        "dataset": record.get("dataset"),
                        "split": split,
                        "setup_id": record.get("setup_id"),
                        "example_idx": record.get("example_idx"),
                        "annotator_id": (record.get("metadata") or {}).get("annotator_id"),
                        **ann,
                    }
                )

    tables_with_text_feedback = sum(
        1
        for item in tables_with_annotations.values()
        if item.get("non_empty_text_fields")
    )

    result = {
        "summary": {
            "input_file": str(input_path),
            "table_count": len(by_table),
            "checked_count": len(checked_kpis),
            "not_checked_count": len(not_checked_kpis),
            "total_kpis": len(checked_kpis) + len(not_checked_kpis),
        },
        "annotation_summary": {
            "tables_with_span_annotations": sum(
                1 for item in tables_with_annotations.values() if item.get("annotation_count", 0) > 0
            ),
            "total_span_annotations": len(all_span_annotations),
            "tables_with_text_feedback": tables_with_text_feedback,
        },
        "tables_with_annotations": tables_with_annotations,
        "all_span_annotations": all_span_annotations,
        "table_source_mapping": table_source_mapping,
        "tables": by_table,
        "checked_kpis": checked_kpis,
        "not_checked_kpis": not_checked_kpis,
    }

    with output_path.open("w", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)

    print(f"Checked KPIs: {len(checked_kpis)}")
    print(f"Not checked KPIs: {len(not_checked_kpis)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
