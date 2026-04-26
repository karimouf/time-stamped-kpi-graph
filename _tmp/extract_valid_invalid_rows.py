import argparse
import json
from pathlib import Path


def parse_kpi_line(line: str) -> dict:
    parts = [part.strip() for part in line.split("|")]
    if len(parts) < 6:
        return {"raw_line": line}

    def extract_value(token: str, key: str) -> str:
        prefix = f"{key}="
        return token[len(prefix):].strip() if token.startswith(prefix) else token

    return {
        "name": parts[0],
        "key": parts[1],
        "country": parts[2],
        "value": extract_value(parts[3], "value"),
        "year": extract_value(parts[4], "year"),
        "units": extract_value(parts[5], "units"),
        "raw_line": line,
    }


def build_line_spans(text: str):
    spans = []
    pos = 0
    for line in text.splitlines():
        start = pos
        end = pos + len(line)
        spans.append((start, end, line))
        pos = end + 1
    return spans


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def invalid_row_indices_from_row_checks(record: dict):
    indices = set()
    for item in record.get("row_checks", []):
        row_index = item.get("rowIndex")
        if row_index is None:
            continue
        checked = item.get("value")
        if checked is None:
            checked = item.get("checked", False)
        if bool(checked):
            indices.add(int(row_index))
    return indices


def invalid_row_indices_from_annotations(record: dict, spans):
    indices = set()
    for ann in record.get("annotations", []):
        ann_start = int(ann.get("start", 0))
        ann_text = str(ann.get("text", ""))
        ann_end = ann_start + len(ann_text)

        for idx, (line_start, line_end, _line_text) in enumerate(spans):
            if overlaps(ann_start, ann_end, line_start, line_end):
                indices.add(idx)
    return indices


def classify_rows(record: dict):
    output = str(record.get("output", ""))
    spans = build_line_spans(output)

    invalid_indices = invalid_row_indices_from_row_checks(record)
    if not invalid_indices:
        invalid_indices = invalid_row_indices_from_annotations(record, spans)

    base = {
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "setup_id": record.get("setup_id"),
        "example_idx": record.get("example_idx"),
        "annotator_id": (record.get("metadata") or {}).get("annotator_id"),
    }

    valid_rows = []
    invalid_rows = []

    for idx, (_start, _end, line_text) in enumerate(spans):
        row = {
            **base,
            "line_index": idx,
            **parse_kpi_line(line_text),
        }
        if idx in invalid_indices:
            invalid_rows.append(row)
        else:
            valid_rows.append(row)

    return valid_rows, invalid_rows


def load_jsonl(path: Path):
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


def main():
    parser = argparse.ArgumentParser(description="Extract valid and invalid KPI rows from Factgenie annotation JSONL")
    parser.add_argument("--input", required=True, help="Path to annotation JSONL")
    parser.add_argument("--output", default="valid_invalid_rows.json", help="Output JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    all_valid = []
    all_invalid = []

    for record in load_jsonl(input_path):
        valid_rows, invalid_rows = classify_rows(record)
        all_valid.extend(valid_rows)
        all_invalid.extend(invalid_rows)

    result = {
        "summary": {
            "input_file": str(input_path),
            "valid_count": len(all_valid),
            "invalid_count": len(all_invalid),
            "total_rows": len(all_valid) + len(all_invalid),
            "rule": "row_checks if present, otherwise annotation span overlap",
        },
        "valid_kpis": all_valid,
        "invalid_kpis": all_invalid,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Valid rows: {len(all_valid)}")
    print(f"Invalid rows: {len(all_invalid)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
