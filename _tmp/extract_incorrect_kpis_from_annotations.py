import argparse
import json
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


def line_spans(text: str):
    spans = []
    pos = 0
    for line in text.splitlines():
        start = pos
        end = pos + len(line)
        spans.append((start, end, line))
        pos = end + 1
    return spans


def overlaps(span_a, span_b):
    a_start, a_end = span_a
    b_start, b_end = span_b
    return a_start < b_end and b_start < a_end


def extract_incorrect_rows(record: dict):
    output = record.get("output", "")
    annotations = record.get("annotations", [])

    if not output or not annotations:
        return []

    spans = line_spans(output)
    incorrect = []
    seen = set()

    for ann in annotations:
        ann_start = int(ann.get("start", 0))
        ann_text = ann.get("text", "")
        ann_end = ann_start + len(ann_text)

        for idx, (line_start, line_end, line_text) in enumerate(spans):
            if overlaps((ann_start, ann_end), (line_start, line_end)):
                if idx in seen:
                    continue
                seen.add(idx)
                incorrect.append(
                    {
                        "line_index": idx,
                        "split": record.get("split"),
                        "dataset": record.get("dataset"),
                        "setup_id": record.get("setup_id"),
                        "example_idx": record.get("example_idx"),
                        "annotation_type": ann.get("type"),
                        "annotation_text": ann_text,
                        "annotation_start": ann_start,
                        **parse_kpi_line(line_text),
                    }
                )

    return incorrect


def main():
    parser = argparse.ArgumentParser(description="Extract incorrect KPI rows from Factgenie annotation JSONL")
    parser.add_argument("--input", required=True, help="Path to annotation JSONL file")
    parser.add_argument("--output", default="incorrect_kpis.json", help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    all_incorrect = []
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace(": NaN", ": null")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            all_incorrect.extend(extract_incorrect_rows(record))

    with output_path.open("w", encoding="utf-8") as out:
        json.dump(all_incorrect, out, indent=2, ensure_ascii=False)

    print(f"Extracted {len(all_incorrect)} incorrect KPI rows")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
