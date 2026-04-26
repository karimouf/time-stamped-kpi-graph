#!/usr/bin/env python3
"""
merge_annotations.py
---------------------
Merges both annotators' results into a single authoritative annotation file.

Rules:
  - Both valid        → valid   (True)
  - Both invalid      → invalid (False)
  - Disagreement      → invalid (False)  ← always take the stricter side
  - Both unreviewed   → skipped (row not included in row_checks)

Output format mirrors the raw annotation JSONL files used by validate.py,
so it can be fed directly into the validation pipeline.

Output: data/output/trial-27-merged-annotations.jsonl
        data/output/trial-27-merged-annotations-summary.json
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE        = Path(__file__).parent.parent
REPORT_FILE = BASE / "data/output/trial-27-annotation-report.json"
OUT_JSONL   = BASE / "data/output/trial-27-merged-annotations.jsonl"
OUT_SUMMARY = BASE / "data/output/trial-27-merged-annotations-summary.json"

# ---------------------------------------------------------------------------
# Load the pre-built report
# ---------------------------------------------------------------------------
report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
tables = report["tables"]

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
merged_records = []
summary_tables = []

total_valid = total_invalid = total_skipped = 0

for t in tables:
    split      = t["split"]
    source_file = t["source_file"]
    rows        = t["rows"]

    row_checks    = []
    skipped_rows  = []
    n_valid = n_invalid = 0

    for row in rows:
        consensus = row["consensus"]

        if consensus == "valid":
            row_checks.append({"rowIndex": row["row_index"], "value": True})
            n_valid += 1
        elif consensus in ("invalid", "disagreement"):
            row_checks.append({"rowIndex": row["row_index"], "value": False})
            n_invalid += 1
        else:
            # unreviewed — omit from row_checks
            skipped_rows.append(row["row_index"])

    total_valid   += n_valid
    total_invalid += n_invalid
    total_skipped += len(skipped_rows)

    # Collect any missing-KPI notes from either annotator
    notes = []
    for ann_info in t.get("annotators", {}).values():
        note = ann_info.get("missing_kpis_note", "").strip()
        if note:
            notes.append(note)
    combined_note = " | ".join(notes) if notes else ""

    record = {
        "split":        split,
        "source_file":  source_file,
        "annotator_id": "merged",
        "merge_rules": {
            "disagreement_resolution": "invalid",
            "unreviewed_handling":     "skip",
        },
        "row_checks":  row_checks,
        "text_fields": [
            {"label": "Remaining kpis", "value": combined_note}
        ] if combined_note else [],
    }

    merged_records.append(record)

    # per-table summary entry
    disagreed_rows = t["agreement"]["disagreed_rows"]
    summary_tables.append({
        "split":             split,
        "source_file":       source_file,
        "total_kpis":        t["total_kpis"],
        "merged_valid":      n_valid,
        "merged_invalid":    n_invalid,
        "skipped_unreviewed":len(skipped_rows),
        "resolved_as_invalid_from_disagreement": len(disagreed_rows),
        "disagreed_rows":    disagreed_rows,
        "skipped_rows":      skipped_rows,
    })

# ---------------------------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------------------------
with open(OUT_JSONL, "w", encoding="utf-8") as fh:
    for rec in merged_records:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Written {len(merged_records)} records → {OUT_JSONL}")

# ---------------------------------------------------------------------------
# Write summary JSON
# ---------------------------------------------------------------------------
summary = {
    "source_report":    str(REPORT_FILE),
    "output_jsonl":     str(OUT_JSONL),
    "num_tables":       len(merged_records),
    "total_valid":      total_valid,
    "total_invalid":    total_invalid,
    "total_skipped_unreviewed": total_skipped,
    "merge_rules": {
        "both_valid":       "valid",
        "both_invalid":     "invalid",
        "disagreement":     "invalid  ← stricter side always wins",
        "both_unreviewed":  "skipped (not included in row_checks)",
    },
    "tables": summary_tables,
}

OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Written summary      → {OUT_SUMMARY}")
print()
print(f"  Valid    : {total_valid}")
print(f"  Invalid  : {total_invalid}  (includes {sum(t['resolved_as_invalid_from_disagreement'] for t in summary_tables)} resolved from disagreements)")
print(f"  Skipped  : {total_skipped}")
