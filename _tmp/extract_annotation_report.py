#!/usr/bin/env python3
"""
extract_annotation_report.py
-----------------------------
Produces a single, human-readable JSON report for the trial-27 / vw-management
annotation campaign.

For each of the 25 annotated tables the output contains:
  - table identity (split, source_file)
  - per-annotator stats (n_valid, n_invalid, which rows are invalid, missing KPI notes)
  - per-row breakdown with both annotators' verdicts
  - inter-annotator agreement metrics
  - overall campaign summary

Usage:
    python _tmp/extract_annotation_report.py

Output:
    data/output/trial-27-annotation-report.json
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent
ANNOTATION_FILES = {
    "FILL_YOUR_NAME_HERE": BASE / "factgenie/factgenie/campaigns/vw-management/files/0-0-FILL_YOUR_NAME_HERE-1776676771.jsonl",
    "annotater_2":         BASE / "factgenie/factgenie/campaigns/vw-management/files/2-1-annotater_2-1776933375.jsonl",
}
FACTGENIE_OUTPUTS = BASE / "factgenie/factgenie/data/outputs/trial27"
EXTRACTION_DIR    = BASE / "data/output/trial-27/vlm_qwen_72b"
OUTPUT_FILE       = BASE / "data/output/trial-27-annotation-report.json"


# ---------------------------------------------------------------------------
# Step 1: load both annotation files → {split: annotation_record}
# ---------------------------------------------------------------------------
def load_annotations(path: Path) -> dict:
    records = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records[d["split"]] = d
    return records


annotations = {name: load_annotations(p) for name, p in ANNOTATION_FILES.items()}
annotator_names = list(annotations.keys())

# All splits from any annotator
all_splits = sorted(set(split for a in annotations.values() for split in a))


# ---------------------------------------------------------------------------
# Step 2: for each split, build the table report object
# ---------------------------------------------------------------------------
def kpi_label(kpi: dict) -> str:
    return f"{kpi.get('name','?')} | {kpi.get('key','?')} | {kpi.get('value','?')} | year={kpi.get('year','?')} | {kpi.get('units','?')}"


table_reports = []

for split in all_splits:
    # --- resolve source_file via factgenie output ---
    fg_file = FACTGENIE_OUTPUTS / f"{split}-manual-kpi.jsonl"
    source_file = None
    if fg_file.exists():
        fg_entry = json.loads(fg_file.read_text(encoding="utf-8"))
        source_file = fg_entry.get("metadata", {}).get("source_file")

    # --- load extraction KPIs for the row labels ---
    kpis = []
    if source_file:
        ext_file = EXTRACTION_DIR / source_file
        if ext_file.exists():
            ext_data = json.loads(ext_file.read_text(encoding="utf-8"))
            kpis = ext_data.get("kpis", [])

    total_kpis = len(kpis)

    # --- per-annotator info ---
    annotator_info = {}
    for name in annotator_names:
        rec = annotations[name].get(split)
        if rec is None:
            annotator_info[name] = {
                "n_valid": None, "n_invalid": None,
                "invalid_row_indices": [], "missing_kpis_note": "",
                "annotated": False,
            }
            continue

        row_checks = {c["rowIndex"]: c["value"] for c in rec.get("row_checks", [])}
        invalid_indices = sorted(idx for idx, valid in row_checks.items() if not valid)
        valid_count   = sum(1 for v in row_checks.values() if v)
        invalid_count = sum(1 for v in row_checks.values() if not v)

        # rows the annotator didn't check at all (not in row_checks)
        unchecked = [i for i in range(total_kpis) if i not in row_checks]

        missing_note = ""
        for tf in rec.get("text_fields", []):
            if tf.get("label") == "Remaining kpis" and tf.get("value", "").strip():
                missing_note = tf["value"].strip()

        # invalid KPI details
        invalid_kpis = []
        for idx in invalid_indices:
            entry = {"row_index": idx}
            if idx < len(kpis):
                entry["kpi"] = kpi_label(kpis[idx])
            invalid_kpis.append(entry)

        annotator_info[name] = {
            "annotated": True,
            "n_valid":   valid_count,
            "n_invalid": invalid_count,
            "invalid_row_indices": invalid_indices,
            "invalid_kpis": invalid_kpis,
            "unchecked_rows": unchecked,
            "missing_kpis_note": missing_note,
        }

    # --- per-row breakdown ---
    rows = []
    for i, kpi in enumerate(kpis):
        row = {
            "row_index": i,
            "kpi": kpi_label(kpi),
        }
        verdicts = {}
        for name in annotator_names:
            rec = annotations[name].get(split)
            if rec is None:
                verdicts[name] = None
            else:
                checks = {c["rowIndex"]: c["value"] for c in rec.get("row_checks", [])}
                verdicts[name] = checks.get(i, None)  # None = not reviewed
        row["verdicts"] = verdicts

        # consensus
        vals = [v for v in verdicts.values() if v is not None]
        if not vals:
            row["consensus"] = "unreviewed"
        elif all(vals):
            row["consensus"] = "valid"
        elif not any(vals):
            row["consensus"] = "invalid"
        else:
            row["consensus"] = "disagreement"
        rows.append(row)

    # --- inter-annotator agreement ---
    both_reviewed = [
        r for r in rows
        if all(r["verdicts"].get(n) is not None for n in annotator_names)
    ]
    agreed_valid    = [r["row_index"] for r in both_reviewed if r["consensus"] == "valid"]
    agreed_invalid  = [r["row_index"] for r in both_reviewed if r["consensus"] == "invalid"]
    disagreed       = [r["row_index"] for r in both_reviewed if r["consensus"] == "disagreement"]
    unreviewed      = [r["row_index"] for r in rows if r["consensus"] == "unreviewed"]

    n_both = len(both_reviewed)
    iaa = (len(agreed_valid) + len(agreed_invalid)) / n_both if n_both else None

    agreement = {
        "both_reviewed": n_both,
        "agreed_valid_count":   len(agreed_valid),
        "agreed_invalid_count": len(agreed_invalid),
        "disagreement_count":   len(disagreed),
        "unreviewed_count":     len(unreviewed),
        "inter_annotator_agreement": round(iaa, 4) if iaa is not None else None,
        "agreed_valid_rows":   agreed_valid,
        "agreed_invalid_rows": agreed_invalid,
        "disagreed_rows":      disagreed,
        "unreviewed_rows":     unreviewed,
    }

    table_reports.append({
        "split":       split,
        "source_file": source_file,
        "total_kpis":  total_kpis,
        "annotators":  annotator_info,
        "agreement":   agreement,
        "rows":        rows,
    })


# ---------------------------------------------------------------------------
# Step 3: overall summary
# ---------------------------------------------------------------------------
total_kpis_all      = sum(t["total_kpis"] for t in table_reports)
total_agreed_valid  = sum(len(t["agreement"]["agreed_valid_rows"])   for t in table_reports)
total_agreed_invalid= sum(len(t["agreement"]["agreed_invalid_rows"]) for t in table_reports)
total_disagreed     = sum(len(t["agreement"]["disagreed_rows"])      for t in table_reports)
total_unreviewed    = sum(len(t["agreement"]["unreviewed_rows"])     for t in table_reports)

tables_with_disagreement = [t["split"] for t in table_reports if t["agreement"]["disagreement_count"] > 0]
tables_with_any_invalid  = [
    t["split"] for t in table_reports
    if t["agreement"]["agreed_invalid_count"] > 0
    or t["agreement"]["disagreement_count"] > 0
]

avg_iaa_vals = [t["agreement"]["inter_annotator_agreement"] for t in table_reports
                if t["agreement"]["inter_annotator_agreement"] is not None]
avg_iaa = round(sum(avg_iaa_vals) / len(avg_iaa_vals), 4) if avg_iaa_vals else None

summary = {
    "annotation_files": {k: str(v) for k, v in ANNOTATION_FILES.items()},
    "extraction_dir":   str(EXTRACTION_DIR),
    "num_tables":       len(table_reports),
    "total_kpis":       total_kpis_all,
    "agreed_valid":     total_agreed_valid,
    "agreed_invalid":   total_agreed_invalid,
    "disagreed":        total_disagreed,
    "unreviewed":       total_unreviewed,
    "avg_inter_annotator_agreement": avg_iaa,
    "tables_with_any_invalid":       tables_with_any_invalid,
    "tables_with_disagreement":      tables_with_disagreement,
    "per_annotator": {
        name: {
            "total_invalid": sum(
                info["n_invalid"] or 0
                for t in table_reports
                for n, info in t["annotators"].items()
                if n == name and info["annotated"]
            ),
            "tables_with_invalid": [
                t["split"]
                for t in table_reports
                if (t["annotators"].get(name, {}).get("n_invalid") or 0) > 0
            ],
        }
        for name in annotator_names
    },
}

report = {
    "summary": summary,
    "tables":  table_reports,
}


# ---------------------------------------------------------------------------
# Step 4: write output
# ---------------------------------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print("=" * 70)
print("ANNOTATION REPORT")
print("=" * 70)
print(f"  Tables               : {summary['num_tables']}")
print(f"  Total KPIs           : {summary['total_kpis']}")
print(f"  Agreed valid         : {summary['agreed_valid']}")
print(f"  Agreed invalid       : {summary['agreed_invalid']}")
print(f"  Disagreements        : {summary['disagreed']}")
print(f"  Avg IAA              : {summary['avg_inter_annotator_agreement']}")
print(f"  Tables w/ disagreement: {tables_with_disagreement}")
for name in annotator_names:
    info = summary["per_annotator"][name]
    print(f"  [{name}]  invalid={info['total_invalid']}  tables={info['tables_with_invalid']}")
print("=" * 70)
print(f"\n  Report saved to: {OUTPUT_FILE}")
