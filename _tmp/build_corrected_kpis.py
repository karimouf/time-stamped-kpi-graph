#!/usr/bin/env python3
"""
build_corrected_kpis.py
------------------------
Produces one JSON file per table (25 total) containing the full merged
annotation result with complete KPI objects.

Structure of each output file
==============================
{
  "split":        "table-NNN",
  "source_file":  "..._kpis.json",
  "source_image": "table_NN.png",
  "year":         2015,           // extracted from source_file name
  "stats": {
    "total_extracted": N,
    "valid": N,
    "invalid": N,
    "unreviewed": N,
    "missing_noted": N
  },
  "annotation_notes": {           // free-text notes left by annotators
    "FILL_YOUR_NAME_HERE": "...",
    "annotater_2": "..."
  },
  "valid_kpis": [                 // confirmed correct — keep as-is
    { ...full kpi object... }
  ],
  "invalid_kpis": [               // invalid or disagreed → needs correction
    {
      "original":   { ...full kpi object... },
      "annotation": {
        "consensus": "invalid" | "disagreement",
        "verdicts":  { "ann1": true/false, "ann2": true/false/null }
      },
      "corrected":  null          // ← fill in the fixed KPI here (or delete entry)
    }
  ],
  "unreviewed_kpis": [            // not reviewed — treat as uncertain
    { ...full kpi object... }
  ],
  "missing_kpis": [               // KPIs annotators said are absent from extraction
    // ← add them here manually
  ]
}

Usage:
    python _tmp/build_corrected_kpis.py

Output directory:
    data/output/trial-27-corrected-kpis/   (one JSON per table)
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE           = Path(__file__).parent.parent
REPORT_FILE    = BASE / "data/output/trial-27-annotation-report.json"
EXTRACTION_DIR = BASE / "data/output/trial-27/vlm_qwen_72b"
OUT_DIR        = BASE / "data/output/trial-27-corrected-kpis"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load report (contains per-row verdicts and annotator notes)
# ---------------------------------------------------------------------------
report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
annotator_names = list(report["summary"]["per_annotator"].keys())

# ---------------------------------------------------------------------------
# Helper: extract year from source filename  e.g. "..._vw_ar15_..." → 2015
# ---------------------------------------------------------------------------
_year_re = re.compile(r"[-_]ar(\d{2})[-_]")

def year_from_filename(name: str) -> int | None:
    m = _year_re.search(name or "")
    if m:
        yy = int(m.group(1))
        return 2000 + yy
    return None


# ---------------------------------------------------------------------------
# Process each table
# ---------------------------------------------------------------------------
written = 0
skipped = 0

for t in report["tables"]:
    split       = t["split"]
    source_file = t["source_file"]

    # --- load raw KPI objects ---
    kpis = []
    if source_file:
        ext_path = EXTRACTION_DIR / source_file
        if ext_path.exists():
            ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
            kpis = ext_data.get("kpis", [])

    if not kpis and not source_file:
        skipped += 1
        print(f"  SKIP {split}: no source file resolved")
        continue

    # source_image: take from first KPI, fall back to None
    source_image = kpis[0].get("source_image") if kpis else None

    # --- build row-index → full kpi lookup ---
    kpi_by_idx = {i: kpi for i, kpi in enumerate(kpis)}

    # --- build row-index → verdict info lookup ---
    row_lookup = {r["row_index"]: r for r in t["rows"]}

    # --- annotator notes ---
    annotation_notes = {}
    for ann_name, ann_info in t.get("annotators", {}).items():
        note = ann_info.get("missing_kpis_note", "").strip()
        if note:
            annotation_notes[ann_name] = note

    # --- partition rows ---
    valid_kpis      = []
    invalid_kpis    = []
    unreviewed_kpis = []

    for idx, kpi in kpi_by_idx.items():
        row   = row_lookup.get(idx)
        if row is None:
            unreviewed_kpis.append(kpi)
            continue

        consensus = row["consensus"]

        if consensus == "valid":
            valid_kpis.append(kpi)

        elif consensus in ("invalid", "disagreement"):
            verdicts = row.get("verdicts", {})
            invalid_kpis.append({
                "original":   kpi,
                "annotation": {
                    "consensus": consensus,
                    "verdicts":  verdicts,
                },
                "corrected":  None,   # ← user fills this in
            })

        else:  # "unreviewed"
            unreviewed_kpis.append(kpi)

    # --- stats ---
    stats = {
        "total_extracted": len(kpis),
        "valid":           len(valid_kpis),
        "invalid":         len(invalid_kpis),
        "unreviewed":      len(unreviewed_kpis),
        "missing_noted":   1 if annotation_notes else 0,
    }

    # --- assemble output ---
    out = {
        "split":            split,
        "source_file":      source_file,
        "source_image":     source_image,
        "year":             year_from_filename(source_file or ""),
        "stats":            stats,
        "annotation_notes": annotation_notes,
        "valid_kpis":       valid_kpis,
        "invalid_kpis":     invalid_kpis,
        "unreviewed_kpis":  unreviewed_kpis,
        "missing_kpis":     [],   # ← user adds manually based on annotation_notes
    }

    out_path = OUT_DIR / f"{split}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    written += 1

print(f"\nWritten {written} table JSON files → {OUT_DIR}")
if skipped:
    print(f"Skipped {skipped} tables (no source file)")

# ---------------------------------------------------------------------------
# Print a quick overview
# ---------------------------------------------------------------------------
total_valid = total_invalid = total_unreviewed = 0
tables_needing_work = []

for path in sorted(OUT_DIR.glob("table-*.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d["stats"]
    total_valid      += s["valid"]
    total_invalid    += s["invalid"]
    total_unreviewed += s["unreviewed"]
    needs = []
    if s["invalid"] > 0:
        needs.append(f"{s['invalid']} invalid")
    if s["missing_noted"]:
        needs.append("missing noted")
    if needs:
        tables_needing_work.append(f"  {d['split']} ({d['year']}): {', '.join(needs)}")

print(f"\nOverall  valid={total_valid}  invalid={total_invalid}  unreviewed={total_unreviewed}")
print(f"\nTables needing corrections ({len(tables_needing_work)}):")
print("\n".join(tables_needing_work) if tables_needing_work else "  none")
