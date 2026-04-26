"""
build_ground_truth.py

For each of the 25 trial-27 corrected tables:
  1. Load valid_kpis from trial-27-corrected-kpis/table-NNN.json
  2. Load original extracted KPIs from trial-27/vlm_qwen_72b/<source_file>
  3. Build a (row_idx, col_idx) grid of valid KPIs
  4. Identify source cells that have no valid KPI → add to missing_kpis
     (skip unreviewed / invalid positions — only flag cells with no valid KPI
      AND no unreviewed/invalid KPI occupying that slot either)
  5. Update the corrected JSON with missing_kpis + updated stats
  6. Write a consolidated ground-truth JSONL:
       data/output/trial-27-ground-truth.jsonl  (one line per table, valid+missing only)
  7. Write per-table CSV grids to data/output/trial-27-ground-truth-grids/
"""

import csv
import json
from pathlib import Path

CORRECTED_DIR = Path("data/output/trial-27-corrected-kpis")
SOURCE_DIR = Path("data/output/trial-27/vlm_qwen_72b")
OUTPUT_GT = Path("data/output/trial-27-ground-truth.jsonl")
OUTPUT_GRIDS = Path("data/output/trial-27-ground-truth-grids")
OUTPUT_GRIDS.mkdir(parents=True, exist_ok=True)

summary_rows = []

for corrected_path in sorted(CORRECTED_DIR.glob("table-*.json")):
    table_id = corrected_path.stem  # e.g. table-001
    d = json.loads(corrected_path.read_text(encoding="utf-8"))

    valid_kpis = d["valid_kpis"]
    unreviewed_kpis = d["unreviewed_kpis"]
    invalid_kpis = [e["original"] if isinstance(e, dict) and "original" in e else e
                    for e in d["invalid_kpis"]]

    # Build set of positions already accounted for (valid + unreviewed + invalid)
    accounted_positions = set()
    for kpi in valid_kpis + unreviewed_kpis + invalid_kpis:
        accounted_positions.add((kpi["row_idx"], kpi["col_idx"]))

    # Build lookup: (row_idx, col_idx) -> valid kpi
    valid_grid = {}
    for kpi in valid_kpis:
        valid_grid[(kpi["row_idx"], kpi["col_idx"])] = kpi

    # Load original source KPIs
    src_path = SOURCE_DIR / d["source_file"]
    src_data = json.loads(src_path.read_text(encoding="utf-8"))
    src_kpis = src_data["kpis"]

    # Find missing: source positions that have no valid KPI AND no other coverage
    missing_kpis = []
    for src_kpi in src_kpis:
        pos = (src_kpi["row_idx"], src_kpi["col_idx"])
        if pos not in valid_grid and pos not in accounted_positions:
            missing_kpis.append(src_kpi)

    # Update the corrected JSON
    d["missing_kpis"] = missing_kpis
    d["stats"]["missing_noted"] = len(missing_kpis)
    corrected_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----------------------------------------------------------------
    # Build CSV grid: rows = row_idx, cols = col_idx
    # Each cell: "name | key | value year units"
    # ----------------------------------------------------------------
    all_for_grid = {}
    for kpi in valid_kpis:
        all_for_grid[(kpi["row_idx"], kpi["col_idx"])] = ("valid", kpi)
    for kpi in unreviewed_kpis:
        pos = (kpi["row_idx"], kpi["col_idx"])
        if pos not in all_for_grid:
            all_for_grid[pos] = ("unreviewed", kpi)
    for kpi in invalid_kpis:
        pos = (kpi["row_idx"], kpi["col_idx"])
        if pos not in all_for_grid:
            all_for_grid[pos] = ("invalid", kpi)
    for kpi in missing_kpis:
        pos = (kpi["row_idx"], kpi["col_idx"])
        if pos not in all_for_grid:
            all_for_grid[pos] = ("missing", kpi)

    if not all_for_grid:
        continue

    max_row = max(r for r, _ in all_for_grid)
    max_col = max(c for _, c in all_for_grid)

    grid_path = OUTPUT_GRIDS / f"{table_id}_grid.csv"
    with open(grid_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # Header row: col indices
        header = ["row\\col"] + [f"col_{c}" for c in range(max_col + 1)]
        writer.writerow(header)
        for r in range(max_row + 1):
            row_cells = [f"row_{r}"]
            for c in range(max_col + 1):
                entry = all_for_grid.get((r, c))
                if entry is None:
                    row_cells.append("")
                else:
                    status, kpi = entry
                    yr = kpi.get("year", "?")
                    val = kpi.get("value", "?")
                    units = kpi.get("units", "")
                    label = f"[{status}] {kpi['name']} | {kpi['key']} | {val} {yr} {units}".strip()
                    row_cells.append(label)
            writer.writerow(row_cells)

    # ----------------------------------------------------------------
    # Append to ground-truth JSONL (valid + missing, table-level record)
    # ----------------------------------------------------------------
    gt_record = {
        "split": table_id,
        "source_file": d["source_file"],
        "source_image": d["source_image"],
        "year": d["year"],
        "valid_kpis": valid_kpis,
        "missing_kpis": missing_kpis,
    }

    summary_rows.append({
        "table": table_id,
        "source_file": d["source_file"],
        "year": d["year"],
        "valid": len(valid_kpis),
        "unreviewed": len(unreviewed_kpis),
        "invalid": len(invalid_kpis),
        "missing": len(missing_kpis),
    })

    print(f"{table_id}: valid={len(valid_kpis)}, unreviewed={len(unreviewed_kpis)}, "
          f"invalid={len(invalid_kpis)}, missing={len(missing_kpis)}")

# Write JSONL
with open(OUTPUT_GT, "w", encoding="utf-8") as f:
    for row in summary_rows:
        # Re-load to get the final valid+missing
        table_id = row["table"]
        corrected_path = CORRECTED_DIR / f"{table_id}.json"
        d = json.loads(corrected_path.read_text(encoding="utf-8"))
        gt_record = {
            "split": table_id,
            "source_file": d["source_file"],
            "source_image": d["source_image"],
            "year": d["year"],
            "valid_kpis": d["valid_kpis"],
            "missing_kpis": d["missing_kpis"],
        }
        f.write(json.dumps(gt_record, ensure_ascii=False) + "\n")

# Print summary
print()
print("=" * 70)
total_valid = sum(r["valid"] for r in summary_rows)
total_unreviewed = sum(r["unreviewed"] for r in summary_rows)
total_invalid = sum(r["invalid"] for r in summary_rows)
total_missing = sum(r["missing"] for r in summary_rows)
print(f"TOTAL  valid={total_valid}, unreviewed={total_unreviewed}, "
      f"invalid={total_invalid}, missing={total_missing}")
print(f"Ground truth JSONL written to: {OUTPUT_GT}")
print(f"Grid CSVs written to:          {OUTPUT_GRIDS}/")
