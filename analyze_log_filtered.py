#!/usr/bin/env python3
"""
Recalculate accuracy and precision from a VLM extraction log,
discarding table entries where 0 ground-truth tables were loaded.

Usage:
    python analyze_log_filtered.py <log_file>
    python analyze_log_filtered.py data/logs/vlm_extraction_365669.log
"""

import re
import sys
from pathlib import Path


def parse_log(log_path: str):
    loaded_re  = re.compile(r"Loaded (\d+) tables matching filters")
    valid_re   = re.compile(r"Validation complete: (\d+)/(\d+) valid")

    entries = []  # list of (loaded_count, valid_kpis, total_kpis)

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Walk lines: when we see "Loaded N tables", look ahead for the next
    # "Validation complete" line and pair them.
    i = 0
    while i < len(lines):
        m_loaded = loaded_re.search(lines[i])
        if m_loaded:
            loaded_count = int(m_loaded.group(1))
            # Find the next validation line
            j = i + 1
            while j < len(lines):
                m_valid = valid_re.search(lines[j])
                if m_valid:
                    valid_kpis = int(m_valid.group(1))
                    total_kpis = int(m_valid.group(2))
                    entries.append((loaded_count, valid_kpis, total_kpis))
                    i = j  # advance outer loop past this pair
                    break
                j += 1
        i += 1

    return entries


def compute_stats(entries, label, skip_zero_gt=False):
    filtered = [e for e in entries if not (skip_zero_gt and e[0] == 0)]
    skipped  = len(entries) - len(filtered)

    total_tables   = len(filtered)
    total_extracted = sum(e[2] for e in filtered)
    total_valid     = sum(e[1] for e in filtered)

    precision = total_valid / total_extracted * 100 if total_extracted > 0 else 0.0
    # Per-table mean accuracy
    per_table_acc = [e[1] / e[2] * 100 if e[2] > 0 else 0.0 for e in filtered]
    mean_acc = sum(per_table_acc) / len(per_table_acc) if per_table_acc else 0.0

    print(f"\n{'=' * 58}")
    print(f"  {label}")
    print(f"{'=' * 58}")
    if skip_zero_gt:
        print(f"  Tables skipped (0 GT rows)  : {skipped}")
    print(f"  Tables evaluated            : {total_tables}")
    print(f"  Total KPIs extracted        : {total_extracted}")
    print(f"  Total KPIs valid            : {total_valid}")
    print(f"  Precision  (valid / ext)    : {precision:.2f}%")
    print(f"  Mean per-table accuracy     : {mean_acc:.2f}%")


def main():
    if len(sys.argv) < 2:
        # Default to the log referenced in the workspace
        log_path = Path("data/logs/vlm_extraction_365669.log")
    else:
        log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing: {log_path}")
    entries = parse_log(str(log_path))
    print(f"Total table entries found: {len(entries)}")

    compute_stats(entries, "OVERALL  (all tables)")
    compute_stats(entries, "FILTERED (excluding 0 GT tables)", skip_zero_gt=True)


if __name__ == "__main__":
    main()
