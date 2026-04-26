from pathlib import Path
import json
import re
import shutil
import zipfile
from html import escape

import yaml

root = Path(r"c:/Users/karim/dev/time-stamped-kpi-graph")
aggregate_kpi_json = root / "random_management_25.json"
setup_id = "manual-kpi"

trial_match = re.search(r"random_tables_(trial\d+)\.json$", aggregate_kpi_json.name)
trial_id = trial_match.group(1) if trial_match else "trial27"

src = root / f"random_tables_{trial_id}_images"
dataset_root = root / f"factgenie/factgenie/data/inputs/{trial_id}"
outputs_root = root / f"factgenie/factgenie/data/outputs/{trial_id}"
datasets_yml_path = root / "factgenie/factgenie/data/datasets.yml"
zip_path = root / f"factgenie/factgenie/data/inputs/{trial_id}_per_split.zip"

# Try a couple of known image roots where table_<idx>.png files may exist.
possible_image_roots = [
    root / "data/detected_tables_test_14/detected_tables_test",
    root / "data/detected_tables_test_14",
    root / "data/detected_tables_test_13",
    root / "data/detected_tables_test",
]

section_context_root = root / "data/detected_tables_test_14/detected_tables_test"

dataset_root.mkdir(parents=True, exist_ok=True)
outputs_root.mkdir(parents=True, exist_ok=True)

payload = json.loads(aggregate_kpi_json.read_text(encoding="utf-8"))
selected_tables = payload.get("selected_tables", [])
all_kpis = payload.get("kpis", [])

kpis_by_source_file = {}
for kpi in all_kpis:
    source = str(kpi.get("source", "")).replace("\\", "/")
    source_file = source.rsplit("/", 1)[-1] if source else ""
    if not source_file:
        continue
    kpis_by_source_file.setdefault(source_file, []).append(kpi)

existing_html_files = []
if src.exists():
    existing_html_files = sorted([p for p in src.glob("*.html") if p.name != "index.html"])

# Use extracted HTML files if available; otherwise synthesize from selected_tables in JSON.
records = []
if existing_html_files:
    for html_file in existing_html_files:
        source_stem = re.sub(r"^\d+_", "", html_file.stem)
        records.append({
            "source_file": f"{source_stem}.json",
            "html_file": html_file,
        })
else:
    for table in selected_tables:
        source_file = table.get("source_file")
        if source_file:
            records.append({
                "source_file": source_file,
                "html_file": None,
            })

pattern = re.compile(r'<img\s+src="([^"]+)"')
table_wrap_pattern = re.compile(r"<div class=\"table-wrap\">.*?</div>", re.DOTALL)
table_wrap_css_pattern = re.compile(r"\s*\.table-wrap\s*\{[^}]*\}\s*", re.DOTALL)
table_css_pattern = re.compile(r"\s*table\s*\{[^}]*\}\s*", re.DOTALL)
th_td_css_pattern = re.compile(r"\s*th,\s*td\s*\{[^}]*\}\s*", re.DOTALL)
th_css_pattern = re.compile(r"\s*th\s*\{[^}]*\}\s*", re.DOTALL)
meta_block_pattern = re.compile(r"(<div class=\"meta\">.*?</div>)", re.DOTALL)
stem_pattern = re.compile(r"^(?P<report>.+?)_page_(?P<page>\d+)_table_(?P<table>\d+)_")


def resolve_section_context(source_stem: str) -> str | None:
    match = stem_pattern.search(source_stem)
    if not match:
        return None

    report_name = match.group("report")
    page_num = int(match.group("page"))
    page_folder = f"page_{page_num:03d}"
    context_path = section_context_root / report_name / page_folder / "section_context.txt"

    if not context_path.exists():
        return None

    try:
        context_text = context_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None

    if not context_text:
        return None

    if re.match(r"^Page\s+\d+\s*:", context_text, flags=re.IGNORECASE):
        return context_text

    return f"Page {page_num}: {context_text}"


def resolve_image_from_source(source_file: str) -> Path | None:
    # Expects something like "divisions_vw_ar15_page_025_table_00_kpis.json"
    match = re.match(r"(.+)_page_(\d{3})_table_(\d{2})_kpis\.json$", source_file)
    if not match:
        return None

    report_name, page_num, table_idx = match.groups()
    rel = Path(report_name) / f"page_{page_num}" / f"table_{table_idx}.png"

    for image_root in possible_image_roots:
        candidate = image_root / rel
        if candidate.exists():
            return candidate

    return None


def build_fallback_html(source_file: str, year, num_kpis: int, img_rel: str | None) -> str:
    image_block = '<p class="missing">Image not found.</p>'
    if img_rel:
        image_block = f'<img src="{escape(img_rel)}" alt="{escape(source_file)}" loading="lazy" />'

    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{escape(source_file)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f8; }}
        .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
        h1 {{ font-size: 18px; margin: 0 0 12px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 12px; }}
        .missing {{ color: #a94442; }}
        .meta {{ margin: 8px 0 14px 0; color: #555; font-size: 13px; }}
    </style>
</head>
<body>
    <div class=\"card\">
        <h1>{escape(source_file)}</h1>
        <div class=\"meta\">Year: {escape(str(year))} | KPIs: {escape(str(num_kpis))}</div>
        {image_block}
    </div>
</body>
</html>
"""


copied = 0
rewritten = 0
missing = []
removed_table_wrap = 0
splits_created = []
outputs_created = 0
missing_kpi_json = []

for old_split in dataset_root.glob("table-*"):
    if old_split.is_dir():
        shutil.rmtree(old_split)

stale_test_split = dataset_root / "test"
if stale_test_split.exists() and stale_test_split.is_dir():
    shutil.rmtree(stale_test_split)

for old_output in outputs_root.glob(f"table-*-{setup_id}.jsonl"):
    if old_output.is_file():
        old_output.unlink()

for idx, record in enumerate(records, start=1):
    split_name = f"table-{idx:03d}"
    split_dir = dataset_root / split_name
    img_dir = split_dir / "img"
    split_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    splits_created.append(split_name)

    source_file = record["source_file"]
    source_stem = Path(source_file).stem

    if record["html_file"] is not None:
        html_file = record["html_file"]
        txt = html_file.read_text(encoding="utf-8")
        match = pattern.search(txt)

        if table_wrap_pattern.search(txt):
            txt = table_wrap_pattern.sub("", txt)
            txt = table_wrap_css_pattern.sub("\n", txt)
            txt = table_css_pattern.sub("\n", txt)
            txt = th_td_css_pattern.sub("\n", txt)
            txt = th_css_pattern.sub("\n", txt)
            removed_table_wrap += 1

        if match:
            rel = match.group(1)
            abs_img = (html_file.parent / rel).resolve()

            if abs_img.exists():
                new_name = f"{html_file.stem}{abs_img.suffix.lower()}"
                target = img_dir / new_name
                shutil.copy2(abs_img, target)
                copied += 1
                txt = txt.replace(rel, f"img/{new_name}")
                rewritten += 1
            else:
                missing.append(str(abs_img))
    else:
        table_meta = next((t for t in selected_tables if t.get("source_file") == source_file), {})
        resolved_img = resolve_image_from_source(source_file)
        rel_img = None
        if resolved_img is not None and resolved_img.exists():
            new_name = f"{source_stem}{resolved_img.suffix.lower()}"
            shutil.copy2(resolved_img, img_dir / new_name)
            copied += 1
            rel_img = f"img/{new_name}"
            rewritten += 1
        else:
            missing.append(source_file)

        txt = build_fallback_html(
            source_file=source_file,
            year=table_meta.get("year"),
            num_kpis=int(table_meta.get("num_kpis", 0)),
            img_rel=rel_img,
        )

    section_context = resolve_section_context(source_stem)
    if section_context:
        context_block = f'\n        <div class="meta"><strong>Section Context:</strong> {escape(section_context)}</div>'
        if meta_block_pattern.search(txt):
            txt = meta_block_pattern.sub(rf"\1{context_block}", txt, count=1)
        elif "<h1" in txt:
            txt = txt.replace("</h1>", "</h1>" + context_block, 1)

    html_name = f"{idx:03d}_{source_stem}.html"
    (split_dir / html_name).write_text(txt, encoding="utf-8")

    output_jsonl_path = outputs_root / f"{split_name}-{setup_id}.jsonl"

    kpis = kpis_by_source_file.get(source_file, [])
    if kpis:
        table_columns = ["Name", "Key", "Country", "Value", "Year", "Units"]
        table_rows = []

        output_lines = []
        for kpi in kpis:
            row_values = [
                str(kpi.get("name", "")),
                str(kpi.get("key", "")),
                str(kpi.get("country", "")),
                str(kpi.get("value", "")),
                str(kpi.get("year", "")),
                str(kpi.get("units", "")),
            ]
            table_rows.append(row_values)

            output_text = (
                f"{kpi.get('name', '')} | {kpi.get('key', '')} | {kpi.get('country', '')} | "
                f"value={kpi.get('value', '')} | year={kpi.get('year', '')} | units={kpi.get('units', '')}"
            )
            output_lines.append(output_text)

        payload_row = {
            "dataset": trial_id,
            "split": split_name,
            "setup_id": setup_id,
            "example_idx": 0,
            "output": "\n".join(output_lines),
            "output_type": "table",
            "output_text": "\n".join(output_lines),
            "output_table": {
                "columns": table_columns,
                "rows": table_rows,
            },
            "metadata": {
                "source_model": kpis[0].get("source_model"),
                "source_image": kpis[0].get("source_image") if kpis else None,
                "num_kpis": len(kpis),
                "source_file": source_file,
            },
        }
        output_jsonl_path.write_text(json.dumps(payload_row, ensure_ascii=False) + "\n", encoding="utf-8")
        outputs_created += 1
    else:
        missing_kpi_json.append(source_file)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in dataset_root.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(dataset_root))

# Auto-populate dataset config with current split set.
if datasets_yml_path.exists():
    datasets_config = yaml.safe_load(datasets_yml_path.read_text(encoding="utf-8"))
    if datasets_config is None:
        datasets_config = {}
else:
    datasets_config = {}

datasets_config[trial_id] = {
    "class": "basic.HTMLDataset",
    "name": trial_id,
    "description": f"Local KPI table HTML dataset ({len(splits_created)} table splits, auto-generated).",
    "enabled": True,
    "splits": splits_created,
}

datasets_yml_path.write_text(yaml.dump(datasets_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

print(f"Trial ID: {trial_id}")
print(f"Aggregate JSON: {aggregate_kpi_json}")
print(f"Source HTML dir exists: {src.exists()}")
print(f"HTML records used: {len(records)}")
print(f"Splits created: {len(splits_created)}")
print(f"Images copied: {copied}")
print(f"HTML rewritten: {rewritten}")
print(f"Table wraps removed: {removed_table_wrap}")
print(f"Missing images: {len(missing)}")
print(f"Output files created: {outputs_created}")
print(f"Missing KPI JSON files: {len(missing_kpi_json)}")
print(f"Dataset root dir: {dataset_root}")
print(f"Outputs root dir: {outputs_root}")
print(f"ZIP: {zip_path}")
print(f"Updated dataset config: {datasets_yml_path}")
