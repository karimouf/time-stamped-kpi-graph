#!/usr/bin/env python3
"""
LLM-Based KPI Extraction From Table Text
========================================

Runs text LLM extraction on table text (instead of table images) while keeping
the same tables.json-driven workflow used by VLM extraction.

Author: Karim Ouf
Date: March 2026
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from json_utils import clean_json_response
from logger import logger
from model import MODEL_CONFIGS, ModelManager
from validate import validate_kpis
from extract_kpis import SYSTEM_PROMPT


def _default_text_model_name() -> str:
    """Pick a reasonable default model from MODEL_CONFIGS."""
    preferred = [
        "deepseek-v2.5",
        "gemma-3-27b-it",
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
    ]
    for name in preferred:
        if name in MODEL_CONFIGS:
            return name

    for name, cfg in MODEL_CONFIGS.items():
        if cfg.get("model_type") != "ocr":
            return name

    return next(iter(MODEL_CONFIGS.keys()))


def _extract_table_text_from_entry(table_entry: Dict[str, Any]) -> str:
    """
    Resolve the text table content for one table entry.

    Priority:
    1) entry['markdown'] if available
    2) parse <table>...</table> from entry['page_markdown'] and select by table_index
    3) fallback to full page_markdown
    """
    markdown = table_entry.get("markdown")
    if markdown:
        return str(markdown)

    page_markdown = str(table_entry.get("page_markdown", ""))
    if not page_markdown:
        return ""

    tables = re.findall(r"<table>.*?</table>", page_markdown, flags=re.DOTALL | re.IGNORECASE)
    table_index = table_entry.get("table_index")

    if isinstance(table_index, int) and 0 <= table_index < len(tables):
        return tables[table_index]

    if tables:
        return tables[0]

    return page_markdown


class LLMTextKPIExtractor:
    """Text-LMM extractor that uses ModelManager.load_model + generate_text."""

    def __init__(self, temperature: float = 0.1, model_name: Optional[str] = None):
        self.model_name = model_name or _default_text_model_name()
        self.model_manager = ModelManager(temperature=temperature)

        logger.info(f"Initializing LLM Text KPI Extractor with {self.model_name}")
        logger.info("=" * 70)

    def load_model(self) -> bool:
        return self.model_manager.load_model(self.model_name)

    def unload_model(self) -> None:
        self.model_manager.unload_model()

    def extract_kpis_from_table_text(
        self,
        table_text: str,
        context: Optional[str] = None,
        title: Optional[str] = None,
        validate_results: bool = False,
        db_path: Optional[str] = None,
        page: Optional[int] = None,
        year: Optional[int] = None,
        bucket: Optional[str] = None,
        table_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.model_manager.current_llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not table_text.strip():
            return {
                "kpis": [],
                "model": self.model_name,
                "error": "No table text provided",
            }

        prompt_parts = [SYSTEM_PROMPT]
        if context:
            prompt_parts.append("\n## ADDITIONAL CONTEXT\n" + context)
        if title:
            prompt_parts.append("\n## TABLE TITLE\n" + f"Table title: {title}")
        prompt_parts.append("\n## TABLE TEXT\n" + table_text)

        full_prompt = "\n".join(prompt_parts)

        start_time = time.time()
        raw_output = self.model_manager.generate_text(full_prompt)
        inference_time = time.time() - start_time

        cleaned = clean_json_response(raw_output)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {
                "kpis": [],
                "model": self.model_name,
                "inference_time_seconds": round(inference_time, 2),
                "error": f"JSON decode error: {str(e)}",
            }

        if "kpis" not in result or not isinstance(result["kpis"], list):
            return {
                "kpis": [],
                "model": self.model_name,
                "inference_time_seconds": round(inference_time, 2),
                "error": "Invalid output structure: missing 'kpis' list",
            }

        for kpi in result["kpis"]:
            kpi["source_model"] = self.model_name
            kpi["source_type"] = "table_text"

        result["model"] = self.model_name
        result["num_kpis"] = len(result["kpis"])
        result["inference_time_seconds"] = round(inference_time, 2)

        if validate_results and db_path and result["kpis"]:
            try:
                logger.info(f"    → Running validation against database...")
                validation_output = validate_kpis(
                    kpis=result["kpis"],
                    db_path=db_path,
                    year=year,
                    page=page,
                    bucket=bucket,
                    table_idx=table_idx,
                )

                # ── Duplicate-KPI retry loop ─────────────────────────────────
                max_dup_retries = 3
                dup_retry_count = 0
                while validation_output.get("has_duplicates") and dup_retry_count < max_dup_retries:
                    dup_retry_count += 1
                    dup_list = validation_output["duplicate_kpis_list"]
                    logger.info(
                        f"    ⚠ {len(dup_list)} duplicate KPI(s) detected "
                        f"(same name/key/units/year). Retrying with disambiguation prompt "
                        f"(attempt {dup_retry_count}/{max_dup_retries})..."
                    )

                    dup_lines = []
                    for d in dup_list:
                        dup_lines.append(
                            f"  • name={d.get('name')!r}  key={d.get('key')!r}"
                            f"  units={d.get('units')!r}  year={d.get('year')}"
                            f"  row_idx={d.get('row_idx')}  col_idx={d.get('col_idx')}  value={d.get('value')}"
                        )
                    dup_summary = "\n".join(dup_lines)

                    # Compact summary of what was already extracted
                    extracted_summary = json.dumps(
                        [{"name": k.get("name"), "key": k.get("key"),
                          "units": k.get("units"), "year": k.get("year"),
                          "value": k.get("value"),
                          "row_idx": k.get("row_idx"), "col_idx": k.get("col_idx")}
                         for k in result["kpis"]],
                        indent=2
                    )

                    retry_prompt = (
                        "## CONTEXT\n"
                        f"A language model extracted {len(result['kpis'])} KPIs from a financial table "
                        f"(table_id={result.get('table_id', 'unknown')}, year={year}).\n"
                        "Here is the full extracted KPI list:\n"
                        f"{extracted_summary}\n\n"
                        "## ⚠ DUPLICATES DETECTED\n\n"
                        "The following entries share identical (name, key, units, year) — they are duplicates:\n\n"
                        + dup_summary
                        + "\n\nrow_idx/col_idx/value are provided so you can locate each row.\n\n"
                        + "## TASK\n"
                        + "Return a corrected version of the full KPI list with all duplicates resolved.\n"
                        + "For each duplicate group, differentiate the entries using these strategies:\n"
                        + "1. Check for **bold/prominent subheader rows** just above the duplicate rows — "
                        + "   include that text in the `key` to distinguish entities.\n"
                        + "2. Sequential rows with the same metric often belong to different entities "
                        + "   separated by a bold label row — use it to disambiguate.\n"
                        + "3. Use parenthetical qualifiers, footnote markers, or indentation.\n"
                        + "4. If two entries truly represent the same cell, keep only ONE.\n\n"
                        + "Respond with ONLY a JSON object: {\"kpis\": [...]} containing the complete corrected list."
                    )

                    try:
                        retry_start = time.time()
                        retry_output = self.model_manager.generate_text(retry_prompt)
                        inference_time += time.time() - retry_start

                        retry_cleaned = clean_json_response(retry_output)
                        retry_result = json.loads(retry_cleaned)

                        if "kpis" in retry_result and isinstance(retry_result["kpis"], list):
                            logger.info(
                                f"    ✓ Retry {dup_retry_count} successful — {len(retry_result['kpis'])} KPIs returned"
                            )
                            for kpi in retry_result["kpis"]:
                                kpi["source_model"] = self.model_name
                                kpi["source_type"] = "table_text"

                            result["kpis"] = retry_result["kpis"]
                            result["num_kpis"] = len(retry_result["kpis"])
                            result["inference_time_seconds"] = round(inference_time, 2)
                            result["duplicate_retry"] = dup_retry_count

                            validation_output = validate_kpis(
                                kpis=result["kpis"],
                                db_path=db_path,
                                year=year,
                                page=page,
                                bucket=bucket,
                                table_idx=table_idx,
                            )
                            if not validation_output.get("has_duplicates"):
                                logger.info(f"    ✓ No duplicates remaining after retry {dup_retry_count}")
                        else:
                            logger.warning(f"    ⚠ Duplicate retry {dup_retry_count} returned invalid JSON structure — stopping retries")
                            break
                    except Exception as retry_exc:
                        logger.warning(f"    ⚠ Duplicate retry {dup_retry_count} failed: {retry_exc} — stopping retries")
                        break

                if dup_retry_count == max_dup_retries and validation_output.get("has_duplicates"):
                    logger.warning(f"    ⚠ Duplicates persist after {max_dup_retries} retries — keeping best result")
                # ── end duplicate retry loop ──────────────────────────────────

                result["validation_statistics"] = validation_output["statistics"]
                result["validation_summary"] = {
                    "total_kpis": validation_output["statistics"]["total_kpis"],
                    "valid_kpis": validation_output["statistics"]["valid_kpis"],
                    "invalid_kpis": validation_output["statistics"]["invalid_kpis"],
                    "accuracy": validation_output["statistics"]["accuracy"],
                    "confidence_avg": validation_output["statistics"]["confidence_avg"],
                }
                result["valid_kpis"] = validation_output["valid_kpis"]
                result["invalid_kpis"] = validation_output["invalid_kpis"]
                logger.info(
                    f"    ✓ Validation complete: "
                    f"{validation_output['statistics']['valid_kpis']}/{validation_output['statistics']['total_kpis']} valid "
                    f"(Accuracy: {validation_output['statistics']['accuracy']:.1f}%)"
                )
            except Exception as exc:
                result["validation_error"] = str(exc)

        return result

    def extract_kpis_from_tables_json(
        self,
        tables_json_path: str,
        output_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        validate_results: bool = True,
        year: Optional[int] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        tables_path = Path(tables_json_path)
        if not tables_path.exists():
            return {
                "error": "Tables JSON file not found",
                "tables_json_path": str(tables_path),
            }

        with tables_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        tables = payload.get("tables", [])
        if not tables:
            return {
                "model": self.model_name,
                "total_tables": 0,
                "total_kpis": 0,
                "tables_json_path": str(tables_path),
                "results": [],
                "all_kpis": [],
            }

        if self.model_manager.current_llm is None and not self.load_model():
            return {"error": "Failed to load model", "results": []}

        logger.info("=" * 70)
        logger.info(f"EXTRACTING KPIs FROM {len(tables)} TABLES (TEXT MODE)")
        logger.info(f"Source: {tables_json_path}")
        logger.info("=" * 70)

        all_results: List[Dict[str, Any]] = []
        all_kpis: List[Dict[str, Any]] = []

        for idx, table_entry in enumerate(tables, 1):
            page_num = table_entry.get("page")
            table_idx = table_entry.get("table_index")
            title = table_entry.get("title", "")
            section_context = table_entry.get("section_context", "Unknown Section")
            pdf_filename = table_entry.get("pdf_file", "")

            table_year = table_entry.get("year")
            effective_year = table_year if table_year is not None else year

            table_bucket = table_entry.get("bucket")
            effective_bucket = table_bucket if table_bucket is not None else bucket

            table_text = _extract_table_text_from_entry(table_entry)

            logger.info(
                f"[{idx}/{len(tables)}] {pdf_filename} page={page_num} table={table_idx} title='{title}'"
            )

            context_parts = ["Financial report table from Volkswagen Group."]
            if effective_bucket:
                if str(effective_bucket).lower() == "divisions":
                    context_parts.append(
                        "This is from the DIVISIONS financial report (brand-specific performance metrics)."
                    )
                elif str(effective_bucket).lower() == "management":
                    context_parts.append(
                        "This is from the MANAGEMENT REPORT (consolidated group-level metrics)."
                    )
                else:
                    context_parts.append(f"Report type: {effective_bucket}")

            context_parts.append(f"Document section: '{section_context}'")
            if effective_year is not None:
                context_parts.append(f"Report year: {effective_year}")

            result = self.extract_kpis_from_table_text(
                table_text=table_text,
                context=" ".join(context_parts),
                title=title,
                validate_results=validate_results,
                db_path=db_path,
                page=page_num,
                year=effective_year,
                bucket=effective_bucket,
                table_idx=table_idx,
            )

            # Keep same metadata shape as VLM output for downstream compatibility.
            result["page"] = page_num
            result["table_index"] = table_idx
            result["bbox"] = table_entry.get("bbox")
            result["pdf_file"] = pdf_filename
            result["section_context"] = section_context
            result["title"] = title

            all_results.append(result)
            if "kpis" in result and isinstance(result["kpis"], list):
                all_kpis.extend(result["kpis"])

            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                pdf_name = Path(pdf_filename).stem if pdf_filename else "unknown"
                out_file = out_dir / f"{pdf_name}_page_{int(page_num):03d}_table_{int(table_idx):02d}_kpis.json"
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

        summary = {
            "model": self.model_name,
            "mode": "text_table",
            "total_tables": len(tables),
            "total_kpis": len(all_kpis),
            "extraction_date": datetime.now().isoformat(),
            "tables_json_path": str(tables_path),
            "results": all_results,
            "all_kpis": all_kpis,
        }

        if output_dir:
            summary_file = Path(output_dir) / "extraction_summary.json"
            with summary_file.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Summary saved to: {summary_file}")

        return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract KPIs from table text using text LLMs (no image inference)."
    )
    parser.add_argument(
        "--tables-json",
        type=str,
        required=True,
        help="Path to tables.json from detect_tables.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-table results and summary",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=_default_text_model_name(),
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Optional path to SQLite DB for validation",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Optional year for validation filtering",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="Optional bucket for validation filtering",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable KPI validation",
    )

    args = parser.parse_args()

    extractor = LLMTextKPIExtractor(
        temperature=args.temperature,
        model_name=args.model_name,
    )

    try:
        extractor.extract_kpis_from_tables_json(
            tables_json_path=args.tables_json,
            output_dir=args.output_dir,
            db_path=args.db_path,
            validate_results=not args.no_validation,
            year=args.year,
            bucket=args.bucket,
        )
    finally:
        extractor.unload_model()

    logger.info("Extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
