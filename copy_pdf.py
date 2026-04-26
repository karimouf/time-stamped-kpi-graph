"""Render a PDF file to one or more images."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PDF to image files.")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument(
        "output_path",
        help=(
            "Output image file path (e.g., out.png) or a directory to write pages into"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI (default: 200)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image format to use when outputting to a directory (default: png)",
    )
    return parser.parse_args()


def _iter_pages(pdf_path: Path) -> Iterable[fitz.Page]:
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            yield page
    finally:
        doc.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_pdf)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a .pdf")

    zoom = args.dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        # Single output file; write only the first page.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_page = next(_iter_pages(input_path), None)
        if first_page is None:
            raise ValueError("PDF has no pages")
        pixmap = first_page.get_pixmap(matrix=matrix)
        pixmap.save(output_path)
        if sum(1 for _ in _iter_pages(input_path)) > 1:
            print(
                "Warning: PDF has multiple pages; only the first page was rendered.",
                file=sys.stderr,
            )
        return

    # Directory output: write all pages.
    output_path.mkdir(parents=True, exist_ok=True)
    for index, page in enumerate(_iter_pages(input_path), start=1):
        image_name = f"page_{index:03d}.{args.format}"
        page_path = output_path / image_name
        pixmap = page.get_pixmap(matrix=matrix)
        pixmap.save(page_path)


if __name__ == "__main__":
    main()
