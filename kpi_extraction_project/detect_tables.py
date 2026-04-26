"""Detect tables in a PDF using OCR models (Tesseract or DeepSeek), export as JSON with titles, and save table crops."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image

from ocr_model import OCRModelManager
from logger import logger


def extract_bucket_from_filename(filename: str) -> Optional[str]:
    """
    Extract bucket from PDF filename patterns like:
    - div-divisions-vw-ar22 -> "divisions"
    - divisions-vw-ar20 -> "divisions"
    - management_report_vw_ar16 -> "management"
    - some-other-file -> None
    
    Args:
        filename: PDF filename (with or without extension)
        
    Returns:
        Extracted bucket as string, or None if no bucket pattern found
    """
    import re
    
    # Remove file extension
    name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    # Convert to lowercase for pattern matching
    name_lower = name_without_ext.lower()
    
    # Look for "division" or "divisions" in the filename
    if 'divisions' in name_lower:
        return 'divisions'
    elif 'division' in name_lower:
        return 'divisions'  # Normalize to plural form
    # Look for "management" in the filename
    elif 'management' in name_lower:
        return 'management'
    
    return None


def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Extract year from PDF filename patterns like:
    - div-divisions-vw-ar22 -> 2022
    - divisions-vw-ar20 -> 2020
    - some-file-ar23 -> 2023
    
    Args:
        filename: PDF filename (with or without extension)
        
    Returns:
        Extracted year as integer, or None if no year pattern found
    """
    import re
    
    # Remove file extension
    name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    # Pattern to match 'ar' followed by 2 digits (representing year)
    # ar22 -> 2022, ar20 -> 2020, etc.
    match = re.search(r'ar(\d{2})$', name_without_ext)
    if match:
        year_suffix = int(match.group(1))
        # Convert 2-digit year to 4-digit year
        # Assume years 00-30 are 2000s, 31-99 are 1900s
        if year_suffix <= 30:
            return 2000 + year_suffix
        else:
            return 1900 + year_suffix
    
    return None


def extract_table_of_contents(pdf_path: str, ocr_manager: OCRModelManager, max_pages: int = 2) -> Dict[int, str]:
    """
    Extract table of contents from the first 1-2 pages of a PDF and create page mappings.
    
    Uses the relative page spacing from TOC entries to map sections to actual document pages.
    For example, if TOC shows "18 Brands..., 21 Volkswagen..., 23 Škoda..." and TOC ends at page 2:
    - Page 3: "Brands and Business Fields" (first section after TOC)
    - Page 6: "Volkswagen Passenger Cars" (3 pages later, like 21-18=3)  
    - Page 8: "Škoda" (2 pages later, like 23-21=2)
    
    Args:
        pdf_path: Path to the PDF file
        ocr_manager: OCR model manager instance
        max_pages: Maximum number of pages to check for TOC (default: 2)
        
    Returns:
        Dictionary mapping page numbers to section names
        Example: {3: "Brands and Business Fields", 4: "Brands and Business Fields", 
                  6: "Volkswagen Passenger Cars", 8: "Škoda", ...}
    """
    page_sections = {}
    toc_entries = []  # List of (toc_page, section_name) tuples
    toc_end_page = max_pages  # Page where TOC ends
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"Extracting table of contents with page relationships from first {min(max_pages, total_pages)} pages...")
        
        for page_num in range(min(max_pages, total_pages)):
            page = doc.load_page(page_num)
            
            # Convert page to image for OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save temporary image
            temp_img_path = Path(f"/tmp/toc_page_{page_num}.png")
            temp_img_path.parent.mkdir(exist_ok=True, parents=True)
            with open(temp_img_path, "wb") as f:
                f.write(img_data)
            
            # Extract text using OCR
            logger.info(f"  Processing TOC page {page_num + 1}...")
            ocr_results = ocr_manager.detect_tables_with_ocr(
                str(temp_img_path),
                output_dir="/tmp/toc_analysis"
            )
            
            # Get markdown text
            text = ocr_results.get('markdown', '')
            if not text:
                continue
            
            # Parse table of contents entries with page numbers
            # Example: "18 Brands and Business Fields" or "21 Volkswagen Passenger Cars"
            lines = text.replace('\n', ' ').strip()
            
            import re
            
            # Pattern to match: number followed by section title
            # This captures entries like "18 Brands and Business Fields 21"
            pattern = r'(\d{1,3})\s+([^0-9]+?)(?=\s+\d{1,3}|$)'
            matches = re.findall(pattern, lines)
            
            # Filter out section headers (e.g., "2 Divisions") vs actual TOC entries
            # Section headers typically have:
            # 1. Small numbers (1-10) indicating section/chapter numbers
            # 2. Only one entry on the page
            # 3. Short section names (often just one word like "Divisions")
            # TOC entries typically have:
            # 1. Larger numbers (10+) indicating page numbers
            # 2. Multiple entries on the same page
            # 3. More descriptive names
            logger.info(f"    → Page text (normalized): '{lines[:100]}...'")
            logger.info(f"    → Found {len(matches)} potential TOC entries: {matches}")
            is_likely_section_header = (       
                len(matches) == 1 and  # Only one entry found
                matches[0][0].isdigit() and int(matches[0][0]) <= 10  # Small number (section number)
            )
            
            if is_likely_section_header:
                logger.info(f"    ⚠ Skipping likely section header (not TOC): '{matches[0][0]} {matches[0][1].strip()}'")
                matches = []  # Clear matches to skip this page
            
            for page_str, section_name in matches:
                try:
                    toc_page = int(page_str)
                    section_clean = section_name.strip()
                    # Remove OCR artifacts like "===============save results:==============="
                    section_clean = section_clean.replace('===============save results:===============', '').strip()
                    if section_clean and toc_page > 0:
                        toc_entries.append((toc_page, section_clean))
                        logger.info(f"    → TOC page {toc_page}: {section_clean}")
                except ValueError:
                    continue
            
            # Clean up temp file
            if temp_img_path.exists():
                temp_img_path.unlink()
        
        doc.close()
        
        if not toc_entries:
            logger.warning("No TOC entries found")
            return {}
        
        # Sort entries by TOC page number to ensure correct order
        toc_entries.sort(key=lambda x: x[0])
        
        # Map TOC entries to actual document pages
        logger.info(f"Mapping {len(toc_entries)} TOC entries to document pages...")
        logger.info(f"TOC ends at page {toc_end_page}, content starts at page {toc_end_page + 1}")
        
        # Calculate the offset: first TOC page -> first content page
        if toc_entries:
            first_toc_page, _ = toc_entries[0]
            first_content_page = toc_end_page + 1
            page_offset = first_content_page - first_toc_page
            
            logger.info(f"Page offset: TOC page {first_toc_page} -> Document page {first_content_page} (offset: {page_offset})")
            
            # Create mappings for each section
            for i, (toc_page, section_name) in enumerate(toc_entries):
                # Calculate actual start page in document
                actual_start_page = toc_page + page_offset
                
                # Calculate end page (before next section starts)
                if i + 1 < len(toc_entries):
                    next_toc_page, _ = toc_entries[i + 1]
                    actual_end_page = next_toc_page + page_offset - 1
                else:
                    # Last section goes to end of document
                    actual_end_page = total_pages
                
                # Ensure we don't go beyond document bounds
                actual_start_page = max(first_content_page, actual_start_page)
                actual_end_page = min(total_pages, actual_end_page)
                
                # Assign section to all pages in range
                if actual_start_page <= actual_end_page:
                    for page_num in range(actual_start_page, actual_end_page + 1):
                        page_sections[page_num] = section_name
                    
                    logger.info(f"    → Pages {actual_start_page}-{actual_end_page}: {section_name}")
                else:
                    logger.warning(f"    → Invalid range for {section_name}: {actual_start_page}-{actual_end_page}")
        
        logger.info(f"TOC mapping complete: {len(page_sections)} pages mapped to {len(set(page_sections.values()))} unique sections")
        
        # Fill any gaps with "Unknown Section"
        for page_num in range(toc_end_page + 1, total_pages + 1):
            if page_num not in page_sections:
                page_sections[page_num] = "Unknown Section (use the page as context for table detection)"
        
    except Exception as e:
        logger.error(f"Error extracting table of contents with page relationships: {e}")
        import traceback
        traceback.print_exc()
    
    return page_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect tables in PDFs using OCR models and save table crops with titles."
    )
    parser.add_argument("input_dir", help="Path to the directory containing PDF files")
    parser.add_argument(
        "output_json",
        help="Path to the output JSON file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save table images and page markdown"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rendering pages (default: 300 for high quality)",
    )
    parser.add_argument(
        "--model-name",
        default="tesseract",
        help="OCR model to use (default: tesseract)",
    )
    parser.add_argument(
        "--page-split",
        action="store_true",
        default=False,
        help="Split pages into header (10%%) and content (90%%) sections for targeted processing",
    )
    parser.add_argument(
        "--header-ratio",
        type=float,
        default=0.10,
        help="Ratio of page height to use for header section (default: 0.10 = 10%%)",
    )
    return parser.parse_args()


def parse_ocr_results(ocr_output: dict) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    """
    Parse DeepSeek OCR output to extract table bounding boxes and titles.
    
    The OCR output format uses grounding tags:
    - <|ref|>title<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>Title Text
    - <|ref|>table<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|><table>...</table>
    
    The output also contains 'image size: (width, height)' which is the actual
    resolution the OCR model processed at (after BASE/PATCHES processing).
    
    Args:
        ocr_output: Dictionary containing OCR results from DeepSeek OCR
        
    Returns:
        Tuple of (tables, ocr_size) where:
        - tables: List of dictionaries with table information (bbox, title, markdown)
        - ocr_size: (width, height) tuple of OCR's processed image size
    """
    logger.info("    → Parsing OCR results to extract tables...")
    tables = []
    ocr_size = (1000, 1000)  # Default fallback
    
    # Initialize tracking of used elements to prevent sharing between tables
    used_elements = set()
    
    if not isinstance(ocr_output, dict) or 'markdown' not in ocr_output:
        logger.warning("    ✗ Invalid OCR output format (missing 'markdown' key)")
        return tables, ocr_size
    
    markdown_text = ocr_output['markdown']
    logger.info(f"    → Markdown text length: {len(markdown_text)} characters")
    

    # Pattern to match grounding tags: <|ref|>TYPE<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    # Using . wildcard to match the pipe character
    grounding_pattern = r'<.ref.>([^<]+)<./ref.><.det.>\[\[([0-9]+),\s*([0-9]+),\s*([0-9]+),\s*([0-9]+)\]\]<./det.>'
    
    # Find all grounded elements (tables, titles, sub_titles, and table_captions)
    grounded_elements = []
    for match in re.finditer(grounding_pattern, markdown_text):
        element_type = match.group(1)
        
        # Only process tables, titles, sub_titles, table_captions, and table_footnotes (ignore text, images, etc.)
        if element_type not in ['table', 'title', 'sub_title', 'table_caption', 'table_footnote']:
            continue
        
        x1, y1, x2, y2 = map(int, match.groups()[1:])
        start_pos = match.end()
        
        grounded_elements.append({
            'type': element_type,
            'bbox': [x1, y1, x2, y2],
            'start_pos': start_pos,
            'match_end': match.end()
        })
    
    logger.info(f"    → Found {len(grounded_elements)} grounded elements")
    
    # Log element types found
    element_types = {}
    for elem in grounded_elements:
        element_types[elem['type']] = element_types.get(elem['type'], 0) + 1
    logger.info(f"    → Element types: {element_types}")
    
    # Extract tables and match them with preceding titles/sub_titles or captions before/after the table
    # Each table will be processed independently to ensure separate screenshots
    for i, element in enumerate(grounded_elements):
        if element['type'] == 'table':
            logger.info(f"    → Processing table {len(tables)} at position {i}")
            logger.info(f"      BBox: {element['bbox']}")
            
            # Extract table content (starts after the grounding tag)
            start_pos = element['start_pos']
            
            # Find the end of the table content (next grounding tag or end of string)
            if i + 1 < len(grounded_elements):
                end_pos = markdown_text.rfind('<|ref|>', 0, grounded_elements[i + 1]['start_pos'])
                if end_pos == -1:
                    end_pos = grounded_elements[i + 1]['start_pos']
            else:
                end_pos = len(markdown_text)
            
            table_content = markdown_text[start_pos:end_pos].strip()
            logger.info(f"      Table content length: {len(table_content)} characters")
            
            # Extract table HTML/markdown
            table_match = re.search(r'<table>.*?</table>', table_content, re.DOTALL)
            table_markdown = table_match.group(0) if table_match else table_content
            
            if table_match:
                logger.info(f"      ✓ Found HTML table in content")
            else:
                logger.warning(f"      ⚠ No HTML table found, using raw content")
            
            # Find elements related to this table based on BBox proximity
            title = f"Table {len(tables)}"
            title_found = False
            title_bbox = None
            subtitle_bbox = None
            caption_bbox = None
            footnote_bbox = None
            
            # Get table's bounding box for distance calculations
            table_x1, table_y1, table_x2, table_y2 = element['bbox']
            table_center_x = (table_x1 + table_x2) / 2
            table_top_y = table_y1
            table_bottom_y = table_y2
            
            # Define maximum reasonable distances for title/footnote association
            MAX_TITLE_DISTANCE = 200  # Maximum distance for title association (in OCR coordinates)
            MAX_FOOTNOTE_DISTANCE = 150  # Maximum distance for footnote association
            
            # Find the nearest title/subtitle/caption element above or below the table
            nearest_title_element = None
            nearest_title_distance = float('inf')
            
            # Look through all elements to find titles/subtitles/captions
            for j, other_element in enumerate(grounded_elements):
                if j == i:  # Skip the current table
                    continue
                    
                if j in used_elements:  # Skip already used elements
                    continue
                    
                if other_element['type'] in ['title', 'sub_title', 'table_caption']:
                    other_x1, other_y1, other_x2, other_y2 = other_element['bbox']
                    other_center_x = (other_x1 + other_x2) / 2
                    other_bottom_y = other_y2
                    other_top_y = other_y1
                    
                    # Check if this element is above the table (other_bottom_y <= table_top_y)
                    if other_bottom_y <= table_top_y:
                        # Apply weighted distance metric: D = d_v + 0.5 × d_h
                        # This prioritizes vertical proximity while considering horizontal alignment
                        vertical_distance = table_top_y - other_bottom_y
                        horizontal_distance = abs(table_center_x - other_center_x)
                        
                        # Weighted sum: vertical distance with full weight, horizontal offset reduced
                        total_distance = vertical_distance + (horizontal_distance * 0.5)
                        
                        # Only consider if within reasonable distance
                        if total_distance <= MAX_TITLE_DISTANCE and total_distance < nearest_title_distance:
                            nearest_title_distance = total_distance
                            nearest_title_element = (j, other_element)
                            logger.info(f"      → Found candidate {other_element['type']} at position {j} (above), distance: {total_distance:.1f}")
                    
                    # Check if this element is below the table (only for table_caption)
                    elif other_element['type'] == 'table_caption' and other_top_y >= table_bottom_y:
                        # Apply weighted distance metric: D = d_v + 0.5 × d_h
                        # Vertical proximity is weighted more heavily for semantic relevance
                        vertical_distance = other_top_y - table_bottom_y
                        horizontal_distance = abs(table_center_x - other_center_x)
                        
                        # Weighted sum: vertical distance prioritized over horizontal offset
                        total_distance = vertical_distance + (horizontal_distance * 0.5)
                        
                        # Only consider if within reasonable distance
                        if total_distance <= MAX_TITLE_DISTANCE and total_distance < nearest_title_distance:
                            nearest_title_distance = total_distance
                            nearest_title_element = (j, other_element)
                            logger.info(f"      → Found candidate {other_element['type']} at position {j} (below), distance: {total_distance:.1f}")
            
            # Process the nearest title element if found and within reasonable distance
            if nearest_title_element and nearest_title_distance <= MAX_TITLE_DISTANCE:
                j, title_element = nearest_title_element
                logger.info(f"      → Using nearest {title_element['type']} at position {j} (distance: {nearest_title_distance:.1f})")
                
                # Mark this element as used
                used_elements.add(j)
                
                # Extract text from the nearest title element
                text_start = title_element['start_pos']
                
                # Find the end position for text extraction
                if j + 1 < len(grounded_elements):
                    next_element_start = grounded_elements[j + 1]['start_pos']
                    text_end = markdown_text.rfind('<|ref|>', 0, next_element_start)
                    if text_end == -1:
                        text_end = next_element_start
                else:
                    text_end = len(markdown_text)
                
                text_content = markdown_text[text_start:text_end].strip()
                logger.info(f"      → Raw {title_element['type']} text: '{text_content[:100]}'")    
                
                # Clean up text (remove markdown heading markers)
                text_content = re.sub(r'^#+\s*', '', text_content)
                text_content = text_content.split('\n')[0].strip()
                
                if text_content:
                    title = text_content
                    title_found = True
                    
                    # Store the appropriate bbox based on element type
                    if title_element['type'] == 'title':
                        title_bbox = title_element['bbox']
                        logger.info(f"      ✓ Using nearest title: '{title}' with bbox: {title_bbox}")
                    elif title_element['type'] == 'sub_title':
                        subtitle_bbox = title_element['bbox']
                        logger.info(f"      ✓ Using nearest subtitle as title with bbox: {subtitle_bbox}")
                    elif title_element['type'] == 'table_caption':
                        caption_bbox = title_element['bbox']
                        logger.info(f"      ✓ Using nearest caption as title with bbox: {caption_bbox}")
            
            if not title_found:
                # Threshold enforcement prevents spurious associations - when no element is found
                # within the defined distance, a default title is used, maintaining data integrity
                # and preventing false header attributions from unrelated document sections
                logger.info(f"      ⚠ No title/subtitle/caption found within threshold ({MAX_TITLE_DISTANCE} pixels), using default title: '{title}'")
            

            # Find the nearest table_footnote element below the table using the same
            # proximity-based association metric as applied to headers above
            nearest_footnote_element = None
            nearest_footnote_distance = float('inf')
            
            # Look through all elements to find footnotes
            for j, other_element in enumerate(grounded_elements):
                if j == i:  # Skip the current table
                    continue
                    
                if j in used_elements:  # Skip already used elements
                    continue
                    
                if other_element['type'] == 'table_footnote':
                    other_x1, other_y1, other_x2, other_y2 = other_element['bbox']
                    other_center_x = (other_x1 + other_x2) / 2
                    other_top_y = other_y1
                    
                    # Check if this element is below the table (other_top_y >= table_bottom_y)
                    if other_top_y >= table_bottom_y:
                        # Apply weighted distance metric: D = d_v + 0.5 × d_h
                        # Ensures footnotes are only associated if spatially proximate and
                        # vertically aligned (below the table)
                        vertical_distance = other_top_y - table_bottom_y
                        horizontal_distance = abs(table_center_x - other_center_x)
                        
                        # Weighted sum: vertical distance with full weight, horizontal offset reduced
                        total_distance = vertical_distance + (horizontal_distance * 0.5)
                        
                        # Only consider if within reasonable distance
                        if total_distance <= MAX_FOOTNOTE_DISTANCE and total_distance < nearest_footnote_distance:
                            nearest_footnote_distance = total_distance
                            nearest_footnote_element = (j, other_element)
                            logger.info(f"      → Found candidate table_footnote at position {j}, distance: {total_distance:.1f}")
            
            # Process the nearest footnote element if found and within reasonable distance
            if nearest_footnote_element and nearest_footnote_distance <= MAX_FOOTNOTE_DISTANCE:
                j, footnote_element = nearest_footnote_element
                footnote_bbox = footnote_element['bbox']
                
                # Mark this element as used
                used_elements.add(j)
                
                logger.info(f"      ✓ Using nearest table_footnote at position {j} (distance: {nearest_footnote_distance:.1f}) with bbox: {footnote_bbox}")
            else:
                # No footnote within threshold - prevents false associations with distant or
                # unrelated footnotes from other tables or page sections
                logger.info(f"      ⚠ No table_footnote found within threshold ({MAX_FOOTNOTE_DISTANCE} pixels) below this table")
            
            if not title_found:
                logger.warning(f"      ⚠ No title found, using default: '{title}'")
            
            tables.append({
                'title': title,
                'bbox': element['bbox'],
                'title_bbox': title_bbox,
                'subtitle_bbox': subtitle_bbox,
                'caption_bbox': caption_bbox,
                'footnote_bbox': footnote_bbox,
                'markdown': table_markdown,
                'table_index': len(tables)  # 0-indexed
            })
            
            logger.info(f"      ✓ Added table {len(tables) - 1}: '{title}'")
    
    logger.info(f"    ✓ Parsing complete: {len(tables)} table(s) extracted")
    return tables, ocr_size


def convert_pdf_page_to_image(page: fitz.Page, dpi: int = 300, split_page: bool = False, header_ratio: float = 0.10) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    Convert a PDF page to PIL Images. Can either return full page or split into header/content sections.
    
    Page splitting helps OCR detection by:
    1. Isolating section headers for targeted section name extraction
    2. Providing clean content area for table detection without header interference
    3. Allowing different OCR processing strategies for different content types
    4. Header section captures section names and table headers when present
    
    Args:
        page: PyMuPDF page object
        dpi: Resolution for rendering (default: 300 for high quality)
        split_page: If True, split page into header and content sections
        header_ratio: Fraction of page height to use for header section (default: 0.10 = 10%)
        
    Returns:
        If split_page=False: (full_image, None)
        If split_page=True: (header_image, content_image)
    """
    # Render the page at specified DPI
    zoom = dpi / 72  # 72 DPI is PDF default
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)
    
    # Convert to PIL Image
    full_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    
    if not split_page:
        return full_image, None
    
    # Split the page
    split_y = int(full_image.height * header_ratio)
    
    # Create header section (top part - contains section headers and sometimes table headers)
    header_image = full_image.crop((0, 0, full_image.width, split_y))
    
    # Create content section (bottom part - contains main tables and content)
    content_image = full_image.crop((0, split_y, full_image.width, full_image.height))
    
    return header_image, content_image
    
    return padded_img


def crop_table_from_image(image: Image.Image, bbox: List[float], scale: float = 1.0) -> Image.Image:
    """
    Crop a table region from an image using bounding box coordinates.
    
    Args:
        image: PIL Image
        bbox: Bounding box [x1, y1, x2, y2] in PDF coordinates
        scale: Scale factor to convert PDF coordinates to image coordinates
        
    Returns:
        Cropped PIL Image
    """
    if bbox is None:
        return image
    
    # Scale bbox coordinates to image size
    x1, y1, x2, y2 = bbox
    x1_img = int(x1 * scale)
    y1_img = int(y1 * scale)
    x2_img = int(x2 * scale)
    y2_img = int(y2 * scale)
    
    # Crop the image
    cropped = image.crop((x1_img, y1_img, x2_img, y2_img))
    return cropped


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_json)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError("Input path must be a directory")
    
    # Find all PDF files in the directory
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {input_dir}")
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info(f"Configuration: DPI={args.dpi}, Model={args.model_name}, Page Split={'Enabled' if args.page_split else 'Disabled'}")
    if args.page_split:
        logger.info(f"  → Header ratio: {args.header_ratio*100:.0f}% (captures section headers and table headers)")
        logger.info(f"  → Content ratio: {(1-args.header_ratio)*100:.0f}% (processed for table detection)")
    logger.info(f"  → Page splitting isolates headers from content for targeted processing")
    for pdf_file in pdf_files:
        logger.info(f"  - {pdf_file.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OCR model manager and load model
    logger.info("Initializing DeepSeek OCR model...")
    ocr_manager = OCRModelManager()
    
    if not ocr_manager.load_model(args.model_name):
        raise RuntimeError(f"Failed to load OCR model: {args.model_name}")
    
    results: List[Dict[str, Any]] = []
    
    # Extract table of contents from each PDF for section context
    pdf_toc_mappings = {}
    for pdf_file in pdf_files:
        logger.info(f"\nExtracting sequential table of contents from: {pdf_file.name}")
        toc_mapping = extract_table_of_contents(str(pdf_file), ocr_manager, max_pages=2)
        pdf_toc_mappings[pdf_file.name] = toc_mapping
        
        if toc_mapping:
            logger.info(f"  Sequential mapping created for {len(toc_mapping)} pages:")
            # Show first few and last few mappings
            sorted_pages = sorted(toc_mapping.keys())
            sample_pages = sorted_pages[:3] + (['...'] if len(sorted_pages) > 6 else []) + sorted_pages[-3:]
            for page_num in sample_pages:
                if page_num == '...':
                    logger.info(f"    ...")
                else:
                    section_name = toc_mapping[page_num]
                    logger.info(f"    Page {page_num}: {section_name}")
        else:
            logger.warning(f"  No table of contents found in {pdf_file.name} - pages will have 'Unknown Section'")

    try:
        # Process each PDF file
        for pdf_index, pdf_file in enumerate(pdf_files, start=1):
            logger.info(f"\nProcessing PDF {pdf_index}/{len(pdf_files)}: {pdf_file.name}")
            
            with fitz.open(pdf_file) as doc:
                for page_index, page in enumerate(doc, start=1):
                    logger.info(f"Processing page {page_index}...")
                    
                    # Get section context from table of contents
                    toc_mapping = pdf_toc_mappings.get(pdf_file.name, {})
                    section_context = toc_mapping.get(page_index, "Unknown Section (use the page as context for table detection)")
                    logger.info(f"  Section context: {section_context}")
                    
                    # Create page directory for this specific PDF
                    pdf_name_clean = pdf_file.stem  # Get filename without extension
                    page_dir = output_dir / pdf_name_clean / f"page_{page_index:03d}"
                    page_dir.mkdir(parents=True, exist_ok=True)
                    
                    if args.page_split:
                        # Split page into header and content sections
                        header_image, content_image = convert_pdf_page_to_image(
                            page, dpi=args.dpi, split_page=True, header_ratio=args.header_ratio
                        )
                        
                        # Get dimensions
                        header_w, header_h = header_image.size
                        content_w, content_h = content_image.size
                        total_height = header_h + content_h
                        
                        logger.info(f"  Split into: Header {header_w}x{header_h}px ({args.header_ratio*100:.0f}%), Content {content_w}x{content_h}px ({(1-args.header_ratio)*100:.0f}%)")
                        
                        # Process content section for table detection (main processing)
                        page_image = content_image
                        img_width, img_height = content_w, content_h
                        original_height = content_h  # Content section is our main processing target
                        
                        # Save header section for reference/section name extraction
                        header_path = page_dir / "page_header.png"
                        header_image.save(header_path)
                        logger.info(f"    → Header section saved: {header_path}")
                        
                        # Save content section
                        content_path = page_dir / "page_content.png"
                        content_image.save(content_path)
                        logger.info(f"    → Content section saved: {content_path}")
                        
                    else:
                        # Process full page without splitting
                        page_image, _ = convert_pdf_page_to_image(page, dpi=args.dpi, split_page=False)
                        img_width, img_height = page_image.size
                        original_height = img_height
                        
                        # Save full page
                        page_path = page_dir / "page.png"
                        page_image.save(page_path)
                        logger.info(f"  Rendered page at {img_width}x{img_height}px ({args.dpi} DPI)")
                    
                    if args.page_split:
                        # Process both header and content sections for table detection
                        
                        # 1. Process header section (10% - section headers and table headers)
                        logger.info(f"  → Processing header section for tables...")
                        header_ocr_results = ocr_manager.detect_tables_with_ocr(
                            str(header_path),
                            output_dir=str(page_dir / "header_analysis")
                        )
                        header_tables, header_ocr_size = parse_ocr_results(header_ocr_results)
                        logger.info(f"  → Header section: {len(header_tables)} table(s) found")
                        
                        # 2. Process content section (90% - main tables and data)
                        logger.info(f"  → Processing content section for tables...")
                        content_ocr_results = ocr_manager.detect_tables_with_ocr(
                            str(content_path),
                            output_dir=str(page_dir / "content_analysis")
                        )
                        content_tables, content_ocr_size = parse_ocr_results(content_ocr_results)
                        logger.info(f"  → Content section: {len(content_tables)} table(s) found")
                        
                        # Combine results from both sections
                        all_page_tables = []
                        
                        # Add header tables with proper coordinate adjustment
                        for table_info in header_tables:
                            table_info['section'] = 'header'
                            table_info['section_ratio'] = args.header_ratio
                            all_page_tables.append(table_info)
                        
                        # Add content tables with proper coordinate adjustment
                        for table_info in content_tables:
                            table_info['section'] = 'content'
                            table_info['section_ratio'] = 1 - args.header_ratio
                            # Adjust coordinates to full page reference
                            if 'bbox' in table_info and table_info['bbox']:
                                bbox = table_info['bbox']
                                # Add header height offset to content section coordinates
                                header_offset_ratio = args.header_ratio
                                adjusted_bbox = [
                                    bbox[0],
                                    bbox[1] + header_offset_ratio,  # Add header ratio offset
                                    bbox[2], 
                                    bbox[3] + header_offset_ratio   # Add header ratio offset
                                ]
                                table_info['bbox_full_page'] = adjusted_bbox
                            all_page_tables.append(table_info)
                        
                        detected_tables = all_page_tables
                        total_tables = len(header_tables) + len(content_tables)
                        
                        logger.info(f"  → Total combined: {total_tables} table(s) ({len(header_tables)} from header + {len(content_tables)} from content)")
                        
                        # Use content section dimensions for scaling (main processing target)
                        page_image = content_image
                        img_width, img_height = content_w, content_h
                        ocr_size = content_ocr_size  # Use content OCR size for scaling
                        
                        # Combine markdown from both sections
                        header_markdown = header_ocr_results.get('markdown', '')
                        content_markdown = content_ocr_results.get('markdown', '')
                        page_markdown_text = f"{header_markdown}\n\n{content_markdown}" if header_markdown else content_markdown
                        
                    else:
                        # Process full page without splitting
                        logger.info(f"  → Processing full page for tables...")
                        ocr_results = ocr_manager.detect_tables_with_ocr(
                            str(page_path),
                            output_dir=str(page_dir)
                        )
                        detected_tables, ocr_size = parse_ocr_results(ocr_results)
                        page_image = page_image  # Already set above
                        page_markdown_text = ocr_results.get('markdown', '')

                    if not detected_tables:
                        logger.info(f"  No tables detected on page {page_index}")
                        continue
                    logger.info(f"  Found {len(detected_tables)} table(s)")
                    
                    # Calculate scale from OCR size to processed image size
                    ocr_width, ocr_height = ocr_size
                    scale_x = img_width / ocr_width
                    scale_y = img_height / ocr_height
                    
                    if args.page_split:
                        logger.info(f"  Scale: OCR {ocr_width}x{ocr_height} → Content {img_width}x{img_height} (x={scale_x:.3f}, y={scale_y:.3f})")
                        # For split pages, we need to track the header offset for coordinate adjustment
                        header_offset = int(total_height * args.header_ratio)
                    else:
                        logger.info(f"  Scale: OCR {ocr_width}x{ocr_height} → Full {img_width}x{img_height} (x={scale_x:.3f}, y={scale_y:.3f})")
                    
                    # Save full page markdown
                    page_markdown_path = page_dir / "page.md"
                    if page_markdown_text:
                        page_markdown_path.write_text(page_markdown_text, encoding="utf-8")
                    
                    # Save section context from table of contents
                    section_context_path = page_dir / "section_context.txt"
                    section_context_path.write_text(f"Page {page_index}: {section_context}", encoding="utf-8")
                    
                    # Process each detected table from both sections
                    for table_info in detected_tables:
                        table_idx = table_info.get('table_index', 0)
                        table_section = table_info.get('section', 'unknown')
                        bbox = table_info.get('bbox', [])
                        title_bbox = table_info.get('title_bbox')
                        subtitle_bbox = table_info.get('subtitle_bbox')
                        caption_bbox = table_info.get('caption_bbox')
                        footnote_bbox = table_info.get('footnote_bbox')
                        title = table_info.get('title', f"Table {table_idx}")
                        table_markdown = table_info.get('markdown', '')
                        
                        logger.info(f"  Table {table_idx} ({table_section} section): {title}")
                        
                        # Use appropriate image and scaling based on section
                        if args.page_split:
                            if table_section == 'header':
                                # Use header image and dimensions for cropping
                                source_image = header_image
                                source_width, source_height = header_w, header_h
                                # Use header OCR size for scaling
                                header_ocr_width, header_ocr_height = header_ocr_size
                                section_scale_x = source_width / header_ocr_width
                                section_scale_y = source_height / header_ocr_height
                                coordinate_offset = 0  # No offset needed for header
                            else:
                                # Use content image and dimensions for cropping
                                source_image = content_image
                                source_width, source_height = content_w, content_h
                                # Use content OCR size for scaling
                                content_ocr_width, content_ocr_height = content_ocr_size
                                section_scale_x = source_width / content_ocr_width
                                section_scale_y = source_height / content_ocr_height
                                coordinate_offset = header_h  # Offset for full page coordinates
                        else:
                            # Full page processing
                            source_image = page_image
                            source_width, source_height = img_width, img_height
                            section_scale_x = scale_x
                            section_scale_y = scale_y
                            coordinate_offset = 0
                        
                        if bbox:
                            logger.info(f"    Table BBox ({table_section}): {bbox}")
                        if title_bbox:
                            logger.info(f"    Title BBox ({table_section}): {title_bbox}")
                        if subtitle_bbox:
                            logger.info(f"    Subtitle BBox ({table_section}): {subtitle_bbox}")
                        if caption_bbox:
                            logger.info(f"    Caption BBox ({table_section}): {caption_bbox}")
                        if footnote_bbox:
                            logger.info(f"    Footnote BBox ({table_section}): {footnote_bbox}")
                            logger.info(f"    Footnote BBox (OCR {ocr_width}x{ocr_height}): {footnote_bbox}")
                        
                        # Crop and save table image from appropriate section
                        image_filename = f"table_{table_idx:02d}.png"
                        image_path = page_dir / image_filename
                        
                        try:
                            if bbox and len(bbox) == 4:
                                # Calculate the combined bounding box (only elements directly above + table + footnote)
                                combined_bbox = bbox.copy()
                                components = ["table"]
                                
                                # Include the one element directly above (title, subtitle, or caption)
                                # Only one of these will be set since we only look at the element directly above
                                if title_bbox and len(title_bbox) == 4:
                                    combined_bbox[0] = min(combined_bbox[0], title_bbox[0])  # x1
                                    combined_bbox[1] = min(combined_bbox[1], title_bbox[1])  # y1 (directly above)
                                    combined_bbox[2] = max(combined_bbox[2], title_bbox[2])  # x2
                                    combined_bbox[3] = max(combined_bbox[3], title_bbox[3])  # y2
                                    components.append("title")
                                elif subtitle_bbox and len(subtitle_bbox) == 4:
                                    combined_bbox[0] = min(combined_bbox[0], subtitle_bbox[0])  # x1
                                    combined_bbox[1] = min(combined_bbox[1], subtitle_bbox[1])  # y1 (directly above)
                                    combined_bbox[2] = max(combined_bbox[2], subtitle_bbox[2])  # x2
                                    combined_bbox[3] = max(combined_bbox[3], subtitle_bbox[3])  # y2
                                    components.append("subtitle")
                                elif caption_bbox and len(caption_bbox) == 4:
                                    combined_bbox[0] = min(combined_bbox[0], caption_bbox[0])  # x1
                                    combined_bbox[1] = min(combined_bbox[1], caption_bbox[1])  # y1 (directly above)
                                    combined_bbox[2] = max(combined_bbox[2], caption_bbox[2])  # x2
                                    combined_bbox[3] = max(combined_bbox[3], caption_bbox[3])  # y2
                                    components.append("caption")
                                
                                # Expand to include footnote (if found below the table)
                                if footnote_bbox and len(footnote_bbox) == 4:
                                    combined_bbox[0] = min(combined_bbox[0], footnote_bbox[0])  # x1
                                    combined_bbox[1] = min(combined_bbox[1], footnote_bbox[1])  # y1
                                    combined_bbox[2] = max(combined_bbox[2], footnote_bbox[2])  # x2
                                    combined_bbox[3] = max(combined_bbox[3], footnote_bbox[3])  # y2 (footnote below)
                                    components.append("footnote")
                                
                                logger.info(f"    Combined BBox ({'+'.join(components)}): {combined_bbox}")
                                
                                # Scale bbox using section-specific scaling
                                x1 = int(combined_bbox[0] * section_scale_x)
                                y1 = int(combined_bbox[1] * section_scale_y)
                                x2 = int(combined_bbox[2] * section_scale_x)
                                y2 = int(combined_bbox[3] * section_scale_y)
                                
                                logger.info(f"    BBox (scaled to {table_section}): [{x1}, {y1}, {x2}, {y2}]")
                                
                                # Calculate full page coordinates
                                if args.page_split:
                                    if table_section == 'header':
                                        # Header coordinates are already correct for full page
                                        full_page_bbox = [x1, y1, x2, y2]
                                    else:
                                        # Content coordinates need header offset added
                                        full_page_bbox = [x1, y1 + coordinate_offset, x2, y2 + coordinate_offset]
                                    logger.info(f"    BBox (full page): {full_page_bbox}")
                                else:
                                    full_page_bbox = [x1, y1, x2, y2]
                                
                                # Ensure coordinates are within section bounds
                                x1 = max(0, min(x1, source_width))
                                y1 = max(0, min(y1, source_height))
                                x2 = max(0, min(x2, source_width))
                                y2 = max(0, min(y2, source_height))
                                
                                # Check if valid crop region
                                if x2 > x1 and y2 > y1:
                                    table_image = source_image.crop((x1, y1, x2, y2))
                                    table_image.save(image_path)
                                else:
                                    logger.warning(f"    Invalid bbox coordinates, saving full section")
                                    source_image.save(image_path)
                            else:
                                # If no bbox, save full section (fallback)
                                source_image.save(image_path)
                                logger.info(f"    Warning: No bbox provided, saved full {table_section} section")
                        except Exception as e:
                            logger.error(f"    Error saving table image: {e}")
                            try:
                                source_image.save(image_path)
                                logger.info(f"    Saved full {table_section} section as fallback")
                            except Exception as e2:
                                logger.error(f"    Failed to save even section image: {e2}")
                                continue
                        
                        # Extract year and bucket from PDF filename
                        extracted_year = extract_year_from_filename(pdf_file.name)
                        extracted_bucket = extract_bucket_from_filename(pdf_file.name)
                        
                        # Add to results with proper coordinate handling for both sections
                        table_result = {
                            "pdf_file": pdf_file.name,
                            "page": page_index,
                            "table_index": table_idx,
                            "title": title,
                            "section_context": section_context,  # From table of contents
                            "bbox": full_page_bbox,  # Full page coordinates
                            "processing_method": f"split_page_{table_section}" if args.page_split else "full_page",
                            "header_ratio": args.header_ratio if args.page_split else None,
                            "year": extracted_year,
                            "bucket": extracted_bucket,
                            "page_markdown": page_markdown_text,
                            "image_path": str(image_path) if image_path is not None else None,
                        }
                        
                        results.append(table_result)
        
        # Save results to tables.json in the output directory
        tables_json_path = output_dir / "tables.json"
        tables_json_path.write_text(
            json.dumps({"tables": results}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # Also save to the specified output path if different
        if output_path != tables_json_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"tables": results}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        # Count total pages processed
        total_pages = sum(1 for result in results)
        unique_pages = len(set((result["pdf_file"], result["page"]) for result in results))
        
        logger.info(f"\n✓ Detection complete! Found {len(results)} table(s) across {len(pdf_files)} PDF(s) and {unique_pages} page(s)")
        logger.info(f"  Processed PDFs: {[pdf.name for pdf in pdf_files]}")
        logger.info(f"  Results saved to: {tables_json_path}")
        if output_path != tables_json_path:
            logger.info(f"  Also saved to: {output_path}")
        
    finally:
        # Unload model to free GPU memory
        logger.info("\nUnloading OCR model...")
        ocr_manager.unload_model()
        logger.info("✓ Model unloaded")


if __name__ == "__main__":
    main()
