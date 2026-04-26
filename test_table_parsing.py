#!/usr/bin/env python3
"""Test script to verify table parsing logic."""

import re
from typing import List, Dict, Any, Tuple

def test_parse_ocr_results():
    # Sample markdown from page_018/page.md
    sample_markdown = """<|ref|>title<|/ref|><|det|>[[92, 72, 208, 83]]<|/det|>
# PRODUCTION

<|ref|>table<|/ref|><|det|>[[90, 110, 490, 336]]<|/det|>

<table><tr><td>Units</td><td>2023</td><td>2022</td></tr><tr><td>Scania</td><td>102,283</td><td>88,142</td></tr><tr><td>Trucks</td><td>97,065</td><td>82,827</td></tr><tr><td>Buses</td><td>5,218</td><td>5,315</td></tr><tr><td>MAN</td><td>117,026</td><td>88,952</td></tr><tr><td>Trucks</td><td>84,696</td><td>62,009</td></tr><tr><td>Buses</td><td>5,780</td><td>4,675</td></tr><tr><td>Light Commercial Vehicles</td><td>26,551</td><td>22,268</td></tr><tr><td>Navistar</td><td>86,740</td><td>82,071</td></tr><tr><td>Trucks</td><td>73,317</td><td>69,488</td></tr><tr><td>Buses</td><td>13,423</td><td>12,583</td></tr><tr><td>Volkswagen Truck & Bus</td><td>32,515</td><td>58,647</td></tr><tr><td>Trucks</td><td>28,161</td><td>50,075</td></tr><tr><td>Buses</td><td>4,354</td><td>8,572</td></tr><tr><td>TRATON</td><td>338,564</td><td>317,812</td></tr></table>

<|ref|>title<|/ref|><|det|>[[92, 372, 366, 382]]<|/det|>
# SALES REVENUE AND EARNINGS"""

    # Parse grounded elements
    grounding_pattern = r'<.ref.>([^<]+)<./ref.><.det.>\[\[([0-9]+),\s*([0-9]+),\s*([0-9]+),\s*([0-9]+)\]\]<./det.>'
    
    grounded_elements = []
    for match in re.finditer(grounding_pattern, sample_markdown):
        element_type = match.group(1)
        
        # Only process tables, titles, sub_titles, and table_captions
        if element_type not in ['table', 'title', 'sub_title', 'table_caption']:
            continue
        
        x1, y1, x2, y2 = map(int, match.groups()[1:])
        start_pos = match.end()
        
        grounded_elements.append({
            'type': element_type,
            'bbox': [x1, y1, x2, y2],
            'start_pos': start_pos,
            'match_end': match.end()
        })
    
    print(f"Found {len(grounded_elements)} grounded elements:")
    
    # Log element types found
    element_types = {}
    for elem in grounded_elements:
        element_types[elem['type']] = element_types.get(elem['type'], 0) + 1
        print(f"  - {elem['type']}: bbox={elem['bbox']}")
    
    print(f"Element type counts: {element_types}")
    
    # Count tables specifically
    tables = []
    for i, element in enumerate(grounded_elements):
        if element['type'] == 'table':
            print(f"Processing table {len(tables)} at position {i}")
            tables.append(element)
    
    print(f"Total tables found: {len(tables)}")
    print("Expected: 1 table")
    
    # Test with multiple tables (page_012/page.md has 2 tables)
    sample_markdown_2 = """<|ref|>title<|/ref|><|det|>[[92, 72, 207, 83]]<|/det|>
# PRODUCTION

<|ref|>table<|/ref|><|det|>[[90, 110, 496, 250]]<|/det|>

<table><tr><td>Units</td><td>2023</td><td>2022</td></tr><tr><td>Transporter</td><td>81,535</td><td>67,508</td></tr></table>

<|ref|>title<|/ref|><|det|>[[506, 72, 899, 83]]<|/det|>
# VOLKSWAGEN COMMERCIAL VEHICLES BRAND

<|ref|>table<|/ref|><|det|>[[504, 110, 908, 225]]<|/det|>

<table><tr><td></td><td>2023</td><td>2022</td><td>%</td></tr><tr><td>Deliveries (thousand units)</td><td>409</td><td>329</td><td>+24.6</td></tr></table>"""

    print("\n--- Testing with 2 tables ---")
    
    grounded_elements_2 = []
    for match in re.finditer(grounding_pattern, sample_markdown_2):
        element_type = match.group(1)
        
        if element_type not in ['table', 'title', 'sub_title', 'table_caption']:
            continue
        
        x1, y1, x2, y2 = map(int, match.groups()[1:])
        start_pos = match.end()
        
        grounded_elements_2.append({
            'type': element_type,
            'bbox': [x1, y1, x2, y2],
            'start_pos': start_pos,
            'match_end': match.end()
        })
    
    element_types_2 = {}
    for elem in grounded_elements_2:
        element_types_2[elem['type']] = element_types_2.get(elem['type'], 0) + 1
        print(f"  - {elem['type']}: bbox={elem['bbox']}")
    
    print(f"Element type counts: {element_types_2}")
    
    tables_2 = []
    for i, element in enumerate(grounded_elements_2):
        if element['type'] == 'table':
            print(f"Processing table {len(tables_2)} at position {i}")
            tables_2.append(element)
    
    print(f"Total tables found: {len(tables_2)}")
    print("Expected: 2 tables")

if __name__ == "__main__":
    test_parse_ocr_results()