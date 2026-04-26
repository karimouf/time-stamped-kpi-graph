import os
import json
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_JSON = os.path.join(BASE_DIR, 'random_15_tables.json')
SRC_ROOT = os.path.join(BASE_DIR, 'data', 'detected_tables_test_13')

with open(GT_JSON, 'r', encoding='utf-8') as f:
    gt = json.load(f)

for entry in gt['selected_tables']:
    table_json = entry['source_file']
    table_name = table_json.replace('.json', '')
    # Remove trailing _kpis if present for image lookup
    table_name_no_kpis = table_name
    if table_name.endswith('_kpis'):
        table_name_no_kpis = table_name[:-5]
    # Parse for doc, page, table
    parts = table_name.split('_page_')
    doc = parts[0]
    page_and_table = parts[1]
    page_num = page_and_table.split('_table_')[0]
    table_idx = page_and_table.split('_table_')[1]
    # Folder to create
    out_dir = os.path.join(BASE_DIR, table_name)
    os.makedirs(out_dir, exist_ok=True)
    # Source folder for images
    src_folder = os.path.join(SRC_ROOT, doc, f'page_{page_num}')
    # Copy page.png
    src_page = os.path.join(src_folder, 'page.png')
    dst_page = os.path.join(out_dir, 'page.png')
    if os.path.exists(src_page):
        shutil.copy2(src_page, dst_page)
    # Copy result_with_boxes.jpg
    src_boxes = os.path.join(src_folder, 'result_with_boxes.jpg')
    dst_boxes = os.path.join(out_dir, 'result_with_boxes.jpg')
    if os.path.exists(src_boxes):
        shutil.copy2(src_boxes, dst_boxes)
    # Copy table image
    # Use table image name without _kpis
    src_table = os.path.join(src_folder, f'table_{table_idx}.png')
    dst_table = os.path.join(out_dir, f'table_{table_idx}.png')
    if os.path.exists(src_table):
        shutil.copy2(src_table, dst_table)
    else:
        # Try fallback: if table_idx has _kpis, strip it
        if table_idx.endswith('_kpis'):
            fallback_idx = table_idx.replace('_kpis', '')
            fallback_src_table = os.path.join(src_folder, f'table_{fallback_idx}.png')
            if os.path.exists(fallback_src_table):
                shutil.copy2(fallback_src_table, dst_table)
                print(f'Used fallback table image for {table_name}: {fallback_src_table}')
            else:
                print(f'WARNING: Table image not found for {table_name}. Searched: {src_table} and {fallback_src_table}')
                print(f'  Directory contents: {os.listdir(src_folder)}')
        else:
            print(f'WARNING: Table image not found for {table_name}. Searched: {src_table}')
            print(f'  Directory contents: {os.listdir(src_folder)}')
        # Copy extracted kpis JSON file
        extracted_json_src = os.path.join(BASE_DIR, 'data', 'output', 'trial-21', 'output', 'vlm_qwen_32b', table_json)
        extracted_json_dst = os.path.join(out_dir, table_json)
        if os.path.exists(extracted_json_src):
            shutil.copy2(extracted_json_src, extracted_json_dst)
        else:
            print(f'WARNING: Extracted KPIs JSON not found for {table_name}. Searched: {extracted_json_src}')
    print(f'Processed {table_name}')
print('Done.')
