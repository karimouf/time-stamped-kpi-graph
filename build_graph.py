"""
Build Graph - Time-stamped KPI Graph Builder
Simple boilerplate to build KPI graph from table data using extractor and graph builder.
"""

import json
import os
import glob
from kpi_visualization import KPIGraphVisualizer
from tskg import KPIGraphBuilder
from kpi_extractor import KPIExtractor
import networkx as nx


def extract_table_data(tables_path):
    """Extract table data from JSONL files and return list of tables"""
    # Check if tables_path is a file or directory
    if os.path.isfile(tables_path):
        # Single file mode
        jsonl_files = [tables_path]
    elif os.path.isdir(tables_path):
        # Directory mode - find all JSONL files
        jsonl_files = glob.glob(os.path.join(tables_path, "*.jsonl"))
    else:
        raise ValueError(f"Path {tables_path} is neither a file nor a directory")
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {tables_path}")
    
    print(f"Processing {len(jsonl_files)} JSONL file(s):")
    for file_path in jsonl_files:
        print(f"  - {os.path.basename(file_path)}")
    
    # Extract all table data
    all_tables = []
    total_tables = 0
    
    for jsonl_file in jsonl_files:
        file_tables = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines and commented lines (starting with // or #)
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('//') and not stripped_line.startswith('#'):
                    table_data = json.loads(line)
                    all_tables.append(table_data)
                    file_tables += 1
                    total_tables += 1
        
        print(f"    → Extracted {file_tables} tables from {os.path.basename(jsonl_file)}")
    
    print(f"Total tables extracted: {total_tables}")
    return all_tables



def build_graph_from_tables(builder, extractor, tables_path) -> nx.DiGraph:
    """Build complete graph from all JSONL files using extracted table data"""
    
    # Extract all table data
    all_tables = extract_table_data(tables_path)
    
    # Process each table
    all_nodes = []
    for table_data in all_tables:
        nodes = extractor.extract_kpis_from_table(table_data)
        all_nodes.extend(nodes)
        for node in nodes:
            builder.add_node(node)

    builder.create_edges_for_kpi("audi_production")
    builder.export_to_json("kpi_graph_output.json")
    
    
    return builder.graph



def main():
    """Main function to build KPI graph using extractor and graph builder"""
    
    # Initialize components
    print("🚀 Initializing KPI Graph Builder Components...")
    extractor = KPIExtractor()
    builder = KPIGraphBuilder()
    
    # Set data path
    tables_path = "data/tables"
    
    # Build the graph
    print("📊 Building graph from tables...")

    graph = build_graph_from_tables(builder, extractor, tables_path)
    visualizer = KPIGraphVisualizer(graph)
    
    print("\n2. Showing network graph for audi production...")
    visualizer.plot_network_graph(kpi_filter="audi_production", max_nodes=100)
    
    print("\n3. Showing timeline for audi production...")
    visualizer.plot_kpi_timeline(kpi_name="audi_production", 
                                entities=["A1","A4", "A6", "A7", "A8","Q3", "Q5", "Q6", "Q8"])
    
    # # Get and display statistics
    # print("\n📈 Graph Statistics:")
    # stats = builder.get_graph_statistics()
    # for key, value in stats.items():
    #     print(f"  {key}: {value}")
    
    # # Perform verification
    # print("\n🔍 Verifying graph...")
    # results = verifier.verify_graph(graph)
    # print(f"  Consistency: {results['consistency_rate']:.1%}")
    
    # Export results
    print("✅ Done!")
    return graph


if __name__ == "__main__":
    graph = main()
