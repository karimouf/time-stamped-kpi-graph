"""
Time-stamped KPI Graph Demo
Complete demonstration of the KPI extraction and graph building system
"""

import json
from kpi_graph_builder import KPIGraphBuilder, SemanticVerifier
from kpi_visualization import KPIGraphVisualizer


def print_separator(title=""):
    """Print a nice separator with optional title"""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


def demonstrate_kpi_extraction():
    """Demonstrate the KPI extraction process"""
    print_separator("KPI EXTRACTION AND GRAPH BUILDING")
    
    print("🔍 Loading and analyzing table data...")
    builder = KPIGraphBuilder()
    graph = builder.build_graph_from_tables("data/tables/linked_tables.jsonl")
    
    # Get statistics
    stats = builder.get_graph_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   • Total KPI nodes: {stats['total_nodes']}")
    print(f"   • Temporal edges: {stats['total_edges']}")
    print(f"   • Unique KPIs: {stats['unique_kpis']}")
    print(f"   • Unique entities: {stats['unique_keys']}")
    print(f"   • Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")
    
    return graph


def demonstrate_semantic_verification(graph):
    """Demonstrate semantic verification of extracted data"""
    print_separator("SEMANTIC VERIFICATION")
    
    verifier = SemanticVerifier()
    results = verifier.verify_graph(graph)
    
    print(f"🔬 Semantic Consistency Check:")
    print(f"   • Total nodes verified: {results['total_nodes']}")
    print(f"   • Valid nodes: {results['valid_nodes']}")
    print(f"   • Invalid nodes: {results['invalid_nodes']}")
    print(f"   • Consistency rate: {results['consistency_rate']:.1%}")
    
    if results['invalid_node_ids']:
        print(f"   • Sample invalid nodes: {', '.join(results['invalid_node_ids'][:3])}")


def show_sample_nodes(graph):
    """Show sample extracted KPI nodes"""
    print_separator("SAMPLE KPI NODES")
    
    print("📝 Sample extracted KPI nodes:")
    sample_nodes = list(graph.nodes(data=True))[:8]
    
    for i, (node_id, data) in enumerate(sample_nodes, 1):
        evidence = data['evidence']
        print(f"\n   {i}. Node ID: {node_id}")
        print(f"      • KPI: {data['kpi_name']}")
        print(f"      • Entity: {data['key']}")
        print(f"      • Value: {data['value']}")
        print(f"      • Year: {data['year']}")
        print(f"      • Source: {evidence.get('doc_id', 'N/A')} (Page {evidence.get('page', 'N/A')})")


def show_temporal_connections(graph):
    """Show examples of temporal connections"""
    print_separator("TEMPORAL CONNECTIONS")
    
    print("🔗 Sample temporal edges (KPI connections across years):")
    sample_edges = list(graph.edges(data=True))[:6]
    
    for i, (source, target, data) in enumerate(sample_edges, 1):
        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        
        print(f"\n   {i}. {source_data['kpi_name'].title()} - {source_data['key']}")
        print(f"      {source_data['year']} → {target_data['year']} (Δ{data.get('year_diff', 0)} years)")
        print(f"      Values: {source_data['value']} → {target_data['value']}")


def analyze_kpi_patterns(graph):
    """Analyze patterns in the KPI data"""
    print_separator("KPI PATTERN ANALYSIS")
    
    # Count KPIs by type
    kpi_counts = {}
    entity_kpi_counts = {}
    year_coverage = {}
    
    for node_id, data in graph.nodes(data=True):
        kpi = data['kpi_name']
        entity = data['key']
        year = data['year']
        
        kpi_counts[kpi] = kpi_counts.get(kpi, 0) + 1
        
        if entity not in entity_kpi_counts:
            entity_kpi_counts[entity] = set()
        entity_kpi_counts[entity].add(kpi)
        
        if kpi not in year_coverage:
            year_coverage[kpi] = set()
        year_coverage[kpi].add(year)
    
    print("📈 KPI Distribution:")
    sorted_kpis = sorted(kpi_counts.items(), key=lambda x: x[1], reverse=True)
    for kpi, count in sorted_kpis[:8]:
        years_span = len(year_coverage[kpi])
        print(f"   • {kpi.title()}: {count} data points across {years_span} years")
    
    print(f"\n🏢 Top entities by KPI diversity:")
    top_entities = sorted(entity_kpi_counts.items(), 
                         key=lambda x: len(x[1]), reverse=True)[:6]
    for entity, kpis in top_entities:
        print(f"   • {entity[:30]}: {len(kpis)} different KPIs")


def export_sample_data(graph):
    """Export sample data for inspection"""
    print_separator("DATA EXPORT")
    
    # Export sample for inspection
    sample_data = {
        'graph_summary': {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'sample_created': True
        },
        'sample_nodes': []
    }
    
    # Get diverse sample
    sample_nodes = list(graph.nodes(data=True))[:20]
    for node_id, data in sample_nodes:
        sample_data['sample_nodes'].append({
            'node_id': node_id,
            'kpi_name': data['kpi_name'],
            'entity': data['key'],
            'value': data['value'],
            'year': data['year'],
            'evidence': {
                'doc_id': data['evidence'].get('doc_id'),
                'page': data['evidence'].get('page'),
                'section': data['evidence'].get('section_name')
            }
        })
    
    with open('kpi_graph_sample.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("💾 Sample data exported:")
    print("   • File: kpi_graph_sample.json")
    print("   • Contains: 20 sample nodes with metadata")
    print("   • Use for: Further analysis or inspection")


def main():
    """Main demonstration function"""
    print("🚀 Time-stamped KPI Graph Builder")
    print("   Building semantic graph from automotive financial data...")
    
    try:
        # Step 1: Extract KPIs and build graph
        graph = demonstrate_kpi_extraction()
        
        # Step 2: Verify semantic consistency
        demonstrate_semantic_verification(graph)
        
        # Step 3: Show sample extracted nodes
        show_sample_nodes(graph)
        
        # Step 4: Show temporal connections
        show_temporal_connections(graph)
        
        # Step 5: Analyze patterns
        analyze_kpi_patterns(graph)
        
        # Step 6: Export sample data
        export_sample_data(graph)
        
        print_separator("COMPLETION SUMMARY")
        print("✅ Successfully built time-stamped KPI graph!")
        print("\n📋 What was accomplished:")
        print("   1. ✅ Parsed JSONL table data with headers/rows alignment")
        print("   2. ✅ Extracted KPIs (name, entity, value, year, evidence)")
        print("   3. ✅ Built graph with nodes and temporal edges")
        print("   4. ✅ Performed semantic consistency verification")
        print("   5. ✅ Handled missing data ('-') appropriately")
        print("   6. ✅ Connected same KPIs across years with edges")
        
        print("\n🎯 Graph Structure:")
        print(f"   • Each NODE contains: KPI name, entity key, value, year, evidence")
        print(f"   • Each EDGE connects: same KPI-entity pairs across years")
        print(f"   • Evidence includes: document ID, page, section, metadata")
        
        print("\n📁 Files created:")
        print("   • kpi_graph_builder.py - Core extraction logic")
        print("   • kpi_visualization.py - Visualization tools")
        print("   • kpi_graph_sample.json - Sample extracted data")
        print("   • requirements.txt - Python dependencies")
        
        print(f"\n🎉 Ready for further analysis and visualization!")
        
    except FileNotFoundError:
        print("❌ Error: Could not find data/tables/linked_tables.jsonl")
        print("   Please ensure the data files are in the correct location.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("   Check the data format and file paths.")


if __name__ == "__main__":
    main()