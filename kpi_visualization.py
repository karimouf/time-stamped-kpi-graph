"""
KPI Graph Visualization Module
Creates visual representations of the time-stamped KPI graph
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from collections import defaultdict
import numpy as np
from typing import Dict, List
import seaborn as sns


class KPIGraphVisualizer:
    """Creates various visualizations of the KPI graph"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        
    def plot_network_graph(self, kpi_filter: str = None, max_nodes: int = 50):
        """Plot the network graph structure"""
        # Filter graph if specified
        if kpi_filter:
            filtered_nodes = [n for n, d in self.graph.nodes(data=True) 
                            if kpi_filter.lower() in d['kpi_name'].lower()]
            subgraph = self.graph.subgraph(filtered_nodes[:max_nodes])
        else:
            # Sample nodes if too many
            nodes = list(self.graph.nodes())
            if len(nodes) > max_nodes:
                nodes = nodes[:max_nodes]
            subgraph = self.graph.subgraph(nodes)

        print(f"Plotting graph with {subgraph} edges")
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Color nodes by KPI type
        node_colors = []
        kpi_types = list(set(d['kpi_name'] for _, d in subgraph.nodes(data=True)))
        color_map = plt.cm.Set3(np.linspace(0, 1, len(kpi_types)))
        kpi_color_dict = dict(zip(kpi_types, color_map))
        
        for node_id, data in subgraph.nodes(data=True):
            node_colors.append(kpi_color_dict[data['kpi_name']])
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=1500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, arrows=True, 
                              arrowsize=20, arrowstyle='->')
        
        # Add labels for key nodes
        labels = {node_id: f"{data['key'][:10]}\n{data['year']}\n{data['value']}" 
                 for node_id, data in subgraph.nodes(data=True)}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title(f"KPI Network Graph{'- ' + kpi_filter if kpi_filter else ''}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_kpi_timeline(self, kpi_name: str, entities: List[str] = None):
        """Plot timeline for specific KPI across entities"""
        # Filter nodes for the specified KPI
        kpi_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data['kpi_name'].lower() == kpi_name.lower():
                kpi_nodes.append((node_id, data))

        if not kpi_nodes:
            return
        
        # Convert to DataFrame
        df_data = []
        for node_id, data in kpi_nodes:
            if entities and data['key'] not in entities:
                continue
        
            
            df_data.append({
                'entity': data['key'],
                'year': data['year'],
                'value': data['value'],
                'kpi': data['kpi_name']
            })
        
        if not df_data:
            print(f"No numeric data found for KPI: {kpi_name}")
            return
        
        df = pd.DataFrame(df_data)
        
        plt.figure(figsize=(12, 8))
        
        # Plot lines for each entity
        for entity in df['entity'].unique():
            entity_data = df[df['entity'] == entity].sort_values('year')
            plt.plot(entity_data['year'], entity_data['value'], 
                    marker='o', linewidth=2, label=entity[:20])
        
        plt.title(f'Timeline: {kpi_name.title()}')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_kpi_distribution(self):
        """Plot distribution of KPIs and entities"""
        # Count KPIs and entities
        kpi_counts = defaultdict(int)
        entity_counts = defaultdict(int)
        year_counts = defaultdict(int)
        
        for node_id, data in self.graph.nodes(data=True):
            kpi_counts[data['kpi_name']] += 1
            entity_counts[data['key']] += 1
            year_counts[data['year']] += 1
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # KPI distribution
        kpis = list(kpi_counts.keys())
        counts = list(kpi_counts.values())
        axes[0, 0].bar(range(len(kpis)), counts)
        axes[0, 0].set_xticks(range(len(kpis)))
        axes[0, 0].set_xticklabels([kpi[:15] for kpi in kpis], rotation=45, ha='right')
        axes[0, 0].set_title('KPI Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # Top entities
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        entities, e_counts = zip(*top_entities)
        axes[0, 1].bar(range(len(entities)), e_counts)
        axes[0, 1].set_xticks(range(len(entities)))
        axes[0, 1].set_xticklabels([ent[:15] for ent in entities], rotation=45, ha='right')
        axes[0, 1].set_title('Top 15 Entities by KPI Count')
        axes[0, 1].set_ylabel('Count')
        
        # Year distribution
        years = sorted(year_counts.keys())
        y_counts = [year_counts[year] for year in years]
        axes[1, 0].bar(years, y_counts)
        axes[1, 0].set_title('KPI Nodes by Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Count')
        
        # Network metrics
        metrics = {
            'Total Nodes': self.graph.number_of_nodes(),
            'Total Edges': self.graph.number_of_edges(),
            'Unique KPIs': len(kpi_counts),
            'Unique Entities': len(entity_counts),
            'Years Covered': len(year_counts)
        }
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Graph Metrics:', fontsize=14, fontweight='bold', 
                       transform=axes[1, 1].transAxes)
        
        y_pos = 0.75
        for metric, value in metrics.items():
            axes[1, 1].text(0.1, y_pos, f'{metric}: {value}', fontsize=12,
                           transform=axes[1, 1].transAxes)
            y_pos -= 0.1
        
        plt.tight_layout()
        plt.show()
    
    def export_graph_data(self, filename: str = "kpi_graph_export.json"):
        """Export graph data to JSON for further analysis"""
        import json
        
        # Prepare data for export
        export_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'creation_date': pd.Timestamp.now().isoformat()
            }
        }
        
        # Export nodes
        for node_id, data in self.graph.nodes(data=True):
            export_data['nodes'].append({
                'id': node_id,
                'kpi_name': data['kpi_name'],
                'key': data['key'],
                'value': data['value'],
                'year': data['year'],
                'evidence': data['evidence']
            })
        
        # Export edges
        for source, target, data in self.graph.edges(data=True):
            export_data['edges'].append({
                'source': source,
                'target': target,
                'year_diff': data.get('year_diff', 0),
                'edge_type': data.get('edge_type', 'temporal')
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph data exported to {filename}")
        
    def generate_sample_queries(self) -> List[str]:
        """Generate sample queries to explore the graph"""
        # Get sample KPIs and entities
        sample_kpis = list(set(d['kpi_name'] for _, d in self.graph.nodes(data=True)))[:5]
        sample_entities = list(set(d['key'] for _, d in self.graph.nodes(data=True)))[:5]
        
        queries = [
            f"Timeline for '{sample_kpis[0]}' across all entities",
            f"Network view filtered by '{sample_kpis[1]}'",
            f"Compare '{sample_entities[0]}' vs '{sample_entities[1]}' for sales metrics",
            "Overall KPI and entity distributions",
            "Export complete graph data for analysis"
        ]
        
        return queries


def demonstrate_visualization():
    """Demonstrate the visualization capabilities"""
    from kpi_graph_builder import KPIGraphBuilder
    
    print("Loading and building KPI graph...")
    builder = KPIGraphBuilder()
    graph = builder.build_graph_from_tables("data/tables")
    
    visualizer = KPIGraphVisualizer(graph)
    
    print("\n1. Showing KPI and entity distributions...")
    visualizer.plot_kpi_distribution()
    
    print("\n2. Showing network graph for sales revenue...")
    visualizer.plot_network_graph(kpi_filter="sales_revenue", max_nodes=30)
    
    print("\n3. Showing timeline for vehicle sales...")
    visualizer.plot_kpi_timeline("vehicle_sales", 
                                entities=["Volkswagen Passenger Cars", "Audi", "ŠKODA"])
    
    print("\n4. Exporting graph data...")
    visualizer.export_graph_data("kpi_graph_export.json")
    
    print("\n5. Sample queries you can try:")
    queries = visualizer.generate_sample_queries()
    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")


if __name__ == "__main__":
    demonstrate_visualization()