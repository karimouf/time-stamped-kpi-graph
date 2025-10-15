import json
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

@dataclass
class KPINode:
    """Represents a KPI node in the graph"""
    kpi_name: str
    key: str  # Company/entity identifier
    value: str
    year: int
    units: str  # Measurement units
    evidence: Dict  # Metadata about the source


@dataclass
class KPIEdge:
    """Represents an edge connecting KPI nodes across years"""
    source_node_id: str
    target_node_id: str
    year_diff: int

class KPIGraphBuilder:
    """Builds time-stamped KPI graph with nodes and edges"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes_by_kpi_key = defaultdict(list)  # Group nodes by (KPI, key) pairs
    
    def generate_node_id(self, node: KPINode) -> str:
        """Generate unique ID for a node"""
        return f"{node.kpi_name}_{node.key}_{node.year}".replace(" ", "_").replace("/", "_")
    
    def add_node(self, node: KPINode):
        """Add a KPI node to the graph"""
        node_id = self.generate_node_id(node)
        
        # Add node with all attributes
        self.graph.add_node(node_id, 
                           kpi_name=node.kpi_name,
                           key=node.key,
                           value=node.value,
                           year=node.year,
                           units=node.units,
                           evidence=node.evidence)
        
        # Group by KPI-key combination for edge creation
        kpi_key = (node.kpi_name, node.key)
        self.nodes_by_kpi_key[kpi_key].append((node_id, node.year))
    
    def create_temporal_edges(self):
        """Create edges connecting same KPI-key pairs across years (no same-year connections)"""
        for kpi_key, node_list in self.nodes_by_kpi_key.items():
            # Sort by year
            node_list.sort(key=lambda x: x[1])
            
            # Group nodes by year to handle multiple nodes per year
            nodes_by_year = defaultdict(list)
            for node_id, year in node_list:
                nodes_by_year[year].append(node_id)
            
            # Get sorted years
            sorted_years = sorted(nodes_by_year.keys())
            
            # Create edges between consecutive years only
            for i in range(len(sorted_years) - 1):
                source_year = sorted_years[i]
                target_year = sorted_years[i + 1]
                
                # Skip if somehow the same year (should not happen with sorted unique keys)
                if source_year == target_year:
                    continue
                
                # Connect all nodes from source year to all nodes in target year
                # (in case there are multiple nodes per year)
                source_nodes = nodes_by_year[source_year]
                target_nodes = nodes_by_year[target_year]
                
                for source_id in source_nodes:
                    for target_id in target_nodes:
                        year_diff = target_year - source_year
                        
                        edge = KPIEdge(
                            source_node_id=source_id,
                            target_node_id=target_id,
                            year_diff=year_diff
                        )
                        
                        self.graph.add_edge(source_id, target_id, 
                                          year_diff=edge.year_diff,
                                          edge_type="temporal")

    def create_edges_for_kpi(self, kpi_name: str, entity_key: str = None):
        """Create temporal edges for a specific KPI, optionally filtered by entity key"""
        # Filter nodes by KPI name and optionally by entity key
        target_kpi_keys = []
        
        for kpi_key, node_list in self.nodes_by_kpi_key.items():
            stored_kpi_name, stored_key = kpi_key
            
            # Match KPI name
            if stored_kpi_name == kpi_name:
                # If entity_key is specified, also match the entity key
                if entity_key is None or stored_key == entity_key:
                    target_kpi_keys.append(kpi_key)
        
        # Create edges for each matching KPI-key combination
        for kpi_key in target_kpi_keys:
            node_list = self.nodes_by_kpi_key[kpi_key]
            
            # Sort by year
            node_list.sort(key=lambda x: x[1])
            
            # Group nodes by year to handle multiple nodes per year
            nodes_by_year = defaultdict(list)
            for node_id, year in node_list:
                nodes_by_year[year].append(node_id)
            
            # Get sorted years
            sorted_years = sorted(nodes_by_year.keys())
            
            # Create edges between consecutive years only
            for i in range(len(sorted_years) - 1):
                source_year = sorted_years[i]
                target_year = sorted_years[i + 1]
                
                # Skip if somehow the same year (should not happen with sorted unique keys)
                if source_year == target_year:
                    continue
                
                # Connect all nodes from source year to all nodes in target year
                source_nodes = nodes_by_year[source_year]
                target_nodes = nodes_by_year[target_year]
                
                for source_id in source_nodes:
                    for target_id in target_nodes:
                        year_diff = target_year - source_year
                        
                        edge = KPIEdge(
                            source_node_id=source_id,
                            target_node_id=target_id,
                            year_diff=year_diff
                        )
                        
                        # Only add edge if it doesn't already exist
                        if not self.graph.has_edge(source_id, target_id):
                            self.graph.add_edge(source_id, target_id, 
                                              year_diff=edge.year_diff,
                                              edge_type="temporal")
        
        print(f"✅ Created temporal edges for KPI: '{kpi_name}'" + 
              (f" (entity: '{entity_key}')" if entity_key else " (all entities)"))

    def _create_kpi_node(self, kpi_name: str, key: str, value: str, year: int, table_data: Dict, 
                        row_idx: int = None, col_idx: int = None, header: str = None, row_data: List[str] = None, 
                        table_units: str = None) -> KPINode:
        """Create a KPI node with complete evidence metadata including position tracking"""

        evidence = {
            "table_id": table_data.get('table_id'),
            "doc_id": table_data.get('doc_id'), 
            "year": table_data.get('year'),
            "page": table_data.get('page'),
            "bucket": table_data.get('bucket'),
            "section_name": table_data.get('section_name'),
            "title": table_data.get('title'),
            # Position tracking
            "row_index": row_idx,  # Row position in the table (0-based)
            "column_index": col_idx,  # Column position in the table (0-based) 
            "header_text": header,  # Original header text for this column

        }
        return KPINode(
            kpi_name=kpi_name,
            key=key,
            value=value,
            year=year,
            units=table_units,  # Units extracted from headers or title
            evidence=evidence
        )
    
    def _get_year_range(self) -> Tuple[int, int]:
        """Get the range of years in the graph"""
        years = [data['year'] for _, data in self.graph.nodes(data=True)]
        return (min(years), max(years)) if years else (None, None)
    
    def export_to_json(self, filename: str = "kpi_graph_export.json") -> None:
        """Export the complete graph to JSON format"""
        import json
        from datetime import datetime
        
        # Prepare data for export
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'unique_kpis': len(set(data['kpi_name'] for _, data in self.graph.nodes(data=True))),
                'unique_entities': len(set(data['key'] for _, data in self.graph.nodes(data=True))),
                'year_range': self._get_year_range()
            },
            'nodes': [],
            'edges': []
        }
        
        # Export nodes
        for node_id, data in self.graph.nodes(data=True):
            export_data['nodes'].append({
                'id': node_id,
                'kpi_name': data['kpi_name'],
                'key': data['key'],
                'units': data['units'],
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
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Graph exported to: {filename}")
        print(f"   • {export_data['metadata']['total_nodes']} nodes")
        print(f"   • {export_data['metadata']['total_edges']} edges")
        print(f"   • {export_data['metadata']['unique_kpis']} unique KPIs")
        print(f"   • {export_data['metadata']['unique_entities']} unique entities")


    