"""
Time-stamped KPI Graph Builder
This module extracts KPIs from linked tables and builds a time-stamped graph structure.
"""

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
    evidence: Dict  # Metadata about the source


@dataclass
class KPIEdge:
    """Represents an edge connecting KPI nodes across years"""
    source_node_id: str
    target_node_id: str
    year_diff: int


class KPIExtractor:
    """Extracts KPIs from table data and builds semantic understanding"""
    
    def __init__(self):
        self.kpi_patterns = {
            'sales_revenue': ['sales revenue', 'revenue', 'turnover'],
            'vehicle_sales': ['vehicle sales', 'unit sales', 'sales'],
            'operating_result': ['operating result', 'operating profit', 'profit'],
            'production': ['production', 'produced', 'manufactured'],
            'deliveries': ['deliveries', 'delivered'],
            'return_on_sales': ['return on sales', 'operating return', 'margin']
        }
    
    def extract_year_from_header(self, header: str) -> Optional[int]:
        """Extract year from merged header like 'SALES REVENUE 2022'"""
        year_match = re.search(r'\b(20\d{2})\b', header)
        return int(year_match.group(1)) if year_match else None
    
    def extract_kpi_name(self, header: str) -> str:
        """Extract KPI name from header, removing year"""
        # Remove year pattern
        kpi_name = re.sub(r'\s*(20\d{2})\s*$', '', header)
        return kpi_name.strip().lower()
    
    def normalize_kpi_name(self, kpi_name: str) -> str:
        """Normalize KPI names for consistency"""
        kpi_name_lower = kpi_name.lower()
        for normalized, patterns in self.kpi_patterns.items():
            for pattern in patterns:
                if pattern in kpi_name_lower:
                    return normalized
        return kpi_name_lower
    
    def is_valid_value(self, value: str) -> bool:
        """Check if value is valid (not missing data marker)"""
        return value.strip() not in ['-', '–', '', 'x', '...']
    
    def clean_value(self, value: str) -> str:
        """Clean and normalize value strings"""
        # Remove footnote markers like ^1, ^2
        cleaned = re.sub(r'\s*\^\d+', '', value)
        return cleaned.strip()
    
    def extract_kpis_from_table(self, table_data: Dict) -> List[KPINode]:
        """Extract all KPIs from a single table"""
        nodes = []
        merged_headers = table_data.get('merged_headers', [])
        rows = table_data.get('rows', [])
        stub_col = table_data.get('stub_col', [])
        
        # Skip first header (usually units/description)
        data_headers = merged_headers[1:] if len(merged_headers) > 1 else []
        
        for row_idx, row in enumerate(rows):
            if row_idx >= len(stub_col):
                continue
                
            key = stub_col[row_idx].strip()
            if not key or key == "":  # Skip empty keys
                continue
            
            # Skip data values (first column is usually units)
            data_values = row[1:] if len(row) > 1 else []
            
            for col_idx, value in enumerate(data_values):
                if col_idx >= len(data_headers):
                    continue
                
                header = data_headers[col_idx]
                if not self.is_valid_value(value):
                    continue
                
                year = self.extract_year_from_header(header)
                if not year:
                    continue
                
                kpi_name = self.extract_kpi_name(header)
                normalized_kpi = self.normalize_kpi_name(kpi_name)
                cleaned_value = self.clean_value(value)
                
                # Create evidence metadata
                evidence = {
                    "table_id": table_data.get('table_id'),
                    "doc_id": table_data.get('doc_id'), 
                    "year": table_data.get('year'),
                    "page": table_data.get('page'),
                    "bucket": table_data.get('bucket'),
                    "section_name": table_data.get('section_name'),
                    "section_level": 3,  # Default level
                    "title": table_data.get('title')
                }
                
                node = KPINode(
                    kpi_name=normalized_kpi,
                    key=key,
                    value=cleaned_value,
                    year=year,
                    evidence=evidence
                )
                nodes.append(node)
        
        return nodes


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
                           evidence=node.evidence)
        
        # Group by KPI-key combination for edge creation
        kpi_key = (node.kpi_name, node.key)
        self.nodes_by_kpi_key[kpi_key].append((node_id, node.year))
    
    def create_temporal_edges(self):
        """Create edges connecting same KPI-key pairs across years"""
        for kpi_key, node_list in self.nodes_by_kpi_key.items():
            # Sort by year
            node_list.sort(key=lambda x: x[1])
            
            # Create edges between consecutive years
            for i in range(len(node_list) - 1):
                source_id, source_year = node_list[i]
                target_id, target_year = node_list[i + 1]
                
                edge = KPIEdge(
                    source_node_id=source_id,
                    target_node_id=target_id,
                    year_diff=target_year - source_year
                )
                
                self.graph.add_edge(source_id, target_id, 
                                  year_diff=edge.year_diff,
                                  edge_type="temporal")
    
    def build_graph_from_tables(self, tables_file: str) -> nx.DiGraph:
        """Build complete graph from tables file"""
        extractor = KPIExtractor()
        
        with open(tables_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    table_data = json.loads(line)
                    nodes = extractor.extract_kpis_from_table(table_data)
                    
                    for node in nodes:
                        self.add_node(node)
        
        # Create temporal edges
        self.create_temporal_edges()
        
        return self.graph
    
    def get_graph_statistics(self) -> Dict:
        """Get basic statistics about the graph"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "unique_kpis": len(set(data['kpi_name'] for _, data in self.graph.nodes(data=True))),
            "unique_keys": len(set(data['key'] for _, data in self.graph.nodes(data=True))),
            "year_range": self._get_year_range()
        }
    
    def _get_year_range(self) -> Tuple[int, int]:
        """Get the range of years in the graph"""
        years = [data['year'] for _, data in self.graph.nodes(data=True)]
        return (min(years), max(years)) if years else (None, None)


class SemanticVerifier:
    """Performs semantic verification of extracted KPI nodes"""
    
    def __init__(self):
        self.value_patterns = {
            'sales_revenue': r'\d+(\.\d+)?\s*(million|billion|€|dollars?)?',
            'vehicle_sales': r'\d+(\.\d+)?',
            'production': r'\d+(\.\d+)?',
            'deliveries': r'\d+(\.\d+)?'
        }
    
    def verify_node_consistency(self, node: KPINode) -> bool:
        """Verify if a node is semantically consistent"""
        # Check if value format matches KPI type
        if node.kpi_name in self.value_patterns:
            pattern = self.value_patterns[node.kpi_name]
            if not re.search(pattern, node.value, re.IGNORECASE):
                return False
        
        # Check if year in evidence matches node year
        if node.evidence.get('year') and node.evidence['year'] != node.year:
            return False
        
        return True
    
    def verify_graph(self, graph: nx.DiGraph) -> Dict:
        """Verify all nodes in the graph"""
        total_nodes = graph.number_of_nodes()
        valid_nodes = 0
        invalid_nodes = []
        
        for node_id, data in graph.nodes(data=True):
            node = KPINode(
                kpi_name=data['kpi_name'],
                key=data['key'],
                value=data['value'],
                year=data['year'],
                evidence=data['evidence']
            )
            
            if self.verify_node_consistency(node):
                valid_nodes += 1
            else:
                invalid_nodes.append(node_id)
        
        return {
            "total_nodes": total_nodes,
            "valid_nodes": valid_nodes,
            "invalid_nodes": len(invalid_nodes),
            "consistency_rate": valid_nodes / total_nodes if total_nodes > 0 else 0,
            "invalid_node_ids": invalid_nodes[:10]  # Show first 10 invalid nodes
        }


def main():
    """Main function to demonstrate the KPI graph building process"""
    tables_file = "data/tables/linked_tables.jsonl"
    
    print("Building time-stamped KPI graph...")
    
    # Build the graph
    builder = KPIGraphBuilder()
    graph = builder.build_graph_from_tables(tables_file)
    
    # Get statistics
    stats = builder.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"- Total nodes: {stats['total_nodes']}")
    print(f"- Total edges: {stats['total_edges']}")
    print(f"- Unique KPIs: {stats['unique_kpis']}")
    print(f"- Unique entities: {stats['unique_keys']}")
    print(f"- Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")
    
    # Perform semantic verification
    verifier = SemanticVerifier()
    verification_results = verifier.verify_graph(graph)
    print(f"\nSemantic Verification:")
    print(f"- Consistency rate: {verification_results['consistency_rate']:.2%}")
    print(f"- Valid nodes: {verification_results['valid_nodes']}")
    print(f"- Invalid nodes: {verification_results['invalid_nodes']}")
    
    # Sample some nodes
    print(f"\nSample KPI nodes:")
    sample_nodes = list(graph.nodes(data=True))[:5]
    for node_id, data in sample_nodes:
        print(f"- {node_id}: {data['kpi_name']} = {data['value']} ({data['key']}, {data['year']})")
    
    return graph


if __name__ == "__main__":
    graph = main()