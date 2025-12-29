import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


class GraphNode:
    """
    Graph node representing a KPI with temporal links.
    
    Attributes:
        kpi_id: Unique identifier for this KPI (hash-based)
        name: KPI metric name (e.g., "Sales", "Operating Cost")
        value: Numeric value of the KPI
        key: Entity/context (e.g., "Audi", "Core Brand Group")
        year: Temporal information
        units: Unit of measurement
        next: Link to next temporal KPI (same name+key, next year)
        prev: Link to previous temporal KPI (same name+key, previous year)
    """
    
    def __init__(self, kpi_id: str, name: str, value: float, key: str, year: int, 
                 units: Optional[str] = None, **metadata):
        self.kpi_id = kpi_id
        self.name = name
        self.value = value
        self.key = key
        self.year = year
        self.units = units
        self.metadata = metadata  # Store additional fields like row_idx, col_idx, etc.
        self.next = None
        self.prev = None
    
    def __repr__(self):
        return f"GraphNode(id={self.kpi_id[:8]}..., name='{self.name}', key='{self.key}', year={self.year}, value={self.value})"
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary format for JSON serialization."""
        return {
            "kpi_id": self.kpi_id,
            "name": self.name,
            "value": self.value,
            "key": self.key,
            "year": self.year,
            "units": self.units,
            "next_kpi_id": self.next.kpi_id if self.next else None,
            "prev_kpi_id": self.prev.kpi_id if self.prev else None,
            **self.metadata
        }


def generate_kpi_id(name: str, key: str, year: int, value: float) -> str:
    """
    Generate a unique KPI ID based on its attributes.
    
    Args:
        name: KPI metric name
        key: Entity/context
        year: Year
        value: Numeric value
    
    Returns:
        SHA256 hash as unique identifier
    """
    composite = f"{name}|{key}|{year}|{value}"
    return hashlib.sha256(composite.encode()).hexdigest()


def link_kpis(kpis: List[Dict]) -> List[GraphNode]:
    """
    Create a temporal graph by linking KPIs across years.
    
    Logic:
        1. Group KPIs by (name, key) - same metric for same entity
        2. Sort each group by year
        3. Link consecutive years: node[i].next = node[i+1], node[i+1].prev = node[i]
        4. Return list of all graph nodes
    
    Args:
        kpis: List of KPI dictionaries with name, key, year, value fields
    
    Returns:
        List of GraphNode objects with temporal links established
    """
    # Group KPIs by (name, key)
    groups = {}
    nodes_map = {}  # kpi_id -> GraphNode
    
    for kpi in kpis:
        # Extract evidence data if available
        evidence = kpi.get("evidence", {})
        
        # Generate unique ID
        kpi_id = generate_kpi_id(
            kpi.get("name"),
            kpi.get("key"),
            kpi.get("year"),
            kpi.get("value")
        )
        
        # Create graph node
        node = GraphNode(
            kpi_id=kpi_id,
            name=kpi.get("name"),
            value=kpi.get("value"),
            key=kpi.get("key"),
            year=kpi.get("year"),
            units=kpi.get("units"),
            row_idx=evidence.get("row_idx"),
            col_idx=evidence.get("col_idx"),
            row_name=evidence.get("row_name"),
            col_name=evidence.get("col_name"),
            table_id=evidence.get("table_id")
        )
        
        nodes_map[kpi_id] = node
        
        # Group by (name, key)
        group_key = (kpi.get("name", ""), kpi.get("key", ""))
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(node)
    
    # Link nodes within each group by year
    for group_key, group_nodes in groups.items():
        # Filter out nodes with missing year and sort by year
        valid_nodes = [n for n in group_nodes if n.year is not None]
        valid_nodes.sort(key=lambda n: n.year)
        
        # Link consecutive years
        for i in range(len(valid_nodes) - 1):
            current = valid_nodes[i]
            next_node = valid_nodes[i + 1]
            
            current.next = next_node
            next_node.prev = current
    
    return list(nodes_map.values())


def save_graph(nodes: List[GraphNode], output_path: Path):
    """
    Save the KPI graph to JSON format.
    
    Args:
        nodes: List of GraphNode objects
        output_path: Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    graph_data = {
        "total_nodes": len(nodes),
        "nodes": [node.to_dict() for node in nodes]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Graph saved with {len(nodes)} nodes to: {output_path}")


def load_kpis_from_validation(validation_file: Path) -> List[Dict]:
    """
    Load valid KPIs from validation JSON file.
    
    Args:
        validation_file: Path to validation JSON file (valid_*.json)
    
    Returns:
        List of valid KPI dictionaries
    """
    with open(validation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("valid_kpis", [])


def analyze_grouping_strategies(all_kpis: List[Dict]):
    """
    Analyze and visualize different grouping strategies to understand temporal linking.
    
    Strategy 1: Group by NAME only
    Strategy 2: Group by (NAME, KEY)
    Strategy 3: Group by (NAME, KEY, UNITS)
    
    Args:
        all_kpis: List of all valid KPI dictionaries
    """
    print("\n" + "="*80)
    print("GROUPING STRATEGY ANALYSIS")
    print("="*80)
    
    # Strategy 1: Group by NAME only
    print("\n" + "-"*80)
    print("STRATEGY 1: GROUP BY NAME ONLY")
    print("-"*80)
    groups_by_name = {}
    for kpi in all_kpis:
        name = kpi.get("name", "")
        if name not in groups_by_name:
            groups_by_name[name] = []
        groups_by_name[name].append(kpi)
    
    print(f"\nTotal groups: {len(groups_by_name)}")
    print(f"\nTop 10 groups by size:")
    sorted_by_name = sorted(groups_by_name.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (name, kpis) in enumerate(sorted_by_name[:10], 1):
        years = sorted(set(k.get("year") for k in kpis if k.get("year") is not None))
        unique_keys = set(k.get("key") for k in kpis)
        unique_units = set(k.get("units") for k in kpis)
        print(f"\n{i}. Name: '{name}'")
        print(f"   Total KPIs: {len(kpis)}")
        print(f"   Years: {years}")
        print(f"   Unique Keys: {len(unique_keys)} - {list(unique_keys)[:3]}{'...' if len(unique_keys) > 3 else ''}")
        print(f"   Unique Units: {unique_units}")
    
    # Strategy 2: Group by (NAME, KEY)
    print("\n" + "-"*80)
    print("STRATEGY 2: GROUP BY (NAME, KEY)")
    print("-"*80)
    groups_by_name_key = {}
    for kpi in all_kpis:
        group_key = (kpi.get("name", ""), kpi.get("key", ""))
        if group_key not in groups_by_name_key:
            groups_by_name_key[group_key] = []
        groups_by_name_key[group_key].append(kpi)
    
    print(f"\nTotal groups: {len(groups_by_name_key)}")
    print(f"\nTop 10 groups by size:")
    sorted_by_name_key = sorted(groups_by_name_key.items(), key=lambda x: len(x[1]), reverse=True)
    for i, ((name, key), kpis) in enumerate(sorted_by_name_key[:10], 1):
        years = sorted(set(k.get("year") for k in kpis if k.get("year") is not None))
        unique_units = set(k.get("units") for k in kpis)
        values = [k.get("value") for k in sorted(kpis, key=lambda x: x.get("year") or 0)]
        print(f"\n{i}. Name: '{name}' | Key: '{key}'")
        print(f"   Total KPIs: {len(kpis)}")
        print(f"   Years: {years}")
        print(f"   Unique Units: {unique_units}")
        print(f"   Value progression: {values}")
    
    # Strategy 3: Group by (NAME, KEY, UNITS)
    print("\n" + "-"*80)
    print("STRATEGY 3: GROUP BY (NAME, KEY, UNITS)")
    print("-"*80)
    groups_by_name_key_units = {}
    for kpi in all_kpis:
        group_key = (kpi.get("name", ""), kpi.get("key", ""), kpi.get("units"))
        if group_key not in groups_by_name_key_units:
            groups_by_name_key_units[group_key] = []
        groups_by_name_key_units[group_key].append(kpi)
    
    print(f"\nTotal groups: {len(groups_by_name_key_units)}")
    print(f"\nTop 10 groups by size:")
    sorted_by_name_key_units = sorted(groups_by_name_key_units.items(), key=lambda x: len(x[1]), reverse=True)
    for i, ((name, key, units), kpis) in enumerate(sorted_by_name_key_units[:10], 1):
        years = sorted(set(k.get("year") for k in kpis if k.get("year") is not None))
        values = [k.get("value") for k in sorted(kpis, key=lambda x: x.get("year") or 0)]
        print(f"\n{i}. Name: '{name}' | Key: '{key}' | Units: '{units}'")
        print(f"   Total KPIs: {len(kpis)}")
        print(f"   Years: {years}")
        print(f"   Value progression: {values}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    # Count groups with temporal coverage
    def count_temporal_groups(groups_dict, min_years=2):
        return sum(1 for kpis in groups_dict.values() if len(set(k.get("year") for k in kpis if k.get("year") is not None)) >= min_years)
    
    temporal_name = count_temporal_groups(groups_by_name)
    temporal_name_key = count_temporal_groups(groups_by_name_key)
    temporal_name_key_units = count_temporal_groups(groups_by_name_key_units)
    
    print(f"\nStrategy 1 (NAME only):")
    print(f"  Total groups: {len(groups_by_name)}")
    print(f"  Groups with 2+ years: {temporal_name}")
    print(f"  Average KPIs per group: {len(all_kpis)/len(groups_by_name):.1f}")
    
    print(f"\nStrategy 2 (NAME + KEY):")
    print(f"  Total groups: {len(groups_by_name_key)}")
    print(f"  Groups with 2+ years: {temporal_name_key}")
    print(f"  Average KPIs per group: {len(all_kpis)/len(groups_by_name_key):.1f}")
    
    print(f"\nStrategy 3 (NAME + KEY + UNITS):")
    print(f"  Total groups: {len(groups_by_name_key_units)}")
    print(f"  Groups with 2+ years: {temporal_name_key_units}")
    print(f"  Average KPIs per group: {len(all_kpis)/len(groups_by_name_key_units):.1f}")
    
    print("\n" + "="*80)
    
    return {
        "by_name": groups_by_name,
        "by_name_key": groups_by_name_key,
        "by_name_key_units": groups_by_name_key_units
    }


def main():
    """
    Main function to link KPIs using name and key only.
    """
    base_dir = Path(__file__).parent
    validation_dir = base_dir / 'validation' / 'valid'
    output_dir = base_dir / 'data' / 'output' / 'kpi_links'
    
    # Collect all valid KPIs from validation files
    all_kpis = []
    validation_files = sorted(validation_dir.glob("valid_*.json"))
    
    print(f"Loading valid KPIs from {len(validation_files)} validation files...")
    for file in validation_files:
        kpis = load_kpis_from_validation(file)
        all_kpis.extend(kpis)
        print(f"  Loaded {len(kpis)} valid KPIs from {file.name}")
    
    print(f"\nTotal valid KPIs loaded: {len(all_kpis)}")
    
    # Build temporal graph using name + key
    print("\nLinking KPIs by (name, key)...")
    nodes = link_kpis(all_kpis)
    
    # Calculate statistics
    linked_nodes = sum(1 for n in nodes if n.next or n.prev)
    chain_starts = sum(1 for n in nodes if n.prev is None and n.next is not None)
    isolated_nodes = sum(1 for n in nodes if n.next is None and n.prev is None)
    
    print(f"\nLinking Statistics:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Linked nodes: {linked_nodes}")
    print(f"  Temporal chains: {chain_starts}")
    print(f"  Isolated nodes: {isolated_nodes}")
    
    # Save links to JSON file
    output_file = output_dir / 'links.json'
    save_graph(nodes, output_file)
    
    print(f"\n✅ Links saved to: {output_file}")


if __name__ == "__main__":
    main()


