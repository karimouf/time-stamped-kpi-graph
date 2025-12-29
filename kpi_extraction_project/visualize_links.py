"""
Visualize and analyze the KPI linking structure.
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_links(links_file: Path):
    """Analyze the linking structure and show statistics."""
    
    with open(links_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    total_nodes = len(nodes)
    
    # Build index for quick lookup
    nodes_by_id = {n['kpi_id']: n for n in nodes}
    
    # Statistics
    linked_nodes = sum(1 for n in nodes if n['next_kpi_id'] or n['prev_kpi_id'])
    isolated_nodes = sum(1 for n in nodes if not n['next_kpi_id'] and not n['prev_kpi_id'])
    
    # Find all temporal chains
    chains = []
    visited = set()
    
    for node in nodes:
        if node['kpi_id'] in visited:
            continue
        
        # Find chain start (node with no prev)
        current = node
        while current['prev_kpi_id']:
            current = nodes_by_id[current['prev_kpi_id']]
        
        # Build chain from start
        chain = []
        while current:
            chain.append(current)
            visited.add(current['kpi_id'])
            current = nodes_by_id[current['next_kpi_id']] if current['next_kpi_id'] else None
        
        if len(chain) > 1:
            chains.append(chain)
        elif len(chain) == 1 and not chain[0]['next_kpi_id'] and not chain[0]['prev_kpi_id']:
            # Isolated node
            pass
    
    # Chain statistics
    chain_lengths = [len(c) for c in chains]
    
    print("=" * 80)
    print("KPI LINKING ANALYSIS")
    print("=" * 80)
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Linked nodes: {linked_nodes} ({linked_nodes/total_nodes*100:.1f}%)")
    print(f"   Isolated nodes: {isolated_nodes} ({isolated_nodes/total_nodes*100:.1f}%)")
    print(f"   Temporal chains: {len(chains)}")
    
    if chain_lengths:
        print(f"\n🔗 CHAIN STATISTICS:")
        print(f"   Average chain length: {sum(chain_lengths)/len(chain_lengths):.1f} years")
        print(f"   Longest chain: {max(chain_lengths)} years")
        print(f"   Shortest chain: {min(chain_lengths)} years")
    
    # Show distribution of chain lengths
    length_dist = defaultdict(int)
    for length in chain_lengths:
        length_dist[length] += 1
    
    print(f"\n📈 CHAIN LENGTH DISTRIBUTION:")
    for length in sorted(length_dist.keys()):
        count = length_dist[length]
        bar = "█" * min(50, count)
        print(f"   {length:2d} years: {count:4d} chains {bar}")
    
    # Show sample chains
    print(f"\n🔍 SAMPLE TEMPORAL CHAINS:")
    
    # Sort chains by length (longest first)
    chains.sort(key=len, reverse=True)
    
    for i, chain in enumerate(chains[:5]):
        print(f"\n   Chain {i+1}: {chain[0]['name']} - {chain[0]['key']}")
        print(f"   Length: {len(chain)} years")
        print(f"   Years: {min(n['year'] for n in chain)} → {max(n['year'] for n in chain)}")
        print(f"   Temporal sequence:")
        for node in chain:
            print(f"      {node['year']}: {node['value']} {node['units']}")
    
    # Group analysis
    print(f"\n📋 UNIQUE KPI GROUPS (name + key):")
    groups = defaultdict(list)
    for node in nodes:
        key = (node['name'], node['key'])
        groups[key].append(node)
    
    print(f"   Total unique groups: {len(groups)}")
    
    # Show groups with most temporal coverage
    group_stats = []
    for (name, key), gnodes in groups.items():
        years = sorted(set(n['year'] for n in gnodes if n['year']))
        if years:
            group_stats.append({
                'name': name,
                'key': key,
                'years': years,
                'count': len(years),
                'range': f"{min(years)}-{max(years)}" if years else "N/A"
            })
    
    group_stats.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\n   Top 10 groups by temporal coverage:")
    for i, g in enumerate(group_stats[:10], 1):
        print(f"   {i:2d}. {g['name']} | {g['key']}")
        print(f"       Coverage: {g['count']} years ({g['range']})")
        print(f"       Years: {g['years']}")

def main():
    base_dir = Path(__file__).parent
    links_file = base_dir / 'data' / 'output' / 'kpi_links' / 'links.json'
    
    if not links_file.exists():
        print(f"❌ Links file not found: {links_file}")
        return
    
    analyze_links(links_file)

if __name__ == "__main__":
    main()
