import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import re


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
        self.alternative_values: list = []  # Other values for same (name, key, year) when allow_multi_year=True
    
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
            "alternative_values": self.alternative_values,
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

def normalize_text(text: str) -> str:
    """
    Normalize text by removing special characters and footnote markers.
    
    Args:
        text: Input text to normalize
    
    Returns:
        Cleaned text with only letters, numbers, and spaces, lowercase
    """
    # Remove footnote markers like ^1, ^2, etc.
    text = re.sub(r'\^\d+', '', text)
    # Strip and lowercase first
    text = text.strip().lower()
    # Collapse multiple spaces/underscores then unify as underscores
    text = re.sub(r'[\s_]+', '_', text)
    return text


_KEY_STOPWORDS = {'the', 'a', 'an', 'of', 'and', 'or', 'in', 'for', 'with', 'by'}
_KEY_NAMING_JACCARD_THRESHOLD = 0.5


def _keys_naming_similar(k1: str, k2: str) -> bool:
    """
    True if two keys plausibly refer to the same entity under a different label.

    Two checks (either is sufficient):
      1. Subset  — significant words of the shorter key are all present in the longer
                   (e.g. 'Porsche' ⊂ 'Porsche Automotive', 'TRATON' ⊂ 'TRATON GROUP').
      2. Jaccard — word-set overlap ≥ 0.5, catching symmetric partial matches
                   (e.g. 'Audi (Premium Brand Group)' ↔ 'Audi (Progressive Brand Group)':
                    intersection={audi,brand,group} → 3/5 = 0.6).
    """
    n1 = re.sub(r'[\s_()+]+', ' ', k1.strip().lower())
    n2 = re.sub(r'[\s_()+]+', ' ', k2.strip().lower())
    if n1 == n2:
        return False

    words1 = {w for w in n1.split() if w not in _KEY_STOPWORDS and len(w) > 1}
    words2 = {w for w in n2.split() if w not in _KEY_STOPWORDS and len(w) > 1}

    if not words1 or not words2:
        return False

    # Subset check
    shorter, longer = (words1, words2) if len(words1) <= len(words2) else (words2, words1)
    if shorter.issubset(longer):
        return True

    # Jaccard check
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    if union > 0 and intersection / union >= _KEY_NAMING_JACCARD_THRESHOLD:
        return True

    return False


def _cont_val_match(v1, v2, tol: float = 1e-4) -> bool:
    try:
        return abs(float(v1) - float(v2)) <= tol
    except (TypeError, ValueError):
        return str(v1).strip().lower() == str(v2).strip().lower()


def merge_continuation_chains(temporal_chains: list) -> list:
    """
    Detect chains that continue one another due to entity renames/restructures
    (e.g. 'TRATON GROUP' → 'TRATON', 'Audi (Premium Brand Group)' → 'Audi (Progressive Brand Group)')
    and merge them into a single chain.

    Matching priority:
      1. Value match  — the boundary value of chain A equals the start value of chain B
                        (gap=0, shared year). This is the PRIMARY signal: same metric, same
                        year, same value almost certainly means the same entity under a new name.
      2. Naming similarity — keys share significant word overlap (subset or Jaccard ≥ 0.5).
                        Used as the SECONDARY confirmation for gap=0 and as the sole criterion
                        for gap=1 (adjacent years, where no shared value is available).

    The merged chain carries a ``key_aliases`` list recording which entity name was used
    for each year segment, making the rename history explicit and auditable.
    """
    chain_map = {c["chain_id"]: c for c in temporal_chains}

    # Group chains by (normalized name, units)
    groups: dict = {}
    for c in temporal_chains:
        name_norm = re.sub(r'[\s_]+', ' ', c["name"].strip().lower())
        nodes = c.get("nodes", [])
        units = str(nodes[0].get("units", "") if nodes else "").strip().lower()
        groups.setdefault((name_norm, units), []).append(c["chain_id"])

    merged_ids: set = set()
    merged_chains: list = []

    for (_name_norm, _units), cids in groups.items():
        if len(cids) < 2:
            continue
        chains_in_group = [chain_map[cid] for cid in cids if cid in chain_map]
        chains_in_group.sort(key=lambda c: c["year_range"]["start"])

        # Greedy forward scan: extend the tail of an open sequence when a match is found
        sequences: list = []  # each element is an ordered list of chain dicts

        for chain in chains_in_group:
            matched = False
            for seq in sequences:
                tail = seq[-1]
                tail_nodes = tail.get("nodes", [])
                head_nodes = chain.get("nodes", [])
                if not tail_nodes or not head_nodes:
                    continue

                gap = chain["year_range"]["start"] - tail["year_range"]["end"]
                if gap not in (0, 1):
                    continue

                if gap == 0:
                    # PRIMARY: boundary value must match (same data point reported under new name)
                    if not _cont_val_match(tail_nodes[-1].get("value"), head_nodes[0].get("value")):
                        continue
                    # SECONDARY: naming similarity confirms it's the same entity
                    if not _keys_naming_similar(tail["key"], chain["key"]):
                        continue
                else:  # gap == 1 — no shared year to compare values
                    # Naming similarity is the only available signal
                    if not _keys_naming_similar(tail["key"], chain["key"]):
                        continue

                seq.append(chain)
                matched = True
                break

            if not matched:
                sequences.append([chain])

        for seq in sequences:
            if len(seq) < 2:
                continue

            for c in seq:
                merged_ids.add(c["chain_id"])

            # Build merged node list and key_aliases
            key_aliases = []
            all_nodes: list = []
            seen_kpi_ids: set = set()

            for i, c in enumerate(seq):
                seg_nodes = c.get("nodes", [])
                if not seg_nodes:
                    continue

                # When boundary year is shared (gap=0), the first node of the later chain
                # is the same data point as the last node of the earlier chain — skip it.
                start_idx = 0
                if i > 0:
                    prev_end_year = seq[i - 1]["year_range"]["end"]
                    if c["year_range"]["start"] == prev_end_year:
                        start_idx = 1

                for nd in seg_nodes[start_idx:]:
                    if nd["kpi_id"] not in seen_kpi_ids:
                        all_nodes.append(nd)
                        seen_kpi_ids.add(nd["kpi_id"])

                key_aliases.append({
                    "key": c["key"],
                    "year_start": seg_nodes[0]["year"],
                    "year_end": seg_nodes[-1]["year"],
                })

            if not all_nodes:
                continue

            # Rewrite prev/next links to form a single continuous chain
            for j, nd in enumerate(all_nodes):
                nd["prev_kpi_id"] = all_nodes[j - 1]["kpi_id"] if j > 0 else None
                nd["next_kpi_id"] = all_nodes[j + 1]["kpi_id"] if j < len(all_nodes) - 1 else None

            canonical_key = seq[-1]["key"]   # most recent entity name
            merged_name = seq[0]["name"]

            merged_chains.append({
                "chain_id": f"{normalize_text(merged_name)} {normalize_text(canonical_key)}",
                "name": merged_name,
                "key": canonical_key,
                "key_aliases": key_aliases,
                "is_continuation_merge": True,
                "length": len(all_nodes),
                "year_range": {
                    "start": all_nodes[0]["year"],
                    "end": all_nodes[-1]["year"],
                },
                "nodes": all_nodes,
            })

    result = [c for c in temporal_chains if c["chain_id"] not in merged_ids]
    result.extend(merged_chains)
    result.sort(key=lambda c: (c["name"].lower(), c["key"].lower()))

    n_merged = len(merged_chains)
    n_consumed = len(merged_ids)
    print(f"   ↳ Continuation merging: {n_consumed} chains collapsed into {n_merged} merged chains "
          f"(net −{n_consumed - n_merged} chains)")
    return result


def addGroup(groups, key):
    # Normalize both name and key using the normalize_text function
    clean_name = normalize_text(key[0])
    clean_key = normalize_text(key[1])
    normalised_key = (clean_name, clean_key)
    if(normalised_key not in groups):
        groups[normalised_key] = []
    
    return normalised_key
    
     

def addKey(group, node, allow_multi_year: bool = False):
    # Check if the group already has a node with the same year
    existing_node = next((n for n in group if n.year == node.year), None)
    
    if existing_node:
        try:
            same_value = abs(float(existing_node.value) - float(node.value)) < 1e-6
        except (TypeError, ValueError):
            same_value = str(existing_node.value) == str(node.value)
        if same_value:
            # Exact same value — true duplicate, skip
            return False
        else:
            if allow_multi_year:
                # Different value for same year — store as alternative on the primary node
                existing_node.alternative_values.append({
                    "kpi_id": node.kpi_id,
                    "value": node.value,
                    "units": node.units,
                    "table_id": node.metadata.get("table_id"),
                    "source_model": node.metadata.get("source_model"),
                })
                return False
            else:
                # Default: treat as separate isolated node (old behavior)
                return True
    group.append(node)
    return True


def link_kpis(kpis: List[Dict], allow_multi_year: bool = False) -> List[GraphNode]:
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
        # Skip KPIs with None value
        if kpi.get("value") is None or kpi.get("year") is None:
            continue
            
        # Extract evidence data if available; fallback to top-level fields for tenth-trial format
        evidence = kpi.get("evidence") or {}
        
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
            row_idx=evidence.get("row_idx", kpi.get("row_idx")),
            col_idx=evidence.get("col_idx", kpi.get("col_idx")),
            row_name=evidence.get("row_name", kpi.get("row_name")),
            col_name=evidence.get("col_name", kpi.get("col_name")),
            table_id=evidence.get("table_id", kpi.get("table_id")),
            source_model=evidence.get("source_model", kpi.get("source_model"))
        )
             
        # Group by (name, key)
        group_key = (kpi.get("name"), kpi.get("key"))
        normalised_key = addGroup(groups, group_key)
        res = addKey(groups[normalised_key], node, allow_multi_year=allow_multi_year)

        if not res:
            continue
        else:
            nodes_map[kpi_id] = node
    
    # Link nodes within each group by year
    print(f"\nTotal groups formed for linking: {len(groups)}")
    print(f"Sample group keys:", list(groups.items())[:50])
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
    Save the KPI graph to JSON format with temporal chains in correct order.
    
    Args:
        nodes: List of GraphNode objects
        output_path: Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all chain starts (nodes with no previous node)
    chain_starts = [n for n in nodes if n.prev is None]
    
    # Build chains by following next links
    temporal_chains = []
    processed = set()
    
    for start_node in chain_starts:
        if start_node.kpi_id in processed:
            continue
            
        chain = []
        current = start_node
        
        # Follow the chain to the end
        while current is not None:
            chain.append(current)
            processed.add(current.kpi_id)
            current = current.next
        
        # Only add chains with at least one node
        if chain:
            temporal_chains.append({
                "chain_id": f"{normalize_text(chain[0].name)} {normalize_text(chain[0].key)}",
                "name": chain[0].name,
                "key": chain[0].key,
                "length": len(chain),
                "year_range": {
                    "start": chain[0].year,
                    "end": chain[-1].year
                },
                "nodes": [node.to_dict() for node in chain]
            })
    
    # Sort chains by name, then key
    temporal_chains.sort(key=lambda c: (c["name"].lower(), c["key"].lower()))

    # Merge chains that are continuations of each other due to entity renames
    print(f"\n   Detecting continuation chains...")
    temporal_chains = merge_continuation_chains(temporal_chains)

    # Find isolated nodes (no prev and no next)
    isolated_nodes = [n for n in nodes if n.prev is None and n.next is None and n.kpi_id not in processed]

    graph_data = {
        "total_nodes": len(nodes),
        "total_chains": len(temporal_chains),
        "continuation_merges": sum(1 for c in temporal_chains if c.get("is_continuation_merge")),
        "total_isolated_nodes": len(isolated_nodes),
        "temporal_chains": temporal_chains,
        "isolated_nodes": [node.to_dict() for node in isolated_nodes]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Graph saved with {len(nodes)} nodes to: {output_path}")
    print(f"   - {len(temporal_chains)} temporal chains")
    print(f"   - {len(isolated_nodes)} isolated nodes")

    return temporal_chains


def load_tables(kpi_file: Path) -> List[Dict]:
    """
    Load table data from a KPI extraction file.
    
    Args:
        kpi_file: Path to KPI JSON file with year_*_kpis.json format
    
    Returns:
        List of table dictionaries from the results section
    """
    with open(kpi_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("results", [])


def load_kpis(kpi_directory: Path) -> List[Dict]:
    """
    Load KPIs from all year_*_kpis.json files in a directory.
    
    Expected file structure:
    {
      "metadata": {"year": 2015, ...},
      "results": [
        {
          "kpis": [...],
          "table_id": "...",
          "table_data": {...}
        }
      ]
    }
    
    Args:
        kpi_directory: Directory containing year_*_kpis.json files
    
    Returns:
        List of all KPI dictionaries with evidence metadata
    """
    all_kpis = []
    
    # Find all KPI files
    kpi_files = sorted(kpi_directory.glob("*year*_kpis.json"))
    
    if not kpi_files:
        print(f"⚠️  No KPI files found in {kpi_directory}")
        return all_kpis
    
    print(f"\nLoading KPIs from {len(kpi_files)} files:")
    
    for file in kpi_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get("metadata", {})
        file_year = metadata.get("year")
        file_kpis = 0
        
        for result in data.get("results", []):
            table_id = result.get("table_id") or result.get("table_data", {}).get("table_id")
            
            for kpi in result.get("kpis", []):
                evidence = {
                    "row_idx": kpi.get("row_idx"),
                    "col_idx": kpi.get("col_idx"),
                    "row_name": kpi.get("row_name"),
                    "col_name": kpi.get("col_name"),
                    "table_id": table_id,
                    "source_model": kpi.get("source_model")
                }
                
                all_kpis.append({
                    **kpi,
                    "year": kpi.get("year", file_year),
                    "evidence": evidence
                })
                file_kpis += 1
        
        print(f"  ✓ {file.name}: {file_kpis} KPIs")
    
    print(f"\n✓ Total KPIs loaded: {len(all_kpis)}")
    return all_kpis


def load_kpis_flat(kpi_directory: Path) -> List[Dict]:
    """
    Load KPIs from individual *_kpis.json files in a directory (per-table format).

    Expected file structure:
    {
      "kpis": [{"name": ..., "key": ..., "value": ..., "year": ..., ...}],
      "metadata": {...},
      "model": "...",
      ...
    }

    Args:
        kpi_directory: Directory containing individual *_kpis.json files

    Returns:
        List of all KPI dictionaries with evidence metadata
    """
    all_kpis = []

    kpi_files = sorted(kpi_directory.glob("*_kpis.json"))

    if not kpi_files:
        print(f"⚠️  No KPI files found in {kpi_directory}")
        return all_kpis

    print(f"\nLoading KPIs from {len(kpi_files)} files:")
    files_with_kpis = 0

    for file in kpi_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        table_id = file.stem.replace("_kpis", "")
        kpis = data.get("kpis", [])

        if kpis:
            files_with_kpis += 1
            print(f"  ✓ {file.name}: {len(kpis)} KPIs")

        for kpi in kpis:
            evidence = {
                "row_idx": kpi.get("row_idx"),
                "col_idx": kpi.get("col_idx"),
                "row_name": kpi.get("row_name"),
                "col_name": kpi.get("col_name"),
                "table_id": table_id,
                "source_model": kpi.get("source_model")
            }
            all_kpis.append({**kpi, "evidence": evidence})

    print(f"\n✓ Total KPIs loaded: {len(all_kpis)} from {files_with_kpis}/{len(kpi_files)} files")
    return all_kpis


def load_kpis_validated(kpi_directory: Path) -> List[Dict]:
    """
    Load KPIs from trial-*-validation directories.

    Expected file structure per table:
    {
      "source_file": "..._kpis.json",
      "year": 2022,
      "valid_kpis":   [{"kpi": {...}, "validation": {"selected_table_id": "VW2022_Txxxxxx", ...}}],
      "invalid_kpis": [{"kpi": {...}, "validation": {...}}]
    }

    Only valid_kpis are included. Entries where selected_table_id is None/missing are
    skipped (no reference table in the DB — the extraction could not be grounded).
    """
    all_kpis = []

    kpi_files = sorted(kpi_directory.glob("*_kpis.json"))
    if not kpi_files:
        print(f"⚠️  No KPI files found in {kpi_directory}")
        return all_kpis

    print(f"\nLoading validated KPIs from {len(kpi_files)} files:")
    files_with_kpis = 0
    skipped_no_table = 0

    for file in kpi_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_year = data.get("year")
        file_kpis = 0

        for entry in data.get("valid_kpis", []):
            kpi = entry.get("kpi", {})
            validation = entry.get("validation", {})

            selected_table_id = validation.get("selected_table_id")
            if not selected_table_id:
                skipped_no_table += 1
                continue

            evidence = {
                "row_idx": validation.get("row_idx", kpi.get("row_idx")),
                "col_idx": validation.get("col_idx", kpi.get("col_idx")),
                "row_name": validation.get("row_name_match", kpi.get("row_name")),
                "col_name": validation.get("col_name_match", kpi.get("col_name")),
                "table_id": selected_table_id,
                "source_model": kpi.get("source_model"),
            }
            all_kpis.append({
                **kpi,
                "year": kpi.get("year") or file_year,
                "evidence": evidence,
            })
            file_kpis += 1

        if file_kpis:
            files_with_kpis += 1
            print(f"  ✓ {file.name}: {file_kpis} valid KPIs")

    print(f"\n✓ Total KPIs loaded: {len(all_kpis)} from {files_with_kpis}/{len(kpi_files)} files")
    if skipped_no_table:
        print(f"  ⚠ Skipped {skipped_no_table} valid_kpis with no selected_table_id (no DB reference)")
    return all_kpis


def analyze_complete_chains(temporal_chains: list, start_year: int = 2015, end_year: int = 2021) -> Dict:
    """
    Analyze temporal chains that span the complete year range.

    Args:
        temporal_chains: List of chain dicts as returned by save_graph (post-merge).
        start_year: Starting year for a complete chain.
        end_year: Ending year for a complete chain.

    Returns:
        Dictionary with statistics about complete chains.
    """
    print(f"\n{'='*80}")
    print(f"COMPLETE CHAIN ANALYSIS ({start_year}-{end_year})")
    print(f"{'='*80}")

    complete_chains = []
    partial_chains = []
    single_node_chains = []

    for chain in temporal_chains:
        nodes = chain.get("nodes", [])
        years_in_chain = sorted(set(n["year"] for n in nodes if n.get("year") is not None))

        if len(nodes) <= 1:
            single_node_chains.append(chain)
            continue

        expected_years = set(range(start_year, end_year + 1))
        actual_years = set(years_in_chain)
        is_complete = actual_years >= expected_years

        chain_info = {
            **chain,
            "years": years_in_chain,
            "start_year": years_in_chain[0] if years_in_chain else None,
            "end_year": years_in_chain[-1] if years_in_chain else None,
            "missing_years": sorted(expected_years - actual_years),
        }

        if is_complete:
            complete_chains.append(chain_info)
        else:
            partial_chains.append(chain_info)

    total_chains = len(temporal_chains)

    # Print summary
    print(f"\nTotal chains: {total_chains}")
    print(f"Complete chains ({start_year}-{end_year}): {len(complete_chains)}")
    print(f"Partial chains (2+ nodes): {len(partial_chains)}")
    print(f"Single-node chains: {len(single_node_chains)}")
    print(f"Completeness rate: {len(complete_chains)/total_chains*100:.1f}%")

    # Print top complete chains by name
    print(f"\n{'-'*80}")
    print(f"TOP 20 COMPLETE CHAINS")
    print(f"{'-'*80}")

    for i, chain in enumerate(sorted(complete_chains, key=lambda c: (c['name'], c['key']))[:20], 1):
        values = [n["value"] for n in chain['nodes']]
        print(f"\n{i}. {chain['name']} | {chain['key']}")
        print(f"   Years: {chain['years']}")
        print(f"   Values: {values}")
    
    # Statistics by KPI name
    print(f"\n{'-'*80}")
    print(f"COMPLETE CHAINS BY KPI NAME")
    print(f"{'-'*80}")

    kpi_name_stats = {}
    for chain in complete_chains:
        name = chain['name']
        if name not in kpi_name_stats:
            kpi_name_stats[name] = []
        kpi_name_stats[name].append(chain)

    sorted_names = sorted(kpi_name_stats.items(), key=lambda x: len(x[1]), reverse=True)
    for name, chains in sorted_names[:15]:
        print(f"\n{name}: {len(chains)} complete chains")
        example_keys = [c['key'] for c in chains[:3]]
        if len(chains) > 3:
            print(f"   Examples: {', '.join(example_keys)}... (+{len(chains)-3} more)")
        else:
            print(f"   Keys: {', '.join(example_keys)}")

    # Analyze missing years for partial chains
    print(f"\n{'-'*80}")
    print(f"PARTIAL CHAIN ANALYSIS")
    print(f"{'-'*80}")

    missing_year_counts = {}
    for chain in partial_chains:
        for year in chain['missing_years']:
            missing_year_counts[year] = missing_year_counts.get(year, 0) + 1

    print(f"\nMissing years frequency:")
    for year in sorted(missing_year_counts.keys()):
        print(f"  {year}: {missing_year_counts[year]} chains missing this year")

    # Show examples of partial chains
    print(f"\n{'-'*80}")
    print(f"EXAMPLES OF PARTIAL CHAINS (Missing Years)")
    print(f"{'-'*80}")

    for i, chain in enumerate(sorted(partial_chains, key=lambda c: len(c['missing_years']), reverse=False)[:10], 1):
        print(f"\n{i}. {chain['name']} | {chain['key']}")
        print(f"   Has years: {chain['years']}")
        print(f"   Missing: {chain['missing_years']}")

    return {
        "total_chains": total_chains,
        "complete_chains": len(complete_chains),
        "partial_chains": len(partial_chains),
        "single_node_chains": len(single_node_chains),
        "completeness_rate": len(complete_chains)/total_chains*100,
        "complete_chain_details": complete_chains,
        "partial_chain_details": partial_chains,
        "single_node_chain_details": single_node_chains,
        "kpi_name_stats": {name: len(chains) for name, chains in kpi_name_stats.items()}
    }


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


def main(trial_dir: Optional[str] = None, allow_multi_year: bool = False):
    """
    Main function to link KPIs across years using (name, key) grouping.

    Args:
        trial_dir: Path to trial directory containing KPI JSON files.
                   Supports both year_*_kpis.json (batched) and individual *_kpis.json formats.
                   If None, defaults to trial-12.
        allow_multi_year: If True, nodes with the same (name, key, year) but different values
                          are stored as alternative_values on the primary node instead of
                          becoming isolated chains.
    """
    from datetime import date
    base_dir = Path(__file__).parent

    # Determine input directory
    if trial_dir:
        kpi_dir = Path(trial_dir).resolve()
        # If the resolved path doesn't exist, try relative to the workspace root (base_dir.parent)
        if not kpi_dir.exists():
            fallback = (base_dir.parent / trial_dir).resolve()
            if fallback.exists():
                kpi_dir = fallback
    else:
        kpi_dir = base_dir.parent / 'data' / 'output' / 'trial-12'

    # Derive output subdir name from the last two path components (e.g., trial-24_vlm_qwen_72b)
    parts = kpi_dir.parts
    output_name = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]
    output_dir = base_dir.parent / 'data' / 'kpi_links' / output_name

    print(f"\n{'='*80}")
    print(f"KPI TEMPORAL LINKING")
    print(f"{'='*80}")
    print(f"Input directory: {kpi_dir}")
    print(f"Output directory: {output_dir}")

    # Auto-detect format
    year_files = list(kpi_dir.glob("*year*_kpis.json"))
    sample_files = list(kpi_dir.glob("*_kpis.json"))[:3]
    is_validated = any(
        "valid_kpis" in json.loads(f.read_text(encoding="utf-8"))
        for f in sample_files
        if f.exists()
    ) if sample_files else False

    if year_files:
        print(f"Detected batched format ({len(year_files)} year files)")
        all_kpis = load_kpis(kpi_dir)
    elif is_validated:
        print("Detected validation format (valid_kpis/invalid_kpis structure)")
        print("  → Only valid_kpis with a DB-grounded selected_table_id will be loaded")
        all_kpis = load_kpis_validated(kpi_dir)
    else:
        print("Detected per-table format (individual *_kpis.json files)")
        all_kpis = load_kpis_flat(kpi_dir)
    
    if not all_kpis:
        print("\n❌ No KPIs found. Exiting.")
        return
    
    # Build temporal graph using name + key
    print(f"\n{'='*80}")
    print("BUILDING TEMPORAL GRAPH")
    print(f"{'='*80}")
    nodes = link_kpis(all_kpis, allow_multi_year=allow_multi_year)
    
    # Calculate statistics
    linked_nodes = sum(1 for n in nodes if n.next or n.prev)
    chain_starts = sum(1 for n in nodes if n.prev is None and n.next is not None)
    isolated_nodes = sum(1 for n in nodes if n.next is None and n.prev is None)
    
    print(f"\nLinking Statistics:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Linked nodes: {linked_nodes}")
    print(f"  Temporal chains: {chain_starts}")
    print(f"  Isolated nodes: {isolated_nodes}")
    
    # Save graph to JSON
    output_file = output_dir / 'links.json'
    temporal_chains = save_graph(nodes, output_file)

    # Determine year range from the merged chains (same source as links.json)
    all_years = [n["year"] for c in temporal_chains for n in c.get("nodes", []) if n.get("year") is not None]
    min_year = min(all_years) if all_years else 2015
    max_year = max(all_years) if all_years else 2021

    # Analyze complete chains across the detected year range
    chain_stats = analyze_complete_chains(temporal_chains, start_year=min_year, end_year=max_year)

    # Save statistics to JSON
    stats_file = output_dir / 'chain_statistics.json'
    stats_data = {
        "analysis_date": date.today().isoformat(),
        "year_range": {"start": min_year, "end": max_year},
        "summary": {
            "total_chains": chain_stats["total_chains"],
            "complete_chains": chain_stats["complete_chains"],
            "partial_chains": chain_stats["partial_chains"],
            "single_node_chains": chain_stats["single_node_chains"],
            "completeness_rate": chain_stats["completeness_rate"]
        },
        "complete_chains_by_kpi": chain_stats["kpi_name_stats"],
        "complete_chains": [
            {
                "name": c["name"],
                "key": c["key"],
                "length": c["length"],
                "years": c["years"],
                "values": [n["value"] for n in c["nodes"]]
            }
            for c in chain_stats["complete_chain_details"]
        ]
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Statistics saved to: {stats_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys

    allow_multi_year = "--allow-multi-year" in sys.argv
    if allow_multi_year:
        sys.argv.remove("--allow-multi-year")
        print("Multi-year mode: duplicate-year nodes stored as alternative_values\n")

    trial_dir = None
    if len(sys.argv) > 1:
        trial_dir = sys.argv[1]
        print(f"Using directory: {trial_dir}\n")
    else:
        print("Using default directory: trial-12")
        print("Usage: python link_kpis.py [path/to/trial-directory] [--allow-multi-year]\n")

    main(trial_dir=trial_dir, allow_multi_year=allow_multi_year)


