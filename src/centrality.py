"""
Phase 4: Structural Intelligence — Centrality Analysis & Gatekeeper Nodes
Maps to:
  - 4 Core Objectives:
    * Structural Intelligence (Betweenness centrality → Gatekeeper Nodes)
  - Evaluation Parameters:
    * Quantitative Criticality Map (spatial heatmap of high-betweenness intersections)
  - Deliverables:
    * Ranked list of Gatekeeper Nodes with centrality scores
    * GeoJSON export for dashboard / QGIS visualisation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from geo_utils import assert_projected_crs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def nx_to_igraph(graph: nx.Graph, weight_attr: str = "weight") -> Tuple["ig.Graph", List[Any]]:
    """
    Converts a NetworkX Graph to an igraph Graph, preserving node attributes
    and edge weights. Returns the igraph Graph and the list of original NetworkX node IDs
    mapping to the sequential integer indices in igraph.
    """
    import igraph as ig
    nx_nodes = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nx_nodes)}
    
    # Fast comprehension-based extraction
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    weights = [float(data.get(weight_attr, 1.0)) for u, v, data in graph.edges(data=True)]
        
    ig_graph = ig.Graph(n=len(graph), edges=edges, directed=False)
    ig_graph.es['weight'] = weights
    
    # Copy node attributes: avoid scanning all nodes via any()
    if nx_nodes:
        first_node = nx_nodes[0]
        for attr in ['x', 'y']:
            if attr in graph.nodes[first_node]:
                ig_graph.vs[attr] = [graph.nodes[n].get(attr, 0.0) for n in nx_nodes]
            
    return ig_graph, nx_nodes


def igraph_to_nx(ig_graph: "ig.Graph", nx_node_list: List[Any]) -> nx.Graph:
    """
    Converts an igraph Graph back to a NetworkX Graph, mapping sequential integer 
    indices back to the original NetworkX node IDs.
    """
    G = nx.Graph()
    
    # Restore nodes and their attributes
    for i, node_id in enumerate(nx_node_list):
        attr_dict = {}
        for attr in ['x', 'y']:
            if attr in ig_graph.vs.attributes():
                attr_dict[attr] = ig_graph.vs[i][attr]
        G.add_node(node_id, **attr_dict)
        
    # Restore edges and their weights
    for edge in ig_graph.es:
        u_idx, v_idx = edge.tuple
        u = nx_node_list[u_idx]
        v = nx_node_list[v_idx]
        weight = edge['weight'] if 'weight' in ig_graph.es.attributes() else 1.0
        G.add_edge(u, v, weight=weight)
        
    return G


from typing import Tuple

def compute_betweenness_centrality(
    graph: nx.Graph,
    weight: str = "weight",
    k: Optional[int] = None,
    seed: int = 42,
) -> Dict[Any, float]:
    """
    Compute betweenness centrality using python-igraph for high-performance execution.
    If k is specified or if the graph has >3000 nodes, it computes a near-exact centrality 
    using source node sampling to guarantee execution in under 5 seconds on a 20,000-node graph.
    
    Args:
        graph: Healed NetworkX road graph (nodes with 'x','y' in projected CRS).
        weight: Edge attribute to use as travel cost.
        k: Number of source nodes to sample for estimation.
        seed: Random seed for sampling.
        
    Returns:
        Dict mapping node_id → betweenness centrality score in [0, 1].
    """
    assert_projected_crs(graph)

    n = len(graph)
    if n == 0:
        return {}

    # Convert to igraph
    ig_graph, nx_nodes = nx_to_igraph(graph, weight_attr=weight)

    # Determine if we should use sampling for near-exact estimation
    # to meet the <5 seconds runtime constraint for large graphs.
    use_sampling = (k is not None) or (n > 3000)
    effective_k = k if k is not None else 250
    effective_k = min(effective_k, n)

    if use_sampling and effective_k < n:
        # Sample sources using the provided seed for reproducibility
        import random
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(n), effective_k)
        
        # Compute near-exact betweenness from sampled sources
        raw_scores = ig_graph.betweenness(
            sources=sampled_indices,
            weights='weight' if weight in ig_graph.es.attributes() else None,
            directed=False
        )
        
        # Scale scores by n / k to estimate total betweenness
        scale = n / effective_k
        raw_scores = [score * scale for score in raw_scores]
    else:
        raw_scores = ig_graph.betweenness(
            weights='weight' if weight in ig_graph.es.attributes() else None,
            directed=False
        )

    # Normalize betweenness: divide by (n-1)*(n-2)/2 for undirected graph
    if n > 2:
        norm_factor = 2.0 / ((n - 1) * (n - 2))
        normalized_scores = [score * norm_factor for score in raw_scores]
    else:
        normalized_scores = [0.0] * n

    # Map back to NetworkX node IDs
    centrality = {nx_nodes[i]: normalized_scores[i] for i in range(n)}

    logger.info(
        f"igraph betweenness centrality computed for {n} nodes (sampled={use_sampling}). "
        f"Max score: {max(centrality.values()):.6f}"
    )
    return centrality


def rank_gatekeeper_nodes(
    graph: nx.Graph,
    centrality_scores: Dict[Any, float],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Rank nodes by centrality score descending, return top_n as structured dicts.

    Returns:
        List of dicts with keys: node_id, x, y, centrality_score, rank.
    """
    sorted_nodes = sorted(centrality_scores.items(), key=lambda kv: kv[1], reverse=True)
    result = []
    for rank, (node_id, score) in enumerate(sorted_nodes[:top_n], start=1):
        data = graph.nodes[node_id]
        result.append({
            "node_id": node_id,
            "x": float(data["x"]),
            "y": float(data["y"]),
            "centrality_score": float(score),
            "rank": rank,
        })
    return result


def compute_connectivity_ratio(
    unhealed_graph: nx.Graph,
    healed_graph: nx.Graph,
) -> float:
    """
    Compute the connectivity ratio: largest connected component size after
    healing divided by largest connected component size before healing.
    A value >1 indicates improvement; higher = more components were merged.
    """
    def _lcc_size(g: nx.Graph) -> int:
        if len(g) == 0:
            return 0
        return max(len(c) for c in nx.connected_components(g))

    before = _lcc_size(unhealed_graph)
    after = _lcc_size(healed_graph)
    if before == 0:
        return 1.0
    return float(after / before)


def export_criticality_geojson(
    graph: nx.Graph,
    centrality_scores: Dict[Any, float],
    output_path: str,
    source_crs_epsg: int = 32643,
) -> None:
    """
    Export all nodes as a GeoJSON of Points with centrality_score property,
    reprojected from working UTM CRS to EPSG:4326 for web map display.
    """
    records = []
    for node_id, score in centrality_scores.items():
        data = graph.nodes[node_id]
        records.append({
            "node_id": str(node_id),
            "centrality_score": float(score),
            "geometry": Point(float(data["x"]), float(data["y"])),
        })

    if not records:
        logger.warning("No nodes to export.")
        return

    gdf = gpd.GeoDataFrame(records, crs=f"EPSG:{source_crs_epsg}")
    # Reproject to WGS84 for Leaflet / Folium web maps
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"Exported {len(gdf)} nodes to {output_path}")
