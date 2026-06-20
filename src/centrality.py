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
from typing import Any, Dict, List, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point

from geo_utils import assert_projected_crs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_betweenness_centrality(
    graph: nx.Graph,
    weight: str = "weight",
    k: Optional[int] = None,
    seed: int = 42,
) -> Dict[Any, float]:
    """
    Compute betweenness centrality on the healed road graph.

    Args:
        graph: Healed NetworkX road graph (nodes with 'x','y' in projected CRS).
        weight: Edge attribute to use as travel cost.
        k: Number of sample source nodes for approximate centrality.
           None = exact (all-pairs). Recommend k=500 for graphs >2000 nodes
           to keep runtime under ~30s on a laptop CPU.
        seed: Random seed for reproducible sampling when k is set.

    Returns:
        Dict mapping node_id → betweenness centrality score in [0, 1].
    """
    assert_projected_crs(graph)

    n = len(graph)
    if n == 0:
        return {}

    # Auto-set k for large graphs if caller didn't specify
    effective_k = k
    if effective_k is None and n > 2000:
        effective_k = min(500, n)
        logger.info(f"Graph has {n} nodes; using approximate centrality with k={effective_k}")

    centrality = nx.betweenness_centrality(
        graph,
        k=effective_k,
        weight=weight,
        normalized=True,
        seed=seed,
    )

    logger.info(
        f"Betweenness centrality computed for {n} nodes. "
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
