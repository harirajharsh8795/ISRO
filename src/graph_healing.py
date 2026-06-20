"""
Phase 3: Topological Reconstruction (Graph Healing)
Maps to:
  - 4 Core Objectives:
    * Topological Reconstruction (MST + Disjoint Set gap-bridging)
  - Evaluation Parameters:
    * Connectivity Ratio (% increase in largest connected component size)
    * Topological Accuracy (Average Path Length comparison)
"""

import logging
from typing import Tuple, List, Set, Dict, Any
import numpy as np
import networkx as nx
from shapely.geometry import LineString
from geo_utils import assert_projected_crs

# Set up logging for feedback during execution
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class UnionFind:
    """
    An efficient Disjoint Set (Union-Find) data structure with path compression
    and union by rank, used to track connected components during MST healing.
    """
    def __init__(self, elements: range):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            # Union by rank to keep the tree flat
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False


def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """
    Computes the absolute angle in degrees between two 2D vectors.
    """
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = np.hypot(v1[0], v1[1])
    norm_v2 = np.hypot(v2[0], v2[1])
    
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
        
    cos_angle = dot_product / (norm_v1 * norm_v2)
    # Clip to handle potential numerical precision issues outside [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def is_aligned_at_node(
    graph: nx.Graph,
    node: Any,
    bridge_vec: Tuple[float, float],
    tolerance_deg: float
) -> bool:
    """
    Checks if the bridge vector aligns with the extension of any road segment
    connected to the candidate node.
    """
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        # Isolated node has no road direction to align with.
        # Accept it to allow connecting isolated segments.
        return True

    nx_node = graph.nodes[node]['x']
    ny_node = graph.nodes[node]['y']

    for nbr in neighbors:
        nbr_x = graph.nodes[nbr]['x']
        nbr_y = graph.nodes[nbr]['y']
        
        # Road vector points from neighbor to the candidate node (dead end).
        # We check if the bridge vector continues in a similar direction.
        road_vec = (nx_node - nbr_x, ny_node - nbr_y)
        
        angle = angle_between_vectors(road_vec, bridge_vec)
        if angle <= tolerance_deg:
            return True
            
    return False


def check_alignment(
    graph: nx.Graph,
    u: Any,
    v: Any,
    tolerance_deg: float
) -> bool:
    """
    Validates the mutual angular alignment of a proposed bridge between nodes u and v.
    """
    ux, uy = graph.nodes[u]['x'], graph.nodes[u]['y']
    vx, vy = graph.nodes[v]['x'], graph.nodes[v]['y']
    
    # Vector from u pointing towards v
    vec_uv = (vx - ux, vy - uy)
    # Vector from v pointing towards u
    vec_vu = (-vec_uv[0], -vec_uv[1])
    
    # Bridge must satisfy angular alignment at both endpoints to ensure continuity
    if not is_aligned_at_node(graph, u, vec_uv, tolerance_deg):
        return False
    if not is_aligned_at_node(graph, v, vec_vu, tolerance_deg):
        return False
        
    return True


def heal_graph_connectivity(
    graph: nx.Graph,
    max_bridge_distance_m: float = 50.0,
    angular_tolerance_deg: float = 30.0
) -> nx.Graph:
    """
    Connects disjoint components of a road network graph using a Minimum Spanning Tree
    (Kruskal's style) approach over filtered candidate bridging edges.

    Assumptions:
        - Input graph is undirected (nx.Graph or nx.MultiGraph).
        - Nodes have 'x' and 'y' attributes in a projected coordinate system (e.g. UTM meters).

    Args:
        graph: NetworkX graph built from a skeletonized road mask.
        max_bridge_distance_m: Maximum geographic distance in meters to bridge a gap.
        angular_tolerance_deg: Maximum allowable deviation in degrees for road continuity.

    Returns:
        The healed graph with bridge edges added and tagged with healed=True.
    """
    # Validate that coordinates are in projected meters, not geographic degrees
    assert_projected_crs(graph)

    # Create a copy to prevent in-place mutation of the input graph
    healed_graph = graph.copy()
    
    # 1. Identify all connected components
    components = [list(c) for c in nx.connected_components(graph)]
    num_components = len(components)
    
    logger.info(f"Starting healing process. Initial components: {num_components}")
    if num_components <= 1:
        logger.info("Graph is already fully connected or empty. No healing required.")
        return healed_graph

    # 2. Identify candidate endpoint nodes (dead ends / degree 1 nodes) per component.
    # Fall back to all nodes in the component if no degree 1 nodes exist (e.g. isolated loops/singletons).
    component_candidates: List[List[Any]] = []
    for comp in components:
        candidates = [node for node in comp if graph.degree(node) == 1]
        if not candidates:
            candidates = comp
        component_candidates.append(candidates)

    # 3. Generate all valid candidate bridges between different components
    candidate_bridges = []
    
    for i in range(num_components):
        for j in range(i + 1, num_components):
            # Evaluate pairs of candidate nodes between component i and component j
            for u in component_candidates[i]:
                ux, uy = graph.nodes[u]['x'], graph.nodes[u]['y']
                for v in component_candidates[j]:
                    vx, vy = graph.nodes[v]['x'], graph.nodes[v]['y']
                    
                    dist = float(np.hypot(ux - vx, uy - vy))
                    if dist <= max_bridge_distance_m:
                        # Validate road alignment at both ends
                        if check_alignment(graph, u, v, angular_tolerance_deg):
                            candidate_bridges.append((dist, u, v, i, j))

    # 4. Sort candidates by distance (Kruskal's greedy choice)
    candidate_bridges.sort(key=lambda x: x[0])
    
    # 5. Apply Union-Find to build a Minimum Spanning Tree of components
    uf = UnionFind(range(num_components))
    bridges_added = 0
    
    for dist, u, v, comp_u, comp_v in candidate_bridges:
        # Check if the endpoints belong to components that are not yet connected
        if uf.union(comp_u, comp_v):
            ux, uy = graph.nodes[u]['x'], graph.nodes[u]['y']
            vx, vy = graph.nodes[v]['x'], graph.nodes[v]['y']
            
            # Construct a shapely LineString for GIS compatibility
            geom = LineString([(ux, uy), (vx, vy)])
            
            # Add the bridge edge to the graph
            healed_graph.add_edge(
                u, v,
                weight=dist,
                healed=True,
                geometry=geom
            )
            bridges_added += 1

    # Calculate final component count
    final_components = len(list(nx.connected_components(healed_graph)))
    logger.info(f"Healing complete. Added {bridges_added} bridge edges. "
                f"Remaining components: {final_components}")
    
    return healed_graph
