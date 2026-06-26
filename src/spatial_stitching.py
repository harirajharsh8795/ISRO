import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Any

def spatial_stitch_nodes(graph: nx.Graph, tolerance_m: float = 2.0) -> nx.Graph:
    """
    Identifies nodes in the graph that are spatially within `tolerance_m` meters
    and merges them to resolve boundary disconnects between adjacent tiles.
    
    Args:
        graph: NetworkX graph containing node attributes 'x' and 'y' (in projected meters).
        tolerance_m: Euclidean distance threshold for merging nodes.
        
    Returns:
        A new stitched NetworkX graph with boundary nodes merged.
    """
    if len(graph) < 2:
        return graph

    stitched_graph = graph.copy()
    nodes = list(stitched_graph.nodes(data=True))
    node_ids = [n[0] for n in nodes]
    coords = np.array([[n[1]['x'], n[1]['y']] for n in nodes])

    # Query KDTree for pairs within tolerance
    tree = KDTree(coords)
    pairs = tree.query_pairs(tolerance_m)

    if not pairs:
        return stitched_graph

    # Disjoint-set to group nodes that need to be merged together
    parent = {nid: nid for nid in node_ids}

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Union all close nodes if they belong to different tiles
    for idx_u, idx_v in pairs:
        u = node_ids[idx_u]
        v = node_ids[idx_v]
        
        # Check if they are from different tiles using the 'tile_{idx}_node' prefix
        u_tile = str(u).split("_node_")[0] if "_node_" in str(u) else "unknown"
        v_tile = str(v).split("_node_")[0] if "_node_" in str(v) else "unknown"
        
        if u_tile != v_tile or u_tile == "unknown":
            union(u, v)

    # Group nodes by their representative root
    groups = {}
    for nid in node_ids:
        root = find(nid)
        if root != nid:
            groups.setdefault(root, []).append(nid)

    # Merge groups into the representative root node
    for root, members in groups.items():
        all_members = [root] + members
        xs = [stitched_graph.nodes[m]['x'] for m in all_members]
        ys = [stitched_graph.nodes[m]['y'] for m in all_members]
        
        stitched_graph.nodes[root]['x'] = float(np.mean(xs))
        stitched_graph.nodes[root]['y'] = float(np.mean(ys))

        # Re-route edges from members to the root
        for member in members:
            neighbors = list(stitched_graph.neighbors(member))
            for nbr in neighbors:
                if nbr in all_members:
                    continue  # skip internal group loops
                
                edge_data = stitched_graph.get_edge_data(member, nbr)
                if stitched_graph.has_edge(root, nbr):
                    existing_data = stitched_graph.get_edge_data(root, nbr)
                    # Keep the edge with the smaller weight
                    if edge_data['weight'] < existing_data['weight']:
                        stitched_graph.add_edge(root, nbr, **edge_data)
                else:
                    stitched_graph.add_edge(root, nbr, **edge_data)
            
            stitched_graph.remove_node(member)

    return stitched_graph
