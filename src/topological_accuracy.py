import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, List, Any
import osmnx as ox

def compute_apl_error(
    predicted_graph: nx.Graph,
    ground_truth_graph: nx.Graph,
    num_point_pairs: int = 100,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes the Average Path Length (APL) error between predicted and ground-truth graphs.
    Snaps random geographic query points to both graphs to compute shortest path lengths.
    
    Args:
        predicted_graph: The predicted/extracted NetworkX graph.
        ground_truth_graph: The ground-truth NetworkX graph.
        num_point_pairs: Number of random query pairs to sample.
        seed: Random seed for reproducibility.
        
    Returns:
        mean_absolute_error: Mean absolute difference in shortest path lengths (meters).
        mean_percentage_error: Mean percentage difference in shortest path lengths.
        reach_delta: Absolute difference in reachability percentage between graphs.
    """
    if len(predicted_graph) < 2 or len(ground_truth_graph) < 2:
        return 0.0, 1.0, 100.0

    # Create spatial lookup trees for snapping
    pred_nodes = list(predicted_graph.nodes(data=True))
    gt_nodes = list(ground_truth_graph.nodes(data=True))

    pred_coords = np.array([[n[1]['x'], n[1]['y']] for n in pred_nodes])
    gt_coords = np.array([[n[1]['x'], n[1]['y']] for n in gt_nodes])

    pred_tree = KDTree(pred_coords)
    gt_tree = KDTree(gt_coords)

    rng = np.random.default_rng(seed)
    
    # Bounding box of ground-truth coordinates
    min_x, min_y = gt_coords.min(axis=0)
    max_x, max_y = gt_coords.max(axis=0)

    maes = []
    mpes = []
    unreachable_pred = 0
    unreachable_gt = 0
    valid_pairs = 0

    # Draw candidates
    for _ in range(num_point_pairs * 5):
        if valid_pairs >= num_point_pairs:
            break
            
        qx1, qy1 = rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)
        qx2, qy2 = rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)

        # Snap to closest nodes in ground-truth graph
        _, gt_idx1 = gt_tree.query([qx1, qy1])
        _, gt_idx2 = gt_tree.query([qx2, qy2])
        gt_u, gt_v = gt_nodes[gt_idx1][0], gt_nodes[gt_idx2][0]

        # Snap to closest nodes in predicted graph
        _, pred_idx1 = pred_tree.query([qx1, qy1])
        _, pred_idx2 = pred_tree.query([qx2, qy2])
        pred_u, pred_v = pred_nodes[pred_idx1][0], pred_nodes[pred_idx2][0]

        if gt_u == gt_v or pred_u == pred_v:
            continue

        # Shortest path in GT
        try:
            gt_dist = nx.shortest_path_length(ground_truth_graph, source=gt_u, target=gt_v, weight='weight')
            gt_reachable = True
        except nx.NetworkXNoPath:
            gt_reachable = False
            unreachable_gt += 1

        # Shortest path in predicted
        try:
            pred_dist = nx.shortest_path_length(predicted_graph, source=pred_u, target=pred_v, weight='weight')
            pred_reachable = True
        except nx.NetworkXNoPath:
            pred_reachable = False
            unreachable_pred += 1

        if gt_reachable and pred_reachable:
            mae = abs(pred_dist - gt_dist)
            mpe = mae / (gt_dist + 1e-8)
            maes.append(mae)
            mpes.append(mpe)
            valid_pairs += 1

    if not maes:
        return 0.0, 1.0, 100.0

    mean_mae = float(np.mean(maes))
    mean_mpe = float(np.mean(mpes))
    
    total_attempts = valid_pairs + unreachable_gt
    reach_gt = 1.0 - (unreachable_gt / (total_attempts + 1e-8))
    reach_pred = 1.0 - (unreachable_pred / (total_attempts + 1e-8))
    reach_delta = float(abs(reach_gt - reach_pred) * 100.0)

    return mean_mae, mean_mpe, reach_delta

def build_osm_ground_truth_graph(
    aoi_bbox: Tuple[float, float, float, float],
    target_epsg: int = 32643
) -> nx.Graph:
    """
    Fetches OSM vector data for the bbox using osmnx and constructs a projected
    NetworkX graph with Euclidean edge weights, representing the true network.
    """
    from data_pipeline import fetch_osm_roads
    gdf = fetch_osm_roads(aoi_bbox, target_epsg=target_epsg)
    
    G = nx.Graph()
    node_counter = 0
    coord_to_node = {}

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.type == 'LineString':
            coords = list(geom.coords)
        elif geom.type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            continue

        if len(coords) < 2:
            continue

        prev_node = None
        for x, y in coords:
            # Round coordinates to snap node endpoints together
            pt = (round(x, 2), round(y, 2))
            if pt not in coord_to_node:
                coord_to_node[pt] = node_counter
                G.add_node(node_counter, x=x, y=y)
                node_counter += 1
            curr_node = coord_to_node[pt]

            if prev_node is not None and prev_node != curr_node:
                px, py = G.nodes[prev_node]['x'], G.nodes[prev_node]['y']
                cx, cy = G.nodes[curr_node]['x'], G.nodes[curr_node]['y']
                dist = float(np.hypot(px - cx, py - cy))
                G.add_edge(prev_node, curr_node, weight=dist)

            prev_node = curr_node
            
    return G
