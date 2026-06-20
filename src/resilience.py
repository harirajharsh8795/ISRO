"""
Phase 5: Simulated Stress Testing & Resilience Analysis
Maps to:
  - 4 Core Objectives:
    * Simulated Stress Testing (Node ablation simulation framework)
  - Evaluation Parameters:
    * Topological Accuracy (Average Path Length comparison)
    * Predictive Impact Assessment (Resilience Index quantification)
"""

import logging
from typing import List, Dict, Any
import numpy as np
import networkx as nx
from geo_utils import assert_projected_crs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def compute_resilience_index(
    graph: nx.Graph,
    nodes_to_disable: List[Any],
    sample_pairs: int = 200,
    seed: int = 42,
    penalty_multiplier: float = 5.0
) -> Dict[str, Any]:
    """
    Simulates a localized network failure by disabling (removing) specified nodes
    and measuring the impact on average shortest path lengths over a sampled set of node pairs.

    Resilience Index Definition:
        Resilience Index = baseline_avg_path_length / post_removal_avg_path_length
        - Value of 1.0 indicates perfect resilience (no change in connectivity/travel cost).
        - Value approaches 0.0 as paths become significantly longer or disconnected.
        - Disconnected pairs are penalized by scaling their baseline length by `penalty_multiplier`.

    Args:
        graph: The healed, routable NetworkX graph (undirected, with edge weight representing travel cost/meters).
        nodes_to_disable: List of node IDs to temporarily disable/remove from the graph.
        sample_pairs: Number of random reachable node pairs to sample for travel time comparison.
        seed: Random seed for reproducible pair sampling.
        penalty_multiplier: Factor to multiply baseline path length by when a pair becomes disconnected
                             due to node removal.

    Returns:
        A dictionary containing:
            - 'baseline_avg_path_length': float, average cost before ablation
            - 'post_removal_avg_path_length': float, average cost after ablation (including penalties)
            - 'resilience_index': float, calculated resilience index in range [0, 1]
            - 'pct_pairs_disconnected': float, percentage of sampled pairs that became unreachable
            - 'affected_node_count': int, number of nodes disabled
    """
    # Validate that coordinates are in projected meters, not geographic degrees
    assert_projected_crs(graph)

    # 1. Edge case handling: Graph too small or empty input
    if len(graph) < 2:
        logger.warning("Graph has fewer than 2 nodes. Cannot compute resilience index.")
        return {
            'baseline_avg_path_length': 0.0,
            'post_removal_avg_path_length': 0.0,
            'resilience_index': 1.0,
            'pct_pairs_disconnected': 0.0,
            'affected_node_count': len(nodes_to_disable)
        }

    all_nodes = list(graph.nodes())
    rng = np.random.default_rng(seed)
    
    # 2. Sample random reachable pairs from the baseline graph
    sampled_pairs = []
    baseline_lengths = []
    
    # Limit maximum attempts to avoid infinite loop on highly disconnected graphs
    max_attempts = sample_pairs * 50
    attempts = 0
    
    while len(sampled_pairs) < sample_pairs and attempts < max_attempts:
        attempts += 1
        # Randomly choose two distinct nodes
        u, v = rng.choice(all_nodes, size=2, replace=False)
        
        try:
            # Check path and compute its baseline cost
            base_len = nx.shortest_path_length(graph, source=u, target=v, weight='weight')
            sampled_pairs.append((u, v))
            baseline_lengths.append(base_len)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not sampled_pairs:
        logger.error("Could not find any reachable node pairs to sample.")
        return {
            'baseline_avg_path_length': 0.0,
            'post_removal_avg_path_length': 0.0,
            'resilience_index': 1.0,
            'pct_pairs_disconnected': 0.0,
            'affected_node_count': len(nodes_to_disable)
        }

    logger.info(f"Successfully sampled {len(sampled_pairs)} reachable node pairs for evaluation.")

    # 3. Create a copy of the graph and remove target nodes
    post_graph = graph.copy()
    post_graph.remove_nodes_from(nodes_to_disable)

    # 4. Recompute shortest path lengths over the exact same pairs
    post_lengths = []
    unreachable_count = 0

    for (u, v), base_len in zip(sampled_pairs, baseline_lengths):
        # Case A: One of the sampled endpoints was disabled directly
        if u not in post_graph or v not in post_graph:
            unreachable_count += 1
            post_lengths.append(base_len * penalty_multiplier)
            continue
            
        # Case B: Compute path length through the remaining network
        try:
            post_len = nx.shortest_path_length(post_graph, source=u, target=v, weight='weight')
            post_lengths.append(post_len)
        except nx.NetworkXNoPath:
            # Pair became disconnected due to ablation
            unreachable_count += 1
            post_lengths.append(base_len * penalty_multiplier)

    # 5. Compute summary statistics
    baseline_avg = float(np.mean(baseline_lengths))
    post_avg = float(np.mean(post_lengths))
    
    # Calculate index (higher post_avg implies lower resilience)
    resilience_index = 1.0 if post_avg == 0.0 else float(baseline_avg / post_avg)
    pct_disconnected = float((unreachable_count / len(sampled_pairs)) * 100.0)

    logger.info(f"Resilience index: {resilience_index:.4f} | "
                f"Disconnected pairs: {pct_disconnected:.1f}%")

    return {
        'baseline_avg_path_length': baseline_avg,
        'post_removal_avg_path_length': post_avg,
        'resilience_index': resilience_index,
        'pct_pairs_disconnected': pct_disconnected,
        'affected_node_count': len(nodes_to_disable)
    }
