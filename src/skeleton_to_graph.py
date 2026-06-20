"""
Connective tissue between Phase 2 (Occlusion-Aware Extraction) and Phase 3
(Topological Reconstruction). Converts predicted binary road masks into a
geo-referenced NetworkX graph that heal_graph_connectivity() accepts directly.

Maps to:
  - 4 Core Objectives:
    * Bridges Occlusion-Aware Extraction → Topological Reconstruction
  - Evaluation Parameters:
    * Connectivity Ratio (input quality directly affects healing performance)
    * Topological Accuracy (graph fidelity depends on correct skeletonization)
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import networkx as nx
from pyproj import Transformer
from skimage.morphology import skeletonize

from geo_utils import assert_projected_crs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 8-connectivity offsets (row_delta, col_delta)
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]


def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Thin a binary road mask to a 1-pixel-wide skeleton."""
    # skimage.morphology.skeletonize expects a bool array
    return skeletonize(binary_mask.astype(bool)).astype(np.uint8)


def _count_neighbors(skel: np.ndarray, r: int, c: int) -> int:
    """Count 8-connected foreground neighbors of pixel (r, c)."""
    h, w = skel.shape
    count = 0
    for dr, dc in _NEIGHBORS_8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
            count += 1
    return count


def _get_neighbors(skel: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
    """Return list of 8-connected foreground neighbor coords."""
    h, w = skel.shape
    out = []
    for dr, dc in _NEIGHBORS_8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
            out.append((nr, nc))
    return out


def _build_pixel_graph(skel: np.ndarray) -> Tuple[nx.Graph, Dict[Tuple[int,int], int]]:
    """
    Walk the skeleton to build a simplified graph where nodes are junctions
    (degree > 2) and endpoints (degree == 1), and edges represent the
    skeleton paths connecting them.

    Approach: manual 8-connectivity scan. We chose this over the `sknw`
    library to avoid adding an extra dependency not in requirements.txt.
    The logic is straightforward for hackathon-scale masks and avoids sknw's
    known issues with self-loops at complex junctions.

    Returns:
        graph: nx.Graph with node attr 'rc' = (row, col) pixel coords,
               edge attr 'path_length_px' = number of pixels along the path.
        coord_to_node: mapping from (row, col) to node ID.
    """
    h, w = skel.shape

    # 1. Classify every foreground pixel
    key_pixels: Set[Tuple[int, int]] = set()       # junctions + endpoints
    skeleton_pixels: Set[Tuple[int, int]] = set()   # all foreground

    for r in range(h):
        for c in range(w):
            if not skel[r, c]:
                continue
            skeleton_pixels.add((r, c))
            n = _count_neighbors(skel, r, c)
            if n != 2:
                # endpoint (1), junction (3+), or isolated (0)
                key_pixels.add((r, c))

    if not key_pixels:
        # Skeleton has no junctions/endpoints (empty or pure loops with no
        # branches). Return empty graph.
        G = nx.Graph()
        return G, {}

    # 2. Assign node IDs to key pixels
    coord_to_node: Dict[Tuple[int, int], int] = {}
    G = nx.Graph()
    for idx, (r, c) in enumerate(sorted(key_pixels)):
        coord_to_node[(r, c)] = idx
        G.add_node(idx, rc=(r, c))

    # 3. Trace paths between key pixels along skeleton
    visited_edges: Set[Tuple[int, int]] = set()  # frozenset-like (min,max) node pairs

    for start_rc in key_pixels:
        start_id = coord_to_node[start_rc]
        for nbr_rc in _get_neighbors(skel, *start_rc):
            # Walk from start_rc through nbr_rc until we hit another key pixel
            prev = start_rc
            curr = nbr_rc
            path_len = 1

            while curr not in key_pixels:
                neighbors = _get_neighbors(skel, *curr)
                # Move to the neighbor that isn't the one we came from
                next_pixels = [n for n in neighbors if n != prev]
                if not next_pixels:
                    break  # dead end that wasn't classified (shouldn't happen)
                prev = curr
                curr = next_pixels[0]
                path_len += 1

            if curr not in key_pixels:
                continue

            end_id = coord_to_node[curr]

            # Skip self-loops and already-visited edges
            if start_id == end_id:
                continue
            edge_key = (min(start_id, end_id), max(start_id, end_id))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            # Keep shortest path if multiple paths connect the same pair
            if G.has_edge(start_id, end_id):
                if G.edges[start_id, end_id]["path_length_px"] <= path_len:
                    continue

            G.add_edge(start_id, end_id, path_length_px=path_len)

    logger.info(f"Pixel graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, coord_to_node


def _prune_spurs(graph: nx.Graph, min_branch_length_px: int) -> nx.Graph:
    """
    Remove dangling degree-1 branches shorter than min_branch_length_px.
    These are typically skeletonization artifacts (whiskers), not real roads.
    Iterates until no more short spurs remain.
    """
    changed = True
    while changed:
        changed = False
        to_remove = []
        for node in list(graph.nodes):
            if graph.degree(node) == 1:
                edge = list(graph.edges(node, data=True))[0]
                path_len = edge[2].get("path_length_px", 0)
                if path_len < min_branch_length_px:
                    to_remove.append(node)
        for node in to_remove:
            graph.remove_node(node)
            changed = True

    return graph


def mask_to_graph(
    binary_mask: np.ndarray,
    raster_transform: "rasterio.Affine",
    source_crs_epsg: int,
    target_utm_epsg: int = 32643,
    min_branch_length_px: int = 5,
) -> nx.Graph:
    """
    Convert a binary road mask to a geo-referenced NetworkX graph compatible
    with heal_graph_connectivity() and compute_resilience_index().

    Args:
        binary_mask: (H, W) uint8/bool predicted road mask.
        raster_transform: rasterio Affine transform from the source image.
        source_crs_epsg: EPSG code of the source raster CRS.
        target_utm_epsg: EPSG code for the projected output CRS (default: UTM 43N).
        min_branch_length_px: Prune skeleton spurs shorter than this (pixels).

    Returns:
        nx.Graph with nodes having 'x','y' in target_utm_epsg meters and
        edges having 'weight' in meters.
    """
    # 1. Skeletonize
    skel = skeletonize_mask(binary_mask)
    fg_count = int(skel.sum())
    logger.info(f"Skeleton: {fg_count} foreground pixels from mask of shape {binary_mask.shape}")

    if fg_count == 0:
        logger.warning("Empty skeleton — returning empty graph.")
        return nx.Graph()

    # 2. Build pixel-level graph (junctions + endpoints as nodes)
    pixel_graph, coord_to_node = _build_pixel_graph(skel)

    if pixel_graph.number_of_nodes() == 0:
        return nx.Graph()

    # 3. Prune short spur branches (skeletonization artifacts)
    pixel_graph = _prune_spurs(pixel_graph, min_branch_length_px)
    logger.info(f"After spur pruning: {pixel_graph.number_of_nodes()} nodes, "
                f"{pixel_graph.number_of_edges()} edges")

    # 4. Convert pixel (row, col) → real-world coordinates via raster_transform
    #    rasterio Affine convention: x, y = transform * (col, row)  ← note col first!
    need_reproject = (source_crs_epsg != target_utm_epsg)
    if need_reproject:
        transformer = Transformer.from_crs(
            f"EPSG:{source_crs_epsg}", f"EPSG:{target_utm_epsg}", always_xy=True
        )

    for node in pixel_graph.nodes:
        r, c = pixel_graph.nodes[node]["rc"]
        # rasterio: (col, row) → (x, y) in source CRS
        src_x, src_y = raster_transform * (c, r)

        if need_reproject:
            proj_x, proj_y = transformer.transform(src_x, src_y)
        else:
            proj_x, proj_y = src_x, src_y

        pixel_graph.nodes[node]["x"] = float(proj_x)
        pixel_graph.nodes[node]["y"] = float(proj_y)

    # 5. Compute edge weights as Euclidean distance in projected meters
    #    (straight-line between endpoints, not full skeleton path length —
    #    a simplification that's acceptable for hackathon scope; a future
    #    refinement would sum segment lengths along the traced pixel path)
    for u, v in pixel_graph.edges:
        ux, uy = pixel_graph.nodes[u]["x"], pixel_graph.nodes[u]["y"]
        vx, vy = pixel_graph.nodes[v]["x"], pixel_graph.nodes[v]["y"]
        pixel_graph.edges[u, v]["weight"] = float(np.hypot(ux - vx, uy - vy))

    # 6. Clean up intermediate attributes not needed downstream
    for node in pixel_graph.nodes:
        if "rc" in pixel_graph.nodes[node]:
            del pixel_graph.nodes[node]["rc"]

    # 7. Self-check: output must pass the CRS guard
    assert_projected_crs(pixel_graph)

    logger.info(f"Final graph: {pixel_graph.number_of_nodes()} nodes, "
                f"{pixel_graph.number_of_edges()} edges, CRS check passed.")
    return pixel_graph
