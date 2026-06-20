"""
Geospatial utility guards shared across graph_healing and resilience modules.
"""

from typing import Any
import networkx as nx


def assert_projected_crs(graph: nx.Graph) -> None:
    """
    Validates that graph node coordinates are in a projected CRS (meters),
    not geographic degrees. Raises ValueError if coordinates look like lat/lon.
    """
    if len(graph) < 2:
        return

    # Sample up to 10 nodes — enough to detect CRS class, cheap on large graphs
    sample_nodes = list(graph.nodes)[:10]

    xs = []
    ys = []
    for n in sample_nodes:
        data = graph.nodes[n]
        if 'x' not in data or 'y' not in data:
            raise ValueError(
                f"Node {n} is missing 'x' and/or 'y' attributes. "
                "Build the graph with node attributes x, y in a projected CRS (meters)."
            )
        xs.append(float(data['x']))
        ys.append(float(data['y']))

    # Heuristic: geographic coordinates (EPSG:4326) have x in [-180, 180] and y in [-90, 90].
    # UTM eastings are typically 100_000–900_000 and northings 0–10_000_000.
    # If ALL sampled values fall inside the degree-plausible box, flag it.
    all_x_look_like_degrees = all(-180.0 <= x <= 180.0 for x in xs)
    all_y_look_like_degrees = all(-90.0 <= y <= 90.0 for y in ys)

    if all_x_look_like_degrees and all_y_look_like_degrees:
        raise ValueError(
            f"Node coordinates appear to be in geographic degrees "
            f"(sampled x range [{min(xs):.4f}, {max(xs):.4f}], "
            f"y range [{min(ys):.4f}, {max(ys):.4f}]). "
            f"Graph healing and resilience analysis require a PROJECTED CRS in meters "
            f"(e.g. UTM EPSG:32643 for Bengaluru). "
            f"Reproject your GeoDataFrame with gdf.to_crs(epsg=32643) BEFORE building the graph."
        )
