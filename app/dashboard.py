"""
Phase 6: Interactive Dashboard — Route Resilience Demo Surface
Maps to:
  - All 4 Core Objectives (visualisation & interaction layer)
  - Deliverables:
    * High-Fidelity Routable Topology (map overlay)
    * Quantitative Criticality Map (heatmap toggle)
    * Predictive Impact Assessment (click-to-disable simulation)

Launch:
    streamlit run app/dashboard.py

Expected pre-baked inputs (generate these BEFORE launching the dashboard):
    outputs/healed_graph.pkl    — pickled nx.Graph (healed, nodes with x/y in UTM EPSG:32643)
    outputs/unhealed_graph.pkl  — pickled nx.Graph (before healing, for before/after toggle)
    outputs/centrality.json     — JSON dict {node_id_str: score} (from centrality.py)
    outputs/metrics.json        — (optional) {"iou": ..., "dice": ..., "occlusion_recall": ...}
"""

import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import networkx as nx
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Allow importing sibling src/ modules when run from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resilience import compute_resilience_index  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG — paths to pre-baked data produced by the pipeline
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

HEALED_GRAPH_PATH = OUTPUTS_DIR / "healed_graph.pkl"
UNHEALED_GRAPH_PATH = OUTPUTS_DIR / "unhealed_graph.pkl"
CENTRALITY_PATH = OUTPUTS_DIR / "centrality.json"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# UTM CRS used throughout the pipeline (must match data_pipeline.DEFAULT_UTM_EPSG)
SOURCE_CRS_EPSG = 32643

# Default map centre (Bengaluru) — WGS84
DEFAULT_CENTER = [12.95, 77.60]
DEFAULT_ZOOM = 13


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_graph(path: Path) -> Optional[nx.Graph]:
    """Load a pickled NetworkX graph from disk."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_centrality(path: Path) -> Optional[Dict[str, float]]:
    """Load centrality scores from JSON."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_metrics(path: Path) -> Optional[Dict[str, float]]:
    """Load segmentation metrics from JSON."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def utm_to_latlon(x: float, y: float, epsg: int = SOURCE_CRS_EPSG) -> Tuple[float, float]:
    """Convert a single UTM point to WGS84 (lat, lon) using pyproj."""
    from pyproj import Transformer
    transformer = _get_transformer(epsg)
    lon, lat = transformer.transform(x, y)
    return (lat, lon)


@st.cache_resource
def _get_transformer(epsg: int):
    from pyproj import Transformer
    return Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)


def find_nearest_node(
    graph: nx.Graph, lat: float, lon: float, epsg: int = SOURCE_CRS_EPSG
) -> Any:
    """Find the graph node nearest to a clicked WGS84 lat/lon."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    click_x, click_y = transformer.transform(lon, lat)

    best_node = None
    best_dist = float("inf")
    for node, data in graph.nodes(data=True):
        dx = data["x"] - click_x
        dy = data["y"] - click_y
        d = math.hypot(dx, dy)
        if d < best_dist:
            best_dist = d
            best_node = node

    return best_node


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def build_road_map(
    graph: nx.Graph,
    centrality_scores: Optional[Dict[str, float]] = None,
    show_centrality: bool = True,
    disabled_node: Any = None,
) -> folium.Map:
    """Build a Folium map with road network edges and optional centrality overlay."""
    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles="OpenStreetMap")

    # --- Edge layer ---
    edge_group = folium.FeatureGroup(name="Road Network", show=True)
    for u, v, data in graph.edges(data=True):
        u_latlon = utm_to_latlon(graph.nodes[u]["x"], graph.nodes[u]["y"])
        v_latlon = utm_to_latlon(graph.nodes[v]["x"], graph.nodes[v]["y"])

        is_healed = data.get("healed", False)
        color = "#FF8C00" if is_healed else "#2563EB"  # orange vs blue
        dash = "8 4" if is_healed else None
        weight_px = 3 if is_healed else 2
        tooltip = f"{'Healed bridge' if is_healed else 'Road'} | {data.get('weight', 0):.0f} m"

        folium.PolyLine(
            [u_latlon, v_latlon],
            color=color,
            weight=weight_px,
            dash_array=dash,
            tooltip=tooltip,
        ).add_to(edge_group)
    edge_group.add_to(m)

    # --- Centrality heatmap layer ---
    if show_centrality and centrality_scores:
        cent_group = folium.FeatureGroup(name="Criticality Heatmap", show=True)
        scores = list(centrality_scores.values())
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for node_id_str, score in centrality_scores.items():
            # Try to find matching node in graph
            node_id = _resolve_node_id(graph, node_id_str)
            if node_id is None:
                continue

            data = graph.nodes[node_id]
            latlon = utm_to_latlon(data["x"], data["y"])
            norm = (score - min_score) / score_range

            # Radius 4-16px, colour green→yellow→red
            radius = 4 + norm * 12
            color = _score_to_color(norm)

            folium.CircleMarker(
                location=latlon,
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"Node {node_id_str} | Centrality: {score:.6f}",
            ).add_to(cent_group)
        cent_group.add_to(m)

    # --- Disabled node marker ---
    if disabled_node is not None and disabled_node in graph:
        d = graph.nodes[disabled_node]
        ll = utm_to_latlon(d["x"], d["y"])
        folium.Marker(
            location=ll,
            icon=folium.Icon(color="red", icon="remove", prefix="glyphicon"),
            tooltip=f"DISABLED: Node {disabled_node}",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _resolve_node_id(graph: nx.Graph, node_id_str: str) -> Any:
    """Resolve a JSON-serialized node ID back to its graph type (int or str)."""
    # Try int first (most skeleton graphs use integer node IDs)
    try:
        nid = int(node_id_str)
        if nid in graph:
            return nid
    except (ValueError, TypeError):
        pass
    if node_id_str in graph:
        return node_id_str
    return None


def _score_to_color(norm: float) -> str:
    """Map normalised score [0,1] to a green→yellow→red hex color."""
    if norm < 0.5:
        r = int(255 * (norm * 2))
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (norm - 0.5) * 2))
    return f"#{r:02x}{g:02x}00"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(
    healed_graph: Optional[nx.Graph],
    unhealed_graph: Optional[nx.Graph],
    centrality_scores: Optional[Dict[str, float]],
    metrics: Optional[Dict[str, float]],
) -> Tuple[bool, bool]:
    """Render sidebar panels. Returns (show_centrality, show_healed) toggles."""
    st.sidebar.title("🛰️ Route Resilience")
    st.sidebar.caption("ISRO PS-4 · Bharatiya Antariksh Hackathon 2026")
    st.sidebar.divider()

    # --- Toggle controls ---
    st.sidebar.subheader("Map Layers")
    show_centrality = st.sidebar.toggle("Show Criticality Heatmap", value=True)
    show_healed = st.sidebar.toggle("Show Healed Graph", value=True,
                                     help="OFF = show raw fragmented graph before MST healing")
    st.sidebar.divider()

    # --- Connectivity Ratio ---
    st.sidebar.subheader("📊 Graph Metrics")
    if healed_graph and unhealed_graph:
        def _lcc(g):
            if len(g) == 0:
                return 0
            return max(len(c) for c in nx.connected_components(g))

        lcc_before = _lcc(unhealed_graph)
        lcc_after = _lcc(healed_graph)
        ratio = lcc_after / lcc_before if lcc_before > 0 else 1.0

        col1, col2 = st.sidebar.columns(2)
        col1.metric("LCC Before", f"{lcc_before:,}")
        col2.metric("LCC After", f"{lcc_after:,}")
        st.sidebar.metric("Connectivity Ratio", f"{ratio:.2f}x",
                          delta=f"+{((ratio - 1) * 100):.0f}%" if ratio > 1 else None)

        n_healed = sum(1 for _, _, d in healed_graph.edges(data=True) if d.get("healed"))
        st.sidebar.metric("Bridge Edges Added", n_healed)

        comps_before = nx.number_connected_components(unhealed_graph)
        comps_after = nx.number_connected_components(healed_graph)
        st.sidebar.metric("Components", f"{comps_before} → {comps_after}")
    else:
        st.sidebar.info("Load both healed & unhealed graphs to see connectivity metrics.")

    st.sidebar.divider()

    # --- Top Gatekeeper Nodes ---
    if centrality_scores:
        st.sidebar.subheader("🏗️ Top Gatekeeper Nodes")
        sorted_nodes = sorted(centrality_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for rank, (nid, score) in enumerate(sorted_nodes, 1):
            st.sidebar.text(f"#{rank}  Node {nid}  →  {score:.6f}")
    st.sidebar.divider()

    # --- Segmentation Metrics ---
    st.sidebar.subheader("🎯 Segmentation Metrics")
    if metrics:
        mcols = st.sidebar.columns(2)
        mcols[0].metric("IoU", f"{metrics.get('iou', 0):.4f}")
        mcols[1].metric("Dice", f"{metrics.get('dice', 0):.4f}")
        if "occlusion_recall" in metrics:
            st.sidebar.metric("Occlusion Recall", f"{metrics['occlusion_recall']:.4f}")
    else:
        st.sidebar.caption("_Run training to populate (outputs/metrics.json)_")

    return show_centrality, show_healed


# ---------------------------------------------------------------------------
# Resilience simulation panel
# ---------------------------------------------------------------------------

def render_resilience_panel(
    graph: nx.Graph,
    clicked_node: Any,
) -> None:
    """Display resilience simulation results for a clicked node."""
    st.subheader(f"⚠️ Resilience Simulation — Node {clicked_node}")

    with st.spinner("Running node-ablation simulation..."):
        result = compute_resilience_index(
            graph,
            nodes_to_disable=[clicked_node],
            sample_pairs=200,
            seed=42,
        )

    # Interpret results
    baseline = result["baseline_avg_path_length"]
    post = result["post_removal_avg_path_length"]
    ri = min(result["resilience_index"], 1.0)  # clamp for display
    pct_disc = result["pct_pairs_disconnected"]

    if baseline > 0:
        pct_increase = ((post - baseline) / baseline) * 100.0
    else:
        pct_increase = 0.0

    # Metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Resilience Index", f"{ri:.3f}",
              help="1.0 = no impact, lower = more critical")
    c2.metric("Avg Path ↑", f"+{pct_increase:.1f}%",
              help="Increase in average travel distance after disabling this node")
    c3.metric("Routes Broken", f"{pct_disc:.1f}%",
              help="Percentage of sampled OD pairs that became unreachable")

    # Plain-language summary
    if pct_increase > 50 or pct_disc > 20:
        st.error(
            f"🚨 **High-impact node.** Disabling this intersection increases average "
            f"travel distance by **{pct_increase:.0f}%** and disconnects **{pct_disc:.0f}%** "
            f"of sampled routes. This is a critical single point of failure."
        )
    elif pct_increase > 10 or pct_disc > 5:
        st.warning(
            f"⚠️ Moderate impact. Average travel distance increases by "
            f"**{pct_increase:.0f}%** with **{pct_disc:.0f}%** routes broken."
        )
    else:
        st.success(
            f"✅ Low impact. The network is resilient to this node's removal "
            f"(+{pct_increase:.1f}% distance, {pct_disc:.1f}% routes broken)."
        )

    # Raw debug details
    with st.expander("Raw simulation data"):
        st.json(result)


# ---------------------------------------------------------------------------
# Demo-mode fallback (synthetic graph for testing without real data)
# ---------------------------------------------------------------------------

def _make_demo_graph() -> Tuple[nx.Graph, nx.Graph]:
    """
    Generate a small synthetic road graph for testing the dashboard
    without running the full pipeline. Returns (unhealed, healed).
    """
    rng = np.random.default_rng(0)
    # Bengaluru approximate UTM 43N center: easting ~830000, northing ~1432000
    cx, cy = 830_000.0, 1_432_000.0

    G = nx.Graph()
    # Grid-ish layout with some noise
    node_id = 0
    grid = {}
    for i in range(8):
        for j in range(8):
            x = cx + i * 200 + rng.normal(0, 20)
            y = cy + j * 200 + rng.normal(0, 20)
            G.add_node(node_id, x=x, y=y)
            grid[(i, j)] = node_id
            node_id += 1

    # Connect grid neighbours
    for i in range(8):
        for j in range(8):
            if i < 7:
                u, v = grid[(i, j)], grid[(i + 1, j)]
                d = math.hypot(G.nodes[u]["x"] - G.nodes[v]["x"],
                               G.nodes[u]["y"] - G.nodes[v]["y"])
                G.add_edge(u, v, weight=d, healed=False)
            if j < 7:
                u, v = grid[(i, j)], grid[(i, j + 1)]
                d = math.hypot(G.nodes[u]["x"] - G.nodes[v]["x"],
                               G.nodes[u]["y"] - G.nodes[v]["y"])
                G.add_edge(u, v, weight=d, healed=False)

    healed = G.copy()

    # Create unhealed version by removing some edges to simulate fragmentation
    unhealed = G.copy()
    edges_to_remove = list(unhealed.edges())
    rng.shuffle(edges_to_remove)
    for e in edges_to_remove[:12]:
        unhealed.remove_edge(*e)

    # Mark the "healed" bridges
    for e in edges_to_remove[:12]:
        if not healed.has_edge(*e):
            continue
        healed.edges[e]["healed"] = True

    return unhealed, healed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Route Resilience Dashboard",
        page_icon="🛰️",
        layout="wide",
    )

    # --- Load data ---
    healed_graph = load_graph(HEALED_GRAPH_PATH)
    unhealed_graph = load_graph(UNHEALED_GRAPH_PATH)
    centrality_scores = load_centrality(CENTRALITY_PATH)
    metrics = load_metrics(METRICS_PATH)

    # Fall back to demo data if real outputs aren't available yet
    demo_mode = False
    if healed_graph is None:
        demo_mode = True
        st.toast("📌 No pre-baked data found — loading demo graph.", icon="ℹ️")
        unhealed_graph, healed_graph = _make_demo_graph()

        # Compute centrality on the fly for the tiny demo graph
        centrality_scores = {
            str(n): float(s)
            for n, s in nx.betweenness_centrality(
                healed_graph, weight="weight", normalized=True
            ).items()
        }

    if demo_mode:
        st.info(
            "**Demo mode** — using a synthetic graph. To use real data, place "
            "pre-baked outputs in `outputs/` (see dashboard.py header docstring).",
            icon="🔬",
        )

    # --- Sidebar ---
    show_centrality, show_healed = render_sidebar(
        healed_graph, unhealed_graph, centrality_scores, metrics
    )

    active_graph = healed_graph if show_healed else unhealed_graph
    if active_graph is None:
        active_graph = healed_graph  # fallback

    # --- Session state for clicked node ---
    if "disabled_node" not in st.session_state:
        st.session_state.disabled_node = None

    # --- Map ---
    st.markdown("### 🗺️ Road Network & Criticality Map")
    st.caption("Click on the map to select a node for resilience simulation.")

    fmap = build_road_map(
        active_graph,
        centrality_scores=centrality_scores,
        show_centrality=show_centrality,
        disabled_node=st.session_state.disabled_node,
    )

    map_data = st_folium(fmap, width=None, height=550, returned_objects=["last_clicked"])

    # --- Handle click ---
    if map_data and map_data.get("last_clicked"):
        click = map_data["last_clicked"]
        lat, lon = click["lat"], click["lng"]

        nearest = find_nearest_node(active_graph, lat, lon)
        if nearest is not None:
            st.session_state.disabled_node = nearest

    if st.session_state.disabled_node is not None:
        render_resilience_panel(healed_graph, st.session_state.disabled_node)

    # --- Legend ---
    with st.expander("Map Legend"):
        st.markdown("""
        | Element | Meaning |
        |---|---|
        | 🔵 **Blue solid line** | Original road edge |
        | 🟠 **Orange dashed line** | Healed bridge edge (added by MST healing) |
        | 🟢→🟡→🔴 **Circle markers** | Node criticality (green=low, red=high betweenness) |
        | 📍 **Red marker** | Currently disabled node (click map to select) |
        """)


if __name__ == "__main__":
    main()
