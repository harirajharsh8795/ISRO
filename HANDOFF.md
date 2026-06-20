# HANDOFF TO NEXT MODEL

This document details the blueprint, implemented functions, and next steps for the **Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis** project (Bharatiya Antariksh Hackathon 2026, PS-4).

## 1. What's Done
- **`requirements.txt`**: Complete python package specification for computer vision, geospatial, network analysis, and dashboard visualization.
- **`src/data_pipeline.py`**: Complete function signatures, typed interfaces, docstrings, and detailed step-by-step implementation blueprints for:
  - `fetch_osm_roads` (OSM vector retrieval)
  - `rasterize_osm_to_mask` (road buffering and rasterization)
  - `tile_image_and_mask` (tiling and negative mining)
  - `add_synthetic_occlusion` (augmentations for occlusion-awareness)
- **`src/graph_healing.py`**: A fully functional, production-ready implementation of **Topological Healing (Function A)** using a Minimum Spanning Tree (MST) / Union-Find (Disjoint Set) approach combined with dual-endpoint angular alignment validation.
- **`src/resilience.py`**: A fully functional, production-ready implementation of **Resilience Index (Function B)** using reproducible random path-cost sampling, node ablation simulation, and path recalculation with disconnection penalty scaling.

## 2. What's NOT Done (Priority Order)
1. **Data Pipeline Implementation (`src/data_pipeline.py`)**: Complete the body of the four skeleton functions using libraries like `osmnx`, `rasterio.features.rasterize`, and `albumentations`.
2. **Phase 2 Segmentation Model Training**:
   - Write `src/segmentation_train.py` using `segmentation_models_pytorch` (e.g. U-Net with ResNet34 backbone as baseline).
   - Set up the training loop, loss function (BCE + Dice), and validation.
   - Run fine-tuning on the target AOI using Cartosat-3 or Sentinel-2 imagery.
3. **Phase 4 Centrality Mapping**:
   - Implement a module `src/centrality.py` to calculate `betweenness_centrality` on the healed graph using NetworkX.
   - Rank intersections and output spatial heatmaps.
4. **Phase 6 Dashboard Integration**:
   - Write a Streamlit dashboard (`app/dashboard.py`) integrating:
     - Map view showing road segmentation masks, healed vector networks, and centrality heatmaps.
     - Interactive node disabling connected to `compute_resilience_index`.
     - Slider showing the "before" (fragmented) vs "after" (healed) graph state.

## 3. Core Assumptions & Engineering Guidelines
> [!IMPORTANT]
> To prevent topology and geometry bugs, the next model **MUST** respect the following guidelines:
> - **Coordinate Systems (CRS)**: All calculations (bounding boxes, distances, node positions) must use a **projected coordinate system in meters** (e.g., UTM EPSG:32643 for Bengaluru) instead of geographic coordinates (degrees, EPSG:4326). Measuring distance in degrees is a critical failure point.
> - **Graph Nodes**: Nodes in the NetworkX graph must store coordinate values as float properties `x` and `y` matching the projected coordinate system.
> - **Edge Weights**: Edges must store geographic lengths (in meters) under the `'weight'` key to ensure Dijkstra computations represent actual distance/travel cost.
> - **Graph Types**: The functions are fully compatible with undirected `nx.Graph` and `nx.MultiGraph` objects.

## 4. Next Prompt to Copy-Paste
```text
Continue implementing the Route Resilience hackathon project (PS-4) based on the existing codebase:
1. Review the requirements.txt, src/data_pipeline.py, src/graph_healing.py, and src/resilience.py files.
2. Complete the full implementation of the data pipeline skeleton in `src/data_pipeline.py`.
3. Implement the training and inference pipeline for the road extraction model in `src/segmentation_train.py` using a U-Net with a ResNet backbone from segmentation-models-pytorch.
4. Write a script `src/centrality.py` to calculate betweenness centrality on the healed graph and generate a GeoJSON map of "Gatekeeper Nodes".
5. Integrate all modules into a fully interactive Streamlit dashboard (`app/dashboard.py`) using folium or leaflet for interactive node-disabling and real-time rerouting simulations.
```
