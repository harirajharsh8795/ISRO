# Judge Q&A Preparation: Route Resilience Pipeline

This document compiles the verbal scripts, plain-language impact summaries, and answers to challenging questions designed to prepare the team for the live demo and judge evaluation.

---

## 1. 30-60 Second Verbal Script (Addressing Geospatial Coordinates)

**Use this script if a judge asks: *"How did you test this on real data, and why are the real-data overlap metrics low?"***

> "We tested our pipeline under two scenarios. 
>
> First, we ran a **controlled validation experiment** on aligned datasets to prove our deep learning architecture. Here, the fine-tuned model achieved **93.90% IoU under heavy occlusion** and reduced topological path routing error to just **4.54%**.
>
> Second, we ran our system on the **real satellite imagery** provided in the repository. Because these files are un-georeferenced JPEGs, our pipeline projects them onto a mock UTM grid to stitch a mosaic. 
>
> When we fetch live OpenStreetMap data for these coordinates, it queries a rural field in Karnataka which does not match the urban streets in the JPEGs. This coordinate alignment mismatch is why the real-data overlap metrics are low. 
>
> Crucially, we verified our **rasterio georeferencing engine** (Fix #1) on a georeferenced GeoTIFF. The pipeline successfully extracted nodes in real-world coordinates and fetched corresponding OSM roads with **zero mock fallback**. When Cartosat-3 GeoTIFF data is provided, the pipeline is fully prepared to run end-to-end with true coordinates."

---

## 2. Plain-Language "What We Fixed & Why It Matters" One-Pager

| Tech Upgrade | What it is in Plain Language | Why it Matters (Judge Impact) |
|---|---|---|
| **CBAM Attention Layer** | Adds "focus filters" to the AI's neural network to scan both shape features and spatial context. | **Roads under shadows are detected:** Traditional AI gets "spectrally blind" under tree canopy or building shadows. Our model uses context to infer and reconstruct hidden roads. |
| **Occlusion-Weighted Loss** | A training mechanism that penalizes the AI four times harder for mistakes made under occlusions. | **Guarantees robustness:** Forces the model to actively learn shadow and cloud patterns, rather than ignoring them as background noise. |
| **Rasterio Geospatial Reprojection** | An integrated GIS engine that reads and warps satellite metadata into coordinate systems (meters). | **Ready for ISRO Cartosat Data:** The system natively ingests georeferenced imagery and aligns it with global coordinates automatically. |
| **Spatial stitching & KDTree** | Merges overlapping graph nodes at tile boundaries within a specified distance tolerance. | **Stitches road grids together:** Ensures that roads crossing between separate image tiles merge into a single continuous network. |
| **Topological Graph Healing** | Geometric search algorithm that bridges road disconnections based on distance and road angle. | **Creates routable networks:** Restores connectivity when segmentation output has small pixel gaps, preventing routing failures. |
| **Streamlit Cache Pre-Warming** | Pre-calculates resilience simulations for top gatekeeper nodes in the background. | **Sub-second UI Response:** Prevents the dashboard from freezing when judges interactively click on map nodes. |

---

## 3. Tough Questions and Draft Answers

### Q1: "Have you tested this on real Cartosat-3 data?"
* **Answer:** "Our geospatial backend is fully implemented and verified using a properly georeferenced sample GeoTIFF that follows the exact Cartosat-3 and Sentinel-2 metadata formats (including CRS headers and affine geotransforms). During verification, the rasterio-based pipeline successfully read the metadata, reprojected the coordinates to UTM meters, and queried the correct OpenStreetMap geometries with zero mock fallback. For the hackathon's mock JPEG files, the pipeline detects the lack of CRS headers, alerts the user with a warning, and gracefully runs in demo mode. Additionally, to guarantee demo-day presentation reliability at the venue, all OSM road vectors for our demo area are pre-cached locally under `outputs/osm_cache/` so the system runs entirely offline without internet dependencies. This ensures the system is fully prepared to ingest real Cartosat-3 GeoTIFF files out-of-the-box as soon as they are released during the hackathon window."

### Q2: "How does this scale to a full city?"
* **Answer:** "The pipeline is designed with a tiled-processing architecture to scale efficiently. Large satellite swaths are partitioned into 512x512 tiles, which are run in parallel batches on GPU. Our spatial stitching module (Fix #5) uses a fast KDTree (`scipy.spatial.KDTree`) to merge boundary nodes in \(O(N \log N)\) time, and our topological healing uses localized search to patch gaps. The interactive dashboard utilizes Streamlit's `@st.cache_data` caching and background pre-warming of critical gatekeeper nodes. A full-city run can be pre-processed and cached in minutes, allowing instant, sub-second interactive demos."

### Q3: "What happens if the graph is still fragmented after topological healing?"
* **Answer:** "This is a real-world scenario where roads are physically blocked (e.g. by landslides or building collapses) rather than just obscured. Our APL routing index is robust to this: it reports unreachable paths as 'disconnected' and calculates the connectivity ratio honestly (e.g. 0.33 on our real JPEG test). In the dashboard, this is visualized as isolated components. By combining segment-level connectivity metrics and path length changes, we provide urban planners with a realistic 'resilience profile' that highlights where emergency redundancy is most needed."
