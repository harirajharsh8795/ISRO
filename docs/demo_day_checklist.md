# Demo-Day Offline & Reliability Checklist

This document details the configuration, pre-cached assets, and operational procedures to guarantee a crash-free live presentation of the **Route Resilience Pipeline** even with zero internet connectivity at the venue.

---

## 1. Pre-Cached OSM Data Status
Live OpenStreetMap (OSM) queries are typically made over the internet via `osmnx.features_from_bbox`. To prevent live-query hangs or timeouts on demo day, we have implemented an **automatic cache-first layer**. 

The system checks for local caches in `outputs/osm_cache/` before attempting any network query.

### Pre-Cached Bounding Box (Verification GeoTIFF Area)
* **Location:** Near Hassan, Karnataka (true georeferenced bounds of `test_georef.tif`)
* **Bounding Box (WSEN):** `(75.936019, 12.929165, 75.940756, 12.933778)`
* **Cache Key / Filename:** `outputs/osm_cache/8ef687336cb883ab88aac3f51c60b5dbe975008e9ea93c545781474e750b539a.geojson`
* **Status:** **Cached & Verified**. Completed in 0.05 seconds locally on cache hit.

---

## 2. How to Cache a New Area (If Demo Plan Changes)
If the team decides to show a different georeferenced GeoTIFF during the presentation, you **must pre-fetch the OSM data while connected to the internet** using the following helper pattern.

Run this simple Python snippet in your terminal to pre-populate the cache:

```python
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath("."))
from src.data_pipeline import fetch_osm_roads

# Define the new bounding box (North, South, East, West)
# Example coordinates:
north, south, east, west = 12.933778, 12.929165, 75.940756, 75.936019
bbox = (north, south, east, west)

print("Pre-fetching OSM data for new demo area...")
# This will query live OSM and write the cache file automatically
gdf = fetch_osm_roads(bbox, target_epsg=32643)
print(f"Pre-fetch completed. Cache file created in outputs/osm_cache/.")
```

---

## 3. Demo-Day Checklist (Step-by-Step)

### [ ] Step 1: Verify the Caches
Ensure the following files are present in the `outputs/` folder before walking up to the podium:
- `outputs/healed_graph.pkl` (cached road network graph)
- `outputs/unhealed_graph.pkl` (cached baseline network graph)
- `outputs/centrality.json` (cached centrality scores)
- `outputs/osm_cache/8ef687336cb883ab88aac3f51c60b5dbe975008e9ea93c545781474e750b539a.geojson` (cached ground-truth roads)

### [ ] Step 2: Launch Streamlit with `--fast-demo` Flag
To guarantee sub-2-second dashboard loading, always run the dashboard using the cached inputs flag:
```bash
streamlit run app/dashboard.py -- --fast-demo
```

### [ ] Step 3: Map Basemap Tile Limitation (Browser Render)
* **The Limitation:** Folium maps load OpenStreetMap background tiles dynamically in the user's web browser from `tile.openstreetmap.org`. If the demo laptop is completely offline, the browser will render a blank grid instead of the map tiles.
* **Resilience:** The dashboard **will not crash**. The interactive layers (road vectors, click-to-disable nodes, centrality heatmap circles) are loaded locally and will render perfectly over the grid.
* **Recommendation:** Keep the presentation laptop connected to a mobile hotspot if possible to render map tiles. If no connection is available, explain to the judges that the vector geometries are fully loaded from local GIS caches.
