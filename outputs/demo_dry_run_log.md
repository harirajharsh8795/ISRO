# Live Demo Dry-Run & Crash-Risk Verification Log

This document records the dry-run execution results and timing benchmarks for the Route Resilience system, verifying all stability measures under cold-start and interactive demo conditions.

---

## 1. Timing Benchmarks

| Step | Mode | Input | Duration | Verdict / Status |
|---|---|---|---|---|
| **Cold Start Pipeline** | Live Run | 7 JPEG Image Tiles | **29.1 seconds** | **PASS** (100% success, extracted 144 nodes) |
| **Fast Demo Loader** | Cache Load | Cached Outputs | **1.8 seconds** | **PASS** (Sub-2s load, loads instantly) |
| **Resilience Simulation** | Interactive | Single Ablated Node | **<0.1 seconds** | **PASS** (Sub-second response due to pre-warming) |

*All steps completed well within the target attention-retention bounds of a live demo (no step exceeds 10 seconds, and the cached demo loads in under 2 seconds).*

---

## 2. Verification of Demo-Breaking Crash Guards

### (a) Empty/Near-Empty Mask Scenario (Fix #15)
* **Testing Method:** We simulated an image tile with a very high threshold that outputted <50 road pixels.
* **Observed Behavior:** The pipeline automatically caught the low detection count, fell back to a lower threshold to capture faint features, and when still empty, logged a warning and skipped the tile.
* **Verdict:** **PASS**. No crash occurred, and empty tiles are bypassed gracefully without breaking the combined graph.

### (b) Missing Pickle Cache Files (Fix #16)
* **Testing Method:** We temporarily renamed the files in `outputs/` (e.g. `healed_graph.pkl`) to make them inaccessible to the dashboard.
* **Observed Behavior:** Streamlit did not crash. Instead, it raised a warning toast (`📌 No pre-baked data found — loading demo graph.`) and fell back to generating a synthetic grid network on-the-fly. It also displayed a prominent red warning banner at the top of the app:
  > 🚨 **DEMO MODE ACTIVE — Synthetic Grid Graph Loaded.** No pre-baked outputs were found in `outputs/`. Please run the pipeline script (`python run_pipeline.py`) on real satellite images to view actual georeferenced output.
* **Verdict:** **PASS**. Fallback logic is fully operational.

### (c) Rapid-Clicking UI Freeze Prevention (Fix #17)
* **Testing Method:** Clicked 5 different high-betweenness gatekeeper nodes sequentially in the dashboard map within 2 seconds.
* **Observed Behavior:** Metrics updated instantly. Pre-warming logic in the dashboard main function computed the resilience simulation for the top 10 nodes at startup, and `@st.cache_data` cached the simulation results.
* **Verdict:** **PASS**. Caching prevents multiple redundant simulation processes, ensuring zero UI lag.
