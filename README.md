# 🛰️ Route Resilience Intelligence System
### ISRO PS-4 · Bharatiya Antariksh Hackathon 2026

> **Problem Statement PS-4:** Develop an AI/ML-based system to extract road networks from satellite imagery and assess their resilience under disaster scenarios.

---

## 🎯 What This System Does

Given a satellite image tile, the system:

1. **Segments roads** using a deep learning model (UNet + ResNet-34)
2. **Extracts a road graph** via skeletonisation → graph conversion
3. **Heals fragmented topology** using Minimum Spanning Tree bridge insertion
4. **Identifies critical junctions** using betweenness centrality
5. **Simulates node failures** to compute network resilience index
6. **Visualises everything** on an interactive Streamlit dashboard

---

## 🏗️ System Architecture

```
Satellite Image (RGB, 512×512)
        │
        ▼
┌─────────────────────┐
│  UNet + ResNet-34   │  ← trained on DeepGlobe Road Extraction Dataset
│  Road Segmentation  │
└─────────────────────┘
        │ binary road mask
        ▼
┌─────────────────────┐
│  Skeletonisation     │  ← Zhang-Suen thinning → 1px centrelines
│  (skimage)           │
└─────────────────────┘
        │ skeleton pixels
        ▼
┌─────────────────────┐
│  Graph Extraction    │  ← pixels → nodes (junctions) + edges (roads)
│  (skeleton_to_graph) │     with UTM-projected spatial coordinates
└─────────────────────┘
        │ fragmented nx.Graph
        ▼
┌─────────────────────┐
│  Graph Healing       │  ← MST-based bridge edge insertion
│  (graph_healing)     │     reconnects isolated components
└─────────────────────┘
        │ healed nx.Graph
        ▼
┌─────────────────────┐
│  Centrality Analysis │  ← betweenness centrality (igraph, fast)
│  (centrality.py)     │     exports criticality.geojson
└─────────────────────┘
        │ criticality scores
        ▼
┌─────────────────────┐
│  Resilience Sim      │  ← node ablation: remove node, sample 200 OD pairs
│  (resilience.py)     │     compute: RI, % routes broken, avg path increase
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Streamlit Dashboard │  ← live map, failure demo, pipeline walkthrough
│  (app/dashboard.py)  │
└─────────────────────┘
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch torchvision segmentation-models-pytorch
pip install rasterio pyproj networkx scikit-image
pip install streamlit streamlit-folium folium igraph
pip install opencv-python albumentations pyogrio
```

### Step 1 — Run the Pipeline
```bash
python run_pipeline.py
```
Processes images from `data/local_validation/images/` (falls back to `hacthon_req/`).

Generates:
- `outputs/healed_graph.pkl`
- `outputs/unhealed_graph.pkl`
- `outputs/centrality.json`
- `outputs/criticality.geojson`

### Step 2 — Launch Dashboard
```bash
streamlit run app/dashboard.py
```
Open → **http://localhost:8501**

---

## 📊 Results

### DeepGlobe Validation (In-Distribution)
| Metric | Score |
|:---|---:|
| IoU | **55.83%** |
| Dice | **71.37%** |
| Architecture | UNet / ResNet-34 |
| Training epochs | 29 |
| Dataset | DeepGlobe Road Extraction |

### Massachusetts Roads (Cross-Dataset Evaluation)
| Metric | Score |
|:---|---:|
| IoU | 2.33% |
| Dice | 4.44% |

> ⚠️ The cross-dataset drop is expected **domain shift** — different sensor (aerial vs satellite), geography (USA vs SE Asia), and resolution (1m vs 50cm GSD). The DeepGlobe 55.83% IoU is the primary validated result.

### Graph Metrics (on real satellite tiles)
| Metric | Before Healing | After Healing |
|:---|---:|---:|
| Connected components | 38 | 23 |
| Largest connected component | 51 nodes | 57 nodes |
| Bridge edges added | — | +15 |
| Total edges | 133 | 148 |

---

## 🔬 Key Innovations

1. **Occlusion-Aware Segmentation** — model trained with synthetic shadow augmentation to handle occluded roads (under trees, bridges, clouds)

2. **Topological Graph Healing** — fragmented road network components reconnected using minimum-cost bridge edges (MST-based), not just spatial proximity

3. **Betweenness Centrality for Criticality** — identifies high-betweenness junctions that, if removed, maximally disconnect the network

4. **Resilience Index** — quantitative metric: fraction of sampled OD pairs that remain reachable after node failure, weighted by path length change

5. **End-to-End Pipeline** — single command `python run_pipeline.py` runs all stages from raw image to GeoJSON criticality export

---

## 📁 Project Structure

```
ISRO/
├── app/
│   └── dashboard.py          ← Streamlit demo dashboard
├── src/
│   ├── segmentation_train.py ← UNet model, training loop, predict_mask
│   ├── skeleton_to_graph.py  ← mask → skeleton → nx.Graph
│   ├── graph_healing.py      ← MST-based bridge insertion
│   ├── centrality.py         ← betweenness centrality + GeoJSON export
│   ├── resilience.py         ← node ablation simulation
│   ├── geo_loading.py        ← rasterio image loading with CRS handling
│   └── spatial_stitching.py  ← cross-tile node stitching
├── checkpoints/
│   └── best_model.pth        ← trained model weights (97.9 MB)
├── data/
│   └── local_validation/
│       ├── images/            ← real satellite tiles (PNG)
│       └── masks/             ← ground truth road masks (PNG)
├── outputs/
│   ├── healed_graph.pkl       ← healed road network graph
│   ├── unhealed_graph.pkl     ← raw fragmented graph
│   ├── centrality.json        ← betweenness scores per node
│   └── criticality.geojson    ← spatial export for GIS tools
├── run_pipeline.py            ← end-to-end batch runner
└── README.md                  ← this file
```

---
## Datasets Used

### Training
- DeepGlobe Road Extraction Dataset
- Resolution: ~50 cm GSD
- RGB satellite imagery

### Local Validation
- Massachusetts Roads Dataset
- Used only for cross-dataset evaluation

### Future Deployment
- Cartosat-3
- Sentinel-2
- ISRO Hackathon imagery
## 🎤 For Judges

**30-second walkthrough:**

1. Open dashboard → **Tab 1: "How It Works"** — see all 6 pipeline stages on one real satellite tile
2. Go to **Tab 2: "Live Map"** — toggle graph layers, centrality heatmap
3. Go to **Tab 3: "Node Failure Demo"** → press **"Run Demo Scenario"** — instant impact metrics
4. Go to **Tab 4: "Metrics"** — see DeepGlobe vs cross-dataset results with domain shift explanation

**Key claim:**The system is designed to process high-resolution satellite imagery and extract road networks automatically.
