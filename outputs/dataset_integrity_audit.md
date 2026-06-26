# Dataset Integrity Audit Report

This report documents the findings of a recursive audit of the dataset folders, imagery folders, and Jupyter Notebooks in the **Route Resilience Pipeline** repository to determine the presence, format, and validity of satellite road datasets.

---

## 1. Local Image & Dataset Search Results

We performed a recursive crawl across the workspace `c:\Users\HP\OneDrive\Desktop\ISRO` (excluding `.git`, `.gemini`, `__pycache__`, and generated folders inside `outputs/`).

### Scanned Folder Results
* **DeepGlobe Folders:** None found locally.
* **SpaceNet Folders:** None found locally.
* **Kaggle Local Datasets:** None found locally.
* **Local Images Found:**

| File Path | Dimensions | File Size | Content / Visual Description | Dataset Type | Status |
|---|---|---|---|---|---|
| **[1.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/1.jpeg)** | 1520x863x3 | 171.4 KB | YouTube frame of the introductory hackathon slide containing mentor names and titles. | Presentation slide screenshot | **INVALID** |
| **[2.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/2.jpeg)** | 1452x832x3 | 169.5 KB | Presentation slide titled "Overview: The Dual Challenge of Urban Spatial Modelling". | Presentation slide screenshot | **INVALID** |
| **[3.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/3.jpeg)** | 1536x863x3 | 141.6 KB | Presentation slide titled "Four Core Objectives". | Presentation slide screenshot | **INVALID** |
| **[4.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/4.jpeg)** | 1536x862x3 | 138.8 KB | Presentation slide titled "Deliverables / Expected Outcomes". | Presentation slide screenshot | **INVALID** |
| **[5.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/5.jpeg)** | 1536x863x3 | 147.8 KB | Presentation slide titled "Datasets Required". | Presentation slide screenshot | **INVALID** |
| **[6.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/6.jpeg)** | 1536x863x3 | 154.8 KB | Presentation slide titled "Suggested Tools & Technologies". | Presentation slide screenshot | **INVALID** |
| **[7.jpeg](file:///c:/Users/HP/OneDrive/Desktop/ISRO/hacthon_req/7.jpeg)** | 1447x863x3 | 127.3 KB | Presentation slide titled "Evaluation Parameters". | Presentation slide screenshot | **INVALID** |

### Synthetic Georeferenced Files (In outputs/)
* **[test_georef.tif](file:///c:/Users/HP/OneDrive/Desktop/ISRO/outputs/test_georef.tif):** 512x512x3 GeoTIFF, EPSG:32643, created programmatically by `scratch/create_test_geotiff.py` to verify the rasterio CRS reprojection and OSM live query. This is a synthetic test file, not a real satellite dataset block.

---

## 2. Remote Dataset References (Training Notebook)

We searched **[kaggle_train_segmentation.ipynb](file:///c:/Users/HP/OneDrive/Desktop/ISRO/kaggle_train_segmentation.ipynb)** to trace the dataset source used during model training:
* **DeepGlobe Path:** `'/kaggle/input/deepglobe-road-extraction-dataset'` (DeepGlobe Road Extraction Dataset used for model pre-training).
* **HybridSAR Path:** `'/kaggle/input/the-hybridsar-road-dataset-hsrd'` (The HybridSAR Road Dataset - HSRD used for domain-specific road training).
* **Source Path:** `'/kaggle/input/route-resillence-src'` (Project source code mounted as a dataset on Kaggle).

*Conclusion:* The real satellite datasets exist only on the Kaggle runtime environment where the model weights were compiled. No local copy of these datasets was included in the workspace repository.

---

## 3. Root Cause of Slide Ingestion

1. **Scan Scope:** The batch runner `run_pipeline.py` searches for all `*.jpeg` files in `hacthon_req/` and processes them as images.
2. **Missing Input Validation:** The pipeline had no input checks to verify image properties (such as aspect ratio, edge density, or local contrast), leading it to blindly ingest presentation screenshots.
3. **Segmentation Artifact:** Because presentation slides have rectangular boundaries, text lines, and box layouts, the road segmentation model predicted these sharp boundaries as road pixels, producing large blob masks and disconnected topological graphs.

---

## 4. Recommended Replacement Datasets

Since no local satellite imagery is present in the repository, we recommend sourcing and placing the following datasets in `hacthon_req/` during the hackathon data-release window:

1. **Cartosat-3 / Sentinel-2 GeoTIFFs (ISRO Hackathon Release):**
   - **Format:** GeoTIFF (contains projected CRS headers and affine coordinates).
   - **Impact:** Automatically activates the real geospatial reprojection pipeline (Fix #1), allowing correct live OSM queries and spatial node stitching without coordinates mismatch.
2. **DeepGlobe Road Extraction (Sample Tiles):**
   - **Source:** [Kaggle / DeepGlobe](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
   - **Format:** 1024x1024 JPEGs (satellite imagery) and PNGs (road binary masks).
   - **Impact:** Serves as a high-fidelity dataset for local testing of inference and centerline skeletonization.
3. **SpaceNet Road Extraction (Sample Tiles):**
   - **Source:** [SpaceNet Dataset](https://spacenet.ai/datasets/)
   - **Format:** 30cm resolution multi-spectral GeoTIFFs with vector road centerlines.
   - **Impact:** Provides real-world projected coordinate grids (UTM meters) to test route connectivity calculations under heavy occlusion.
