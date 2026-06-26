# Local Validation Checklist

This checklist guides the team or judges through configuring the local environment, fetching real satellite data, and running the local validation checks.

---

## 1. Setup & Environment Preparation
- [ ] **Kaggle CLI Installed:** Ensure python package `kaggle` is installed:
  ```bash
  pip install kaggle
  ```
- [ ] **Kaggle API Key configured:** Place your `kaggle.json` key in `C:\Users\HP\.kaggle\kaggle.json`.
- [ ] **Directories Created:** Verify that the folder structures are initialized:
  * `data/local_validation/images/`
  * `data/local_validation/masks/`

---

## 2. Sourcing Satellite Imagery

### Option A: DeepGlobe Road Extraction Dataset (1.4 GB)
- [ ] Run the download command:
  ```bash
  kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset
  ```
- [ ] Extract the archive and copy 5–10 sample images (e.g. `100034_sat.jpg` to `100099_sat.jpg`) to:
  `data/local_validation/images/`
- [ ] Copy the corresponding mask files (e.g. `100034_mask.png` to `100099_mask.png`) to:
  `data/local_validation/masks/` (rename them to match the image basenames, e.g. `100034_sat.png`).

### Option B: HybridSAR Road Dataset (HSRD)
- [ ] Run the download command:
  ```bash
  kaggle datasets download -d satyveeryadav/the-hybridsar-road-dataset-hsrd
  ```
- [ ] Extract the files and place the validation subset JPEGs into:
  `data/local_validation/images/`

---

## 3. Running Validation & Pipeline

- [ ] **Execute Validation Runner:** Run the single orchestration command:
  ```bash
  python run_local_validation.py
  ```
- [ ] **Check Console Output:** Verify that:
  - Slide screenshots/screenshots are detected and rejected.
  - Image-mask pairs are matched.
  - Segmentation inference and centerline graphs are processed.
- [ ] **Inspect Output Files:** Verify the generated validation files in `outputs/local_validation/`:
  * `dataset_manifest.json` (lists validation status of all scanned images)
  * `metrics.json` (stores mean IoU, Dice, and graph nodes/edges stats)
  * `unhealed_graph.pkl` & `healed_graph.pkl` (pickled NetworkX outputs)
  * `criticality.geojson` (vector coordinates of gatekeeper intersections)
  * `mask_check_5.png` (plot overlay showing real satellite road segmentations)
  * `diagnostic_occlusion_mask_check.png` (4-panel occlusion recovery plot using real satellite data)
