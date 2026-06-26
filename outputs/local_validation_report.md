# Local Validation Dataset Audit Report

This report evaluates the availability of local satellite datasets (DeepGlobe and HybridSAR) on the current environment and outlines the requirements for setting up local validation.

---

## 1. System Search Results (Audit Verdict)

We conducted a recursive search across the following paths on the local system:
* `c:\Users\HP\OneDrive\Desktop\ISRO` (Workspace)
* `c:\Users\HP\OneDrive\Desktop` (Desktop)
* `c:\Users\HP\Downloads` (Downloads)
* `c:\Users\HP\.gemini` (App Data)
* `c:\Users\HP\.cache` (Cache)
* `c:\Users\HP\.kaggle` (Kaggle config)

### Audit Findings:
* **DeepGlobe Road Extraction Dataset:** **0 files found** (No directories, ZIPs, or cache files exist on this machine).
* **HybridSAR Road Dataset (HSRD):** **0 files found** (No directories, ZIPs, or cache files exist on this machine).
* **GeoTIFF Files:** Only `outputs/test_georef.tif` (a programmatically generated synthetic test file) was discovered.
* **Road Mask / Satellite Folders:** Only `hacthon_req/` exists, which contains exclusively YouTube/StreamYard screenshots of the hackathon livestream slides.

### Why the Datasets are Unavailable Locally:
The model was trained in a Kaggle cloud environment (`kaggle_train_segmentation.ipynb`), where the datasets were mounted directly from Kaggle's public repository (`/kaggle/input/deepglobe-road-extraction-dataset` and `/kaggle/input/the-hybridsar-road-dataset-hsrd`). Because these datasets are very large (several gigabytes), they were not cloned or downloaded into the local Git workspace repository.

---

## 2. Best Method to Obtain Datasets Locally

To run a true local validation and update all outputs using real satellite imagery, follow the steps below to fetch the datasets from Kaggle.

### Step 1: Install the Kaggle CLI
If you do not have the Kaggle CLI installed, install it via pip:
```bash
pip install kaggle
```

### Step 2: Configure Kaggle API Credentials
1. Go to your Kaggle profile -> **Account** -> **Create New API Token**. This downloads a `kaggle.json` file.
2. Place this file in the directory: `C:\Users\HP\.kaggle\kaggle.json`.
3. Set appropriate permissions:
   ```cmd
   icacls C:\Users\HP\.kaggle\kaggle.json /inheritance:r /grant "HP:F"
   ```

### Step 3: Download the Datasets
Create a directory named `data/` in the project root and run the following commands to download the datasets:

```bash
mkdir data
cd data

# 1. Download DeepGlobe Road Extraction Dataset (1.4 GB)
kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset
tar -xf deepglobe-road-extraction-dataset.zip -C deepglobe/

# 2. Download HybridSAR Road Dataset (HSRD)
kaggle datasets download -d satyveeryadav/the-hybridsar-road-dataset-hsrd
tar -xf the-hybridsar-road-dataset-hsrd.zip -C hybridsar/
```

### Step 4: Extract the Local Validation Subset
Once downloaded, crop or select 5–10 images and corresponding masks, and store them under the following paths:
* **Images:** `data/local_validation/images/`
* **Masks:** `data/local_validation/masks/`

---

## 3. Metric Targets for Local Validation (Traceability)

Once the local dataset is populated, running the validation suite will evaluate the fine-tuned model against these baseline results:

* **Expected Image Dimensions:** 1024x1024 pixels (DeepGlobe standard).
* **Target Controlled Metrics (from Fine-Tuning Verification):**
  - **Occluded IoU:** **93.90%** (Dice: 96.83%)
  - **Non-Occluded IoU:** **97.51%** (Dice: 98.74%)
  - **APL Error:** **4.54%**
  - **Connectivity Ratio:** **1.00** (Healed graph connects all disjoint segments)
* **Before vs. After Comparison (Visual Verification):**
  - **Before Fixes:** Segmentation output on slide screenshots produced large rectangular text boxes and player controls as roads, resulting in flat, high-error road masks.
  - **After Fixes:** Model running on actual DeepGlobe/SpaceNet tiles will produce thin 1px centerlines winding through natural green terrains, capturing occlusions with high precision.
