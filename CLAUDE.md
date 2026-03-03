# Crop Mapping Research — Project Context

## Overview

This is a **thesis research project** on crop mapping using satellite imagery and deep learning. The goal is to map crop types using multi-temporal Sentinel-2 multispectral imagery, trained against USDA Cropland Data Layer (CDL) labels.

Study area: **Sacramento Valley, California**
Coordinates: 122°18'36"W to 121°20'26"W, 38°41'7"N to 39°49'47"N
Area: ~2,038 km² (as covered by the S2 export grid)

---

## Tech Stack

- **Python** with virtual environment at `.venv/`
- **geoai** (`geoai/`) — git submodule, a geospatial AI library (opengeos/geoai). Provides U-Net training, tiled inference, chip generation, and Sentinel-2 download via Planetary Computer STAC.
- **segmentation-models-pytorch** — U-Net, DeepLabV3+, SegFormer architectures
- **rasterio** — raster I/O
- **MLflow** — experiment tracking (remote server at `http://mlflow-geoai.stelarea.com`)
- **Jupyter notebooks** — main workspace for experiments

---

## Project Structure

```
research-crop-mapping-geoai/
├── CLAUDE.md                   # This file
├── s2_segmentation.py          # Legacy pipeline script
├── geoai/                      # Git submodule: opengeos/geoai library
├── notebooks/                  # Jupyter notebooks (numbered pipeline order)
│   ├── 00_setup_env.ipynb          # Environment setup
│   ├── 01_fetch_data.ipynb         # Data download (gdown from Google Drive)
│   ├── 02_image_processing.ipynb   # Preprocessing (CRS alignment, NoData, filtering)
│   ├── 03_image_analysis.ipynb     # EDA & visualization
│   ├── 04_feature_analysis.ipynb   # 3-stage band selection
│   ├── 05_train_segmentation_model.ipynb  # Model training
│   ├── 06_segmentation_model_00.ipynb     # U-Net baseline
│   ├── 07_segmentation_model_01.ipynb     # DeepLabV3+ experiments
│   ├── 08_segmentation_model_01-1.ipynb   # DeepLabV3+ variants
│   ├── 09_segmentation_model_02.ipynb     # SegFormer experiments
│   ├── 10_eval_segmentation_model.ipynb   # Evaluation
│   └── 11_mlflow_tracking.ipynb           # MLflow logging
├── data/
│   ├── raw/
│   │   └── cdl/                # CDL rasters (2022/2023/2024, EPSG:5070, 30m)
│   └── processed/
│       ├── s2/                 # S2 with NoData assigned (*_processed.tif)
│       └── cdl/                # CDL reprojected to S2 grid (EPSG:4326, ~10m)
│           ├── cdl_2022_study_area.tif          # reprojected
│           └── cdl_2022_study_area_filtered.tif # filtered (11 classes + background)
├── models/                     # Saved model checkpoints
├── utils/
│   ├── constants.py            # USDA_CDL_COLORS, USDA_CDL_NAMES dicts
│   ├── band_selection.py       # GSI, RF importance, joint score, top-k selection
│   ├── general.py              # Google Drive download helper (gdown)
│   └── label.py                # Label utilities (label_filtering with .copy() fix)
├── mlflow-research/            # MLflow Docker (local fallback)
│   ├── Dockerfile              # python:3.11-slim, mlflow==2.14.3, psycopg2-binary
│   ├── docker-compose.yml      # PostgreSQL backend, port 8088→8080
│   └── mlruns/
│       └── artifacts/
├── documents/
│   ├── crop_mapping_pipeline.excalidraw
│   ├── reports/                # Progress reports (report_YYYYMMDD.md)
│   ├── paper/
│   └── thesis/
│       ├── main.tex
│       ├── chapters/           # abstract, introduction, literature_review,
│       │                       # methodology, results, conclusion
│       ├── figures/            # publication-quality figures
│       └── references.bib
└── ssh/                        # SSH configs (ignored by git)
```

---

## Data

### Satellite Imagery
- **Source**: Google Earth Engine — `COPERNICUS/S2_SR_HARMONIZED`
- **Export folder**: `S2_Annual_15d_sacramento_3` (Google Drive File Stream)
- **Years**: 2022, 2023, 2024 — acquisition interval ~15 days
- **Bands**: 11 bands (B1–B12, excluding B9/B10)
- **Dimensions**: 5,596 × 4,684 px, ~10m pixel size
- **CRS**: EPSG:4326 (WGS84)
- **Cloud filter**: ≤10% cloud cover
- **Best coverage date (2022)**: 2022-07-30 (99.6% valid pixels)

### Labels
- **Source**: USDA NASS Cropland Data Layer (CDL)
- **Original**: EPSG:5070 (NAD83 / Conus Albers), 30m resolution, USA-wide
- **Processed**: reprojected + clipped + resampled to S2 grid in a single `rasterio.warp.reproject()` call

### CDL Class Setup (11 classes + background)
```python
KEEP_CLASSES = [3, 6, 24, 36, 37, 54, 61, 69, 75, 76, 220]
# Extended: [3, 6, 24, 33, 36, 37, 54, 61, 69, 75, 76, 204, 220]
```

| ID | Class | Coverage |
|---|---|---|
| 61 | Fallow/Idle Cropland | 28.3% |
| 75 | Almonds | 11.5% |
| 76 | Walnuts | 9.1% |
| 54 | Tomatoes | 7.4% |
| 3 | Rice | 6.6% |
| 24 | Winter Wheat | 4.7% |
| 6 | Sunflower | 3.5% |
| 36 | Alfalfa | 2.2% |
| 220 | Plums | 2.1% |
| 37 | Other Hay/Non Alfalfa | 1.4% |
| 69 | Grapes | 1.3% |

### Preprocessing Pipeline (`02_image_processing.ipynb`)
Order matters — NoData must be assigned BEFORE resampling to prevent interpolation artifacts:
```
CDL (EPSG:5070, 30m) → reproject+clip+resample to S2 grid → filter classes
S2 files → assign NoData (invalid/neg/NaN → -9999, cast float32)
Verify: same CRS, bounds, dimensions
```

---

## Novel Method: 3-Stage Band Selection

### Stage 1 — Filter Preselection (`04_feature_analysis.ipynb`)
Scores all input channels using:
- **GSI** (Global Separation Index): class separability per band
- **RF Importance**: mean decrease in Gini impurity (`compute_rf_importance()`)
- **Joint Score**: `Score_b = α·GSI_norm + (1-α)·RF_norm` (α=0.5, `compute_joint_score()`)
- Top-K selection via `select_top_k()`

### Stage 2 — CNN Forward Selection (Novel Contribution)
- Evaluates features **in Stage 1 rank order** (O(N) vs O(N²) for pure greedy search)
- Oracle: lightweight U-Net (ResNet-18 encoder), `RasterPatchDataset` (on-the-fly patches)
- Accept band iff `mIoU_{t+1} > mIoU_t + δ` (δ=0.005)
- Stop: 3 consecutive rejections or max 15 bands
- Each iteration: 15 epochs, early stopping (patience=5)
- **Stage 1 determines ORDER; Stage 2 determines HOW MANY (K\*)**

### Stage 3 — Full Model Validation (Experiment Design)

| Config | Input | Channels | Purpose |
|---|---|---|---|
| **Exp A** | Single-date baseline (Jul 30, peak season) | 9 | No temporal info |
| **Exp B** | 4 phenological dates (Jan, Mar, Jul, Nov) | 36 | Temporal, no selection |
| **Exp C** | Stage 2 output (proposed method) | K* < 36 | Temporal + optimal selection |

A→B: does temporal information help? B→C: does selection find a more compact subset?
Each config × 2 architectures (DeepLabV3+CBAM, SegFormer) = **6 total runs**.

---

## Key Workflows

### 1. Notebooks Flow (in order)
1. `00_setup_env.ipynb` — environment check
2. `01_fetch_data.ipynb` — download from Google Drive (gdown)
3. `02_image_processing.ipynb` → outputs to `data/processed/`
4. `03_image_analysis.ipynb` — EDA, NDVI temporal analysis, figures
5. `04_feature_analysis.ipynb` — Stage 1 + Stage 2 band selection (MLflow logged)
6. `05–09` — model training (U-Net, DeepLabV3+, SegFormer)
7. `10_eval_segmentation_model.ipynb` — evaluation
8. `11_mlflow_tracking.ipynb` — MLflow experiment review

### 2. MLflow
- **Remote server**: `http://mlflow-geoai.stelarea.com` (primary)
- **Experiment name**: `research-crop-mapping`
- **Local Docker fallback**: `mlflow-research/` (PostgreSQL backend, port 8088)
  - Start: `cd mlflow-research && docker compose up -d`
  - Stop: `docker compose down`
- Stage 1 MLflow run: `stage1_gsi_rf_YYYYMMDD-HHMMSS`
- Stage 2 MLflow run: `stage2_cnn_fwd_YYYYMMDD-HHMMSS` (linked to stage1_run_id)

---

## Model Architectures
- **U-Net** with ResNet encoder (baseline)
- **DeepLabV3+** with CBAM attention
- **SegFormer** (transformer-based)
- All via segmentation-models-pytorch; tiled inference on large GeoTIFFs

---

## Dataset Scale

| Dataset | Files | Patches (256×256) | Storage (float32) |
|---|---|---|---|
| S2 2022 (20 files) | 20 | 7,560 | ~20 GB |
| S2 2022–2024 full | ~60 | ~22,680 | ~60 GB |

**Hardware**: RTX 4090 (24GB VRAM) — sufficient for all configurations.

---

## Conventions
- Data files are gitignored; download via `utils/general.py` (gdown) or mount Google Drive File Stream
- `geoai/` is a git submodule — work on it independently, then update the pointer
- Use `.venv/` for the Python environment
- Use subset/sample data for fast iteration before full-scale runs
- CDL class mapping: sequential 1–11 via `CLASS_REMAP` / `REMAP_LUT` (CDL IDs → model indices)
- Processed outputs always in `data/processed/` — never modify `data/raw/`
