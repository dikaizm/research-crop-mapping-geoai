# Crop Mapping Research — Project Context

## Overview

This is a **thesis research project** on crop mapping using satellite imagery and deep learning. The goal is to map crop types using Sentinel-2 multispectral imagery, trained against USDA Cropland Data Layer (CDL) labels.

Study areas: **Sacramento**, **Sacramento_2**, and **Stockton** (California).

---

## Tech Stack

- **Python** with virtual environment at `.venv/`
- **geoai** (`geoai/`) — git submodule, a geospatial AI library (opengeos/geoai). Provides U-Net training, tiled inference, chip generation, and Sentinel-2 download via Planetary Computer STAC.
- **segmentation-models-pytorch** — U-Net and other segmentation architectures
- **rasterio** — raster I/O
- **MLflow** — experiment tracking (`mlflow-research/`)
- **Jupyter notebooks** — main workspace for experiments

---

## Project Structure

```
research-crop-mapping-geoai/
├── CLAUDE.md                   # This file
├── s2_segmentation.py          # Main pipeline script (download → chip → train → infer)
├── geoai/                      # Git submodule: opengeos/geoai library
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── segmentation_model_*.ipynb   # Model training iterations (00, 01, 02...)
│   ├── train_segmentation_model.ipynb
│   ├── image_analysis.ipynb
│   ├── image_processing.ipynb
│   ├── ingest_data.ipynb
│   ├── eval_segmentation_model.ipynb
│   ├── mlflow_tracking.ipynb
│   └── band_combination_results.csv
├── data/
│   ├── raw/
│   │   ├── images/             # Sentinel-2 GeoTIFFs (S2H_YYYY_YYYY_MM_DD_nodata.tif)
│   │   │   ├── sacramento/
│   │   │   ├── sacramento_2/
│   │   │   └── stockton/
│   │   ├── images_subset/      # Subset crops for fast iteration
│   │   ├── images_temp_band/   # Temporal band experiments
│   │   ├── labels/             # Vector/raster labels
│   │   └── cdl/                # USDA CDL rasters (2023_30m_cdls_*.tif)
│   ├── processed/              # Processed outputs
│   └── csv/                    # Label CSVs per area
│       ├── sacramento_cdl_labels.csv
│       ├── sacramento_2_cdl_labels.csv
│       └── stockton_cdl_labels.csv
├── models/                     # Saved model checkpoints
│   ├── crop_mapping/
│   ├── crop_mapping_multi_images/
│   ├── single_6c_2023_07_30/   # 6-channel single-date model
│   ├── single_7c_2023_06_30/   # 7-channel single-date model
│   └── ...
├── utils/
│   ├── constants.py            # USDA_CDL_COLORS, USDA_CDL_NAMES dicts
│   ├── band_selection.py       # GSI-based band selection for crop classification
│   ├── general.py              # Google Drive download helper (gdown)
│   └── label.py                # Label utilities
├── mlflow-research/            # MLflow tracking server (Docker)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── mlruns/
├── documents/
│   ├── paper/
│   └── thesis/
└── ssh/                        # SSH configs (ignored by git)
```

---

## Data

### Satellite Imagery
- **Sentinel-2** Harmonized (S2H) GeoTIFFs
- Naming: `S2H_{year}_{year}_{MM}_{DD}_nodata.tif`
- Dates used: 2023-05-01, 2023-05-31, 2023-06-30, 2023-07-30, 2024-07-29
- Full images and `_subset` versions for fast testing

### Labels
- **USDA CDL** (Cropland Data Layer) — 30m resolution, reprojected to 10m
- Key CDL rasters: `2023_30m_cdls_10m.tif`, `2023_30m_cdls_6c_10m.tif` (6-class version)
- Labels remapped to fewer classes for training (6-class setup)

### Band Configuration
- Models named with channel count: `6c`, `7c`, etc.
- Band selection done via **Global Separation Index (GSI)** (`utils/band_selection.py`)

---

## Key Workflows

### 1. Main Pipeline (`s2_segmentation.py`)
```
download_sentinel2() → chip_training_data() → train() → inference()
```

### 2. Notebooks Flow
- `ingest_data.ipynb` — data download and preprocessing
- `image_analysis.ipynb` / `image_processing.ipynb` — EDA and processing
- `segmentation_model_00/01/02.ipynb` — model training iterations
- `train_segmentation_model.ipynb` — training with geoai
- `eval_segmentation_model.ipynb` — evaluation
- `mlflow_tracking.ipynb` — experiment logging

### 3. MLflow
- Tracking server runs via Docker in `mlflow-research/`
- Experiments logged from notebooks

---

## Model Architecture
- **U-Net** with ResNet encoder (via segmentation-models-pytorch)
- Binary or multi-class segmentation of crop types
- Tiled inference on large GeoTIFFs

---

## Conventions
- Data files are gitignored; download via `utils/general.py` (Google Drive) or Planetary Computer STAC
- `geoai/` is a git submodule — work on it independently, then update the pointer in the outer repo
- Use `.venv/` for the Python environment
- Subset datasets (`images_subset/`) for rapid iteration before full-scale runs
