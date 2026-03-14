# Crop Mapping Research — Project Context

## Overview

This is a **thesis research project** on crop mapping using satellite imagery and deep learning. The goal is to map crop types using multi-temporal Sentinel-2 multispectral imagery, trained against USDA Cropland Data Layer (CDL) labels.

Study area: **Sacramento Valley, California**
Coordinates: 122°18'36"W to 121°20'26"W, 38°41'7"N to 39°49'47"N
Area: ~2,038 km² (as covered by the S2 export grid)

---

## Tech Stack

- **Python** with virtual environment at `.venv/` (local) or `crop_mapping_pipeline/venv/` (GPU server)
- **geoai** (`geoai/`) — git submodule, a geospatial AI library (opengeos/geoai). Provides U-Net training, tiled inference, chip generation, and Sentinel-2 download via Planetary Computer STAC.
- **segmentation-models-pytorch** — U-Net, DeepLabV3+, SegFormer architectures
- **rasterio** — raster I/O
- **MLflow** — experiment tracking (remote server at `https://mlflow-geoai.stelarea.com`)
- **Jupyter notebooks** — main workspace for local experiments
- **crop_mapping_pipeline/** — standalone production pipeline for GPU server deployment

---

## Project Structure

```
research-crop-mapping-geoai/
├── AGENTS.md                   # This file
├── geoai/                      # Git submodule: opengeos/geoai library
├── notebooks/                  # Jupyter notebooks (numbered pipeline order)
│   ├── 00_setup_env.ipynb
│   ├── 01_fetch_data.ipynb
│   ├── 02_image_processing.ipynb
│   ├── 03_image_analysis.ipynb
│   ├── 04_feature_analysis.ipynb       # Stage 1 global joint score (v1)
│   ├── 04_feature_analysis_v2.ipynb    # Stage 1+2 per-crop binary selection (v2, active)
│   ├── 05_train_segmentation_model.ipynb
│   └── 06_mlflow_tracking.ipynb        # MLflow connectivity tests
├── crop_mapping_pipeline/      # Standalone GPU pipeline repo (independent git repo)
│   ├── pipeline.py             # CLI entry point
│   ├── config.py               # All constants, hyperparams, GDrive IDs
│   ├── requirements.txt
│   ├── README.md
│   ├── stages/
│   │   ├── fetch_data.py       # Stage 0: download processed S2+CDL from GDrive
│   │   ├── process_data.py     # Stage 0b: process raw S2+CDL, upload to GDrive
│   │   ├── feature_analysis.py # Stage 1+2: per-crop binary CNN forward selection
│   │   └── train_segmentation.py # Stage 3: Exp A/B/C × 2 archs = 6 runs
│   ├── models/
│   │   ├── cbam.py
│   │   ├── deeplabv3plus.py    # DeepLabV3+ with CBAM attention
│   │   └── segformer.py
│   ├── utils/
│   │   ├── band_selection.py   # GSI, RF importance, joint score, top-k
│   │   ├── constants.py        # USDA_CDL_COLORS, USDA_CDL_NAMES
│   │   ├── general.py          # GDrive download helper (gdown)
│   │   └── label.py            # Label utilities
│   └── ssh/                    # SSH keys + OAuth token (gitignored)
│       ├── runpod-cropmap      # ed25519 private key for RunPod
│       ├── runpod-cropmap.pub
│       ├── client_secret_*.json  # OAuth 2.0 Desktop app credentials
│       └── gdrive_token.pickle   # OAuth token (generated locally, transferred to VPS)
├── data/
│   ├── raw/
│   │   ├── s2/{year}/          # Raw S2 TIFs per year (temporary, deleted after processing)
│   │   └── cdl/                # CDL rasters (2022/2023/2024, EPSG:5070, 30m)
│   └── processed/
│       ├── s2/{year}/          # S2 with NoData assigned (*_processed.tif), per year
│       ├── cdl/                # CDL reprojected to S2 grid (EPSG:4326, ~10m)
│       │   └── cdl_{year}_study_area_filtered.tif
│       ├── stage2v2_per_crop_results.csv  # Stage 2 output: per-crop K*, IoU
│       └── stage3_exp_c_bands.txt         # Exp C band list (union of per-crop selections)
├── mlflow-research/            # MLflow Docker (local fallback)
│   ├── Dockerfile              # mlflow==3.10.1, PostgreSQL backend
│   ├── docker-compose.yml      # port 8088→8080
│   └── mlruns/artifacts/
├── documents/
│   ├── reports/                # Progress reports (report_YYYYMMDD.md)
│   ├── paper/
│   └── thesis/
│       ├── main.tex
│       ├── chapters/
│       ├── figures/
│       └── references.bib
└── src/                        # Legacy local utilities (superseded by crop_mapping_pipeline/)
    ├── models/
    └── utils/
```

---

## Data

### Satellite Imagery
- **Source**: Google Earth Engine — `COPERNICUS/S2_SR_HARMONIZED`
- **Export folder**: `S2_Annual_15d_sacramento_3` (Google Drive)
- **Years**: 2022, 2023, 2024 — acquisition interval ~15 days
- **Bands**: 11 bands (B1–B12, excluding B9/B10) → `S2_BAND_NAMES` in config.py
- **Dimensions**: 5,596 × 4,684 px, ~10m pixel size
- **CRS**: EPSG:4326 (WGS84)
- **Cloud filter**: ≤10% cloud cover
- **Best coverage date (2022)**: 2022-07-30 (99.6% valid pixels)
- **Files per year**: ~25 dates × 11 bands = 275 channels

### Labels
- **Source**: USDA NASS Cropland Data Layer (CDL)
- **Original**: EPSG:5070 (NAD83 / Conus Albers), 30m resolution, USA-wide
- **Processed**: reprojected + clipped + resampled to S2 grid in a single `rasterio.warp.reproject()` call

### CDL Class Setup (10 crops + background)
```python
KEEP_CLASSES = [3, 6, 24, 36, 37, 54, 69, 75, 76, 220]   # Fallow/61 → background (0)
CLASS_REMAP  = {cls_id: i+1 for i, cls_id in enumerate(KEEP_CLASSES)}
NUM_CLASSES  = 11   # 0=background + 1–10=crops
```

| ID | Class | Coverage |
|---|---|---|
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

Note: Fallow/Idle Cropland (CDL id=61) is remapped to background (class 0), not a crop class.

### Preprocessing Pipeline (`process_data.py` / `02_image_processing.ipynb`)
Order matters — NoData must be assigned BEFORE resampling:
```
Raw S2 TIFs → assign NoData (invalid/neg/NaN → -9999, float32) → save *_processed.tif
Raw CDL (EPSG:5070, 30m) → reproject+clip+resample to S2 grid → filter KEEP_CLASSES
Verify: same CRS, bounds, dimensions → upload to GDrive → (optionally) delete raw
```

---

## Novel Method: 3-Stage Band Selection

### Stage 1 — Per-Crop Filter Preselection
- Computes **per-crop SIglobal** (GSI per crop vs. rest) for all 275 channels
- Produces `candidates_per_crop[crop_id]` — top-20 bands ranked by SIglobal per crop
- Saves `stage1v2_candidates.json` to `data/processed/s2/2022/`
- MLflow: experiment `cropmap_feature_analysis_s2`, run `stage1v2_...`

### Stage 2 — Per-Crop Binary CNN Forward Selection (Novel Contribution)
- For each crop: evaluates candidates in Stage 1 rank order (O(N), not O(N²))
- Oracle: lightweight U-Net (ResNet-18 encoder), binary labels (crop vs. rest)
- Accept band iff `IoU(class1)_{t+1} > IoU(class1)_t + δ` (δ=0.005)
- Stop: `S2_NO_IMPROVE=5` consecutive rejections or `S2_MAX_BANDS=20`
- Each iteration: `S2_EPOCHS=15` epochs, early stopping (`S2_PATIENCE=5`)
- GPU optimisation: patches pre-loaded into `TensorDataset` → `num_workers=4 + pin_memory=True`
- Output: `selected_per_crop[crop_id]` → union → `stage3_exp_c_bands.txt`
- MLflow: nested runs — one parent + one child run per crop class
- **Stage 1 determines ORDER; Stage 2 determines HOW MANY (K\* per crop)**
- Uses 2022 data only as reference year

### Stage 3 — Full Model Validation

| Config | Input | Channels | exp_name | Purpose |
|---|---|---|---|---|
| **Exp A** | Single-date (Jul 30) | 9 | `exp_A_{arch}` | Conventional baseline |
| **Exp B** | 4 phenological dates (Jan/Mar/Jul/Nov) | 36 | `exp_B_{arch}` | Multi-temporal naive |
| **Exp C** | Stage 2 union of per-crop selections | K* | `exp_C_{arch}` | Proposed method |

- 2 architectures: `deeplabv3plus_cbam` (ResNet-50) + `segformer` (mit_b2)
- **6 total runs**: Exp A/B/C × 2 archs
- Train years: 2022 + 2023 | Test year: 2024
- MLflow: experiment `cropmap_segmentation_s2`
- Artifacts per run: `best_model.pth`, `last_model.pth`, `training_history.csv`, `training_curve.png`, `test_per_class_iou.csv`, `confusion_matrix.png`, `test_segmentation_map.png`

---

## Pipeline — GPU Server Deployment

### `crop_mapping_pipeline/` as standalone repo

All path calculations are relative to `crop_mapping_pipeline/` (not the notebook repo root):
```python
_ROOT = Path(__file__).parent.parent   # → crop_mapping_pipeline/  (in stages/)
sys.path.insert(0, str(_ROOT.parent))  # parent on sys.path so "from crop_mapping_pipeline.x" works
```

### Stage Commands

```bash
# Stage 0 — download processed S2 + CDL from GDrive
python stages/fetch_data.py --years 2022
python stages/fetch_data.py --verify-only

# Stage 0b — process raw S2, upload processed, delete raw (storage-constrained)
python stages/process_data.py --years 2022 --delete --shutdown
python stages/process_data.py --auth   # generate OAuth token locally first

# Stage 1+2 — per-crop feature analysis (GPU)
python stages/feature_analysis.py --stage 1   # CPU, run locally
python stages/feature_analysis.py --stage 2 --data-dir /workspace/crop_mapping_pipeline/data/processed

# Stage 3 — full training (GPU)
python stages/train_segmentation.py
python stages/train_segmentation.py --exp A --arch segformer   # single run

# Full pipeline
python pipeline.py --stages all --shutdown
python pipeline.py --stages feature train --years 2022
```

### Google Drive

- **OAuth 2.0** (Desktop app) for uploads — token at `ssh/gdrive_token.pickle`
- Token generated locally with `python stages/process_data.py --auth`, transferred via `0x0.st`
- Processed S2 folder IDs per year in `config.py` → `GDRIVE_PROCESSED_S2_FOLDER_IDS`
- Raw S2 folder IDs per year → `GDRIVE_RAW_S2_FOLDER_IDS`

### Auto-Shutdown (RunPod)

Set `RUNPOD_API_KEY` and `RUNPOD_POD_ID` in `crop_mapping_pipeline/.env`. The `--shutdown` flag calls RunPod GraphQL API `podStop` mutation (systemd unavailable in containers).

```bash
ssh -i ssh/runpod-cropmap h17406ec7sfiic-64411da1@ssh.runpod.io
```

---

## MLflow

- **Remote server**: `https://mlflow-geoai.stelarea.com` (primary, HTTPS required)
- **Local Docker fallback**: `mlflow-research/` (PostgreSQL backend, port 8088)
  - Start: `cd mlflow-research && docker compose up -d`
- **Experiments**:
  - `cropmap_pipeline_runs` — pipeline log uploads
  - `cropmap_feature_analysis_s2` — Stage 1 + Stage 2 runs
  - `cropmap_segmentation_s2` — Stage 3 training runs
- Set `MLFLOW_DISABLE_TELEMETRY=true` before `import mlflow` (prevents hang on MLflow 3.x)

---

## Dataset Scale

| Dataset | Files | Channels | Storage (float32) |
|---|---|---|---|
| S2 2022 (25 dates) | 25 | 275 | ~25 GB |
| S2 2022–2024 full | ~73 | 803 | ~73 GB |

**Hardware**: RTX 2000 Ada (16GB VRAM, 31GB RAM) on RunPod — sufficient for Stage 2 + Stage 3.

---

## Conventions

- Data files are gitignored; download via `stages/fetch_data.py` (gdown)
- S2 files organised in year subdirs: `data/processed/s2/{year}/*_processed.tif`
- Raw S2 also in year subdirs: `data/raw/s2/{year}/`
- `geoai/` is a git submodule — work on it independently, then update the pointer
- CDL class mapping: sequential 1–10 via `CLASS_REMAP` / `REMAP_LUT` (CDL IDs → model indices)
- Processed outputs always in `data/processed/` — never modify `data/raw/`
- `crop_mapping_pipeline/` is its own independent git repo — deployed standalone on GPU server
- Stage 2 uses 2022 data only as reference year for band selection
