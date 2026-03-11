# Crop Type Mapping with Multi-Temporal Sentinel-2 and Semantic Segmentation

**Thesis research** — Master's level project on automated crop type mapping using satellite imagery and deep learning.

**Study area**: Sacramento Valley, California (122°18'36"W–121°20'26"W, 38°41'7"N–39°49'47"N, ~2,038 km²)

---

## Research Overview

This project develops and evaluates a **3-stage spectro-temporal band selection method** for crop mapping using multi-temporal Sentinel-2 (S2) multispectral imagery. The selected band set is validated by training semantic segmentation models against USDA Cropland Data Layer (CDL) labels.

### Research Questions

- Does multi-temporal imagery improve crop mapping accuracy over single-date imagery?
- Can an automated band selection method find a compact, discriminative feature subset that matches or exceeds the accuracy of using all temporal bands?

### Novel Contribution

A per-crop binary CNN forward selection method (ASTFS-inspired) that:
1. Ranks bands per crop using GSI (Global Separation Index) — O(N)
2. Evaluates ranked candidates with a lightweight U-Net oracle — O(N), not O(N²)
3. Produces a compact per-crop selected set; the union becomes the Exp C input

---

## Method: 3-Stage Band Selection

### Stage 1 — Per-Crop GSI Ranking (CPU)

For each of 10 crop classes, rank all 275 input channels (25 dates × 11 bands) by SIglobal — the mean pairwise separation index between the target crop and all others. Produces a ranked candidate list of top-20 bands per crop.

### Stage 2 — Binary CNN Forward Selection (GPU)

Per-crop greedy forward search in Stage 1 rank order. A binary U-Net (ResNet-18 encoder) is trained and evaluated for each candidate band in sequence. A band is accepted if the IoU gain ≥ δ (0.005). Stops after 5 consecutive rejections or 20 selected bands. The union of all per-crop selections becomes the Exp C feature set.

### Stage 3 — Comparative Experiment Design

| Config | Input | Channels | Purpose |
|---|---|---|---|
| **Exp A** | Single date (Jul 30, peak season) | 9 | Conventional baseline — no temporal info |
| **Exp B** | 4 phenological dates (Jan/Mar/Jul/Nov) | 36 | Multi-temporal naive — no selection |
| **Exp C** | Stage 2 union (proposed method) | K* ≤ 36 | Temporal + automated selection |

**A→B**: does temporal information help?
**B→C**: does selection find a more compact, equally accurate subset?

Each config × 2 architectures (DeepLabV3+CBAM, SegFormer) = **6 total runs**.
Train: 2022 + 2023 | Test: 2024 (temporal split).

---

## Data

| Item | Details |
|---|---|
| S2 source | Google Earth Engine — `COPERNICUS/S2_SR_HARMONIZED` |
| Years | 2022, 2023, 2024 |
| Acquisition interval | ~15 days |
| Bands | 11 (B1–B12, excl. B9/B10) |
| Dimensions | 5,596 × 4,684 px at ~10 m |
| CRS | EPSG:4326 (WGS84) |
| Cloud filter | ≤10% cloud cover |
| CDL source | USDA NASS Cropland Data Layer |
| CDL resolution | 30 m, EPSG:5070 → reprojected to S2 grid |
| Crop classes | 10 crops + background (see table below) |

### Crop Classes

| CDL ID | Class | Approx. Coverage |
|---|---|---|
| 75 | Almonds | 11.5% |
| 76 | Walnuts | 9.1% |
| 54 | Tomatoes | 7.4% |
| 3 | Rice | 6.6% |
| 24 | Winter Wheat | 4.7% |
| 6 | Sunflower | 3.5% |
| 36 | Alfalfa | 2.2% |
| 220 | Plums | 2.1% |
| 37 | Other Hay | 1.4% |
| 69 | Grapes | 1.3% |

Fallow/Idle Cropland (CDL id=61) → remapped to background (class 0).

---

## Project Structure

```
research-crop-mapping-geoai/
├── crop_mapping_pipeline/      # Standalone production pipeline (GPU server)
│   ├── pipeline.py
│   ├── config.py
│   ├── stages/
│   │   ├── fetch_data.py       # Stage 0: download from GDrive
│   │   ├── process_data.py     # Stage 0b: process raw S2+CDL, upload to GDrive
│   │   ├── feature_analysis.py # Stage 1+2: band selection
│   │   └── train_segmentation.py # Stage 3: 6 experiments
│   ├── models/                 # DeepLabV3+CBAM, SegFormer, CBAM
│   └── utils/                  # Band selection, CDL constants, label utilities
├── notebooks/                  # Jupyter notebooks for local exploration
│   ├── 02_image_processing.ipynb
│   ├── 03_image_analysis.ipynb
│   ├── 04_feature_analysis_v2.ipynb   # Per-crop band selection (v2)
│   └── 05_train_segmentation_model.ipynb
├── data/
│   ├── raw/s2/{year}/          # Raw GEE-exported S2 TIFs (temporary)
│   └── processed/
│       ├── s2/{year}/          # Processed S2 (*_processed.tif)
│       ├── cdl/                # CDL filtered rasters
│       ├── stage2v2_per_crop_results.csv
│       └── stage3_exp_c_bands.txt
├── documents/
│   ├── thesis/                 # LaTeX thesis document
│   └── reports/                # Progress reports (report_YYYYMMDD.md)
├── mlflow-research/            # MLflow Docker (local fallback)
└── geoai/                      # Git submodule: opengeos/geoai
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Satellite imagery | Google Earth Engine → Google Drive |
| Raster I/O | rasterio |
| Band selection | scikit-learn (RF), custom GSI |
| Deep learning | PyTorch, segmentation-models-pytorch |
| Architectures | DeepLabV3+ (ResNet-50 + CBAM), SegFormer (mit_b2) |
| Experiment tracking | MLflow (`https://mlflow-geoai.stelarea.com`) |
| GPU server | RunPod (NVIDIA RTX 2000 Ada, 16 GB VRAM) |

---

## Running the Pipeline

See [`crop_mapping_pipeline/README.md`](crop_mapping_pipeline/README.md) for full setup and usage instructions.

### Quick Start (GPU server)

```bash
# 1. Download processed data
python stages/fetch_data.py --years 2022

# 2. Run Stage 1 locally (CPU), copy handoff file to server
python stages/feature_analysis.py --stage 1
# scp / upload stage1v2_candidates.json to server

# 3. Run Stage 2 on GPU server
python stages/feature_analysis.py --stage 2 \
    --data-dir /workspace/crop_mapping_pipeline/data/processed

# 4. Run Stage 3 (all 6 experiments)
python stages/train_segmentation.py --shutdown
```

---

## MLflow Experiments

| Experiment | Runs |
|---|---|
| `cropmap_feature_analysis_s2` | Stage 1 + Stage 2 (nested per-crop runs) |
| `cropmap_segmentation_s2` | Stage 3 (6 runs: Exp A/B/C × 2 archs) |
| `cropmap_pipeline_runs` | Full pipeline execution logs |

Remote server: `https://mlflow-geoai.stelarea.com`

---

## References

- Yin, Q. et al. (2020). *Automatic Spectro-Temporal Feature Selection for Crop Mapping*. Remote Sensing 12(1):162. https://doi.org/10.3390/rs12010162
- Chen, L.C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+)*. ECCV.
- Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS.
- Woo, S. et al. (2018). *CBAM: Convolutional Block Attention Module*. ECCV.
