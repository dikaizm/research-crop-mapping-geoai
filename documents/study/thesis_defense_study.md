# Thesis Defense Study Guide
**Pemetaan Tanaman Pertanian Berbasis Citra Multi-Temporal Sentinel-2**
*Defense: June 2026*

---

## 1. Research Overview (Hafal ini dulu)

### Problem
- Sentinel-2 multi-temporal → up to **648 band-date combinations** (9 bands × ~25 dates × 1 year)
- Not all combinations are equally informative → redundancy
- Using all features → high compute cost + risk of overfitting (Hughes phenomenon)
- Previous work: GSI/RF selection is **static** — not optimized against segmentation metric

### Proposed Method (thesis_v2)
**2-stage band selection:**
1. **Stage 1** — GSI-based per-crop per-date ranking
2. **Stage 2 (K-sweep)** — train full model directly at multiple K values → empirical IoU–K curve

### Research Questions
1. Seberapa besar perbedaan akurasi *single-date* vs *multi-temporal*?
2. Apakah pemilihan jendela fenologi memengaruhi performa *single-date*?
3. Berapa jumlah channel optimal K dari metode seleksi yang diusulkan?

---

## 2. Data

### Sentinel-2
| Property | Value |
|---|---|
| Source | GEE `COPERNICUS/S2_SR_HARMONIZED` |
| Bands | 9 bands: B2, B3, B4, B5, B6, B7, B8, B11, B12 |
| Interval | 15 days |
| Years | 2022 (train/ref), 2023 (train), 2024 (test) |
| Resolution | ~10m, EPSG:4326 |
| Image size | 5,596 × 4,684 px |
| Cloud filter | ≤10% cloud cover |
| Best date | 30 Jul 2022 (99.6% valid pixels) |

### CDL Labels
| Property | Value |
|---|---|
| Source | USDA NASS Cropland Data Layer |
| Original CRS | EPSG:5070, 30m |
| Processed | Reprojected + clipped + resampled to S2 grid |
| Classes used | 10 crops + background (class 0) |

### 10 Crop Classes
| ID | Crop | Coverage |
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

**Note:** Fallow/Idle Cropland (CDL id=61) → background (class 0), not a crop class.

---

## 3. Method — Stage 1: GSI Ranking

### Global Separation Index (GSI)
For each band $b$ at date $t$, for crop class $s$:

$$\text{GSI}_s^{(b,t)} = \frac{|\mu_s^{(b,t)} - \mu_o^{(b,t)}|}{1.96 \times (\sigma_s^{(b,t)} + \sigma_o^{(b,t)})}$$

- $\mu_s, \sigma_s$ = mean/std of crop class $s$ pixels
- $\mu_o, \sigma_o$ = mean/std of all other pixels
- Computed **per crop, per band, per date** → full separability map

### Phase A — Band Sweep
- Per crop: fix to top-1 SI date → rank all 9 bands by GSI descending
- Union top-K bands across all crops → $S_{\text{C\_v3}}(K)$ for Phase A
- k = 1..9

### Phase B — Date Sweep
- Per crop: use all 9 bands → rank all dates by GSI descending
- Union top-K dates × 9 bands across all crops → $S_{\text{C\_v3}}(K)$ for Phase B
- k = 1..10

### Key Stage 1 Results (from Stage 2v3 sweep)
**Phase A — top-1 date per crop:**
| Crop | Top-1 Date | Top Band |
|---|---|---|
| Rice | Jul 2022 | B12 |
| Sunflower | Oct 2022 | B8 |
| Winter Wheat | Dec 2022 | B1 |
| Alfalfa | Apr 2022 | B8A |
| Tomatoes | Dec 2022 | B1 |
| Grapes | May 2022 | B11 |
| Almonds | Dec 2022 | B1 |
| Walnuts | Oct 2022 | B6 |
| Plums | Jun 2022 | B7 |

**Phase A — union channel count:**
k=1→8ch, k=2→17ch, k=3→25ch, k=5→39ch, k=9→59ch

**Phase B — union channel count:**
k=1→59ch, k=2→107ch, k=3→147ch, k=5→212ch

---

## 4. Method — Stage 2: K-Sweep

Instead of CNN forward selection (expensive, sequential), directly train the full model at multiple K values.

**Why not CNN forward selection?**
- Forward selection requires $O(N)$ oracle training runs sequentially
- Each oracle run = 15 epochs × binary U-Net per crop → hundreds of GPU hours
- Results not directly optimized against multi-class mIoU

**Why K-sweep is better:**
- Each K is independent → embarrassingly parallel
- Uses the actual final model (DeepLabV3+CBAM / SegFormer), not a proxy
- Produces IoU–K curve → directly shows optimal K empirically
- Total compute = number of K points × 1 full training run

---

## 5. Experiments

| Exp | Input | Channels | MLflow Experiment | Purpose |
|---|---|---|---|---|
| **A** | Single date (Jul 30) | 9 | `cropmap_segmentation_s2` | Conventional baseline |
| **A_v2_Jan/Mar/Jul/Nov** | Single date per window | 9 each | `cropmap_segmentation_s2_v3` | Phenological window baseline |
| **B** | 4 phenological dates | 36 | `cropmap_segmentation_s2` | Multi-temporal naive |
| **C_v3_band_k01..k09** | Phase A sweep | 8–59 | `cropmap_segmentation_s2_v3` | Band count ablation |
| **C_v3_date_k01..k10** | Phase B sweep | 59–257 | `cropmap_segmentation_s2_v3` | Temporal depth ablation |

**2 architectures:** DeepLabV3+CBAM (ResNet-50) + SegFormer (MiT-B2)

**Train:** 2022 + 2023 | **Test:** 2024

### Exp A_v2 Rationale
Exp A uses Jul 30 (best data quality, not best phenology).
- Winter Wheat peaks Apr–May → already harvested in July → looks like bare soil in Exp A
- A_v2 tests: *"does choosing the right window fix this?"*
- A_v2_Mar expected to outperform A for Winter Wheat specifically

---

## 6. Architectures

### DeepLabV3+CBAM
- **Encoder:** ResNet-50 (ImageNet pretrained)
- **ASPP:** atrous convolution r ∈ {6, 12, 18} → multi-scale context
- **CBAM:** Channel Attention + Spatial Attention after ASPP
  - Channel Attention: which bands matter most
  - Spatial Attention: where in the image to focus
- **Why good for multi-temporal:** CBAM's channel attention naturally weights informative bands/dates

### SegFormer
- **Encoder:** Mix Transformer (MiT-B2) — hierarchical, no fixed positional encoding
- **Efficient Self-Attention:** sequence reduction for computational efficiency
- **Decoder:** All-MLP (lightweight)
- **Why good for single-date:** captures long-range spatial context → good for large uniform crop fields
- **Why struggles with multi-temporal:** self-attention sensitive to temporal noise/shift

---

## 7. Previous Results (Exp A/B/C baseline)

| Exp | Arch | Channels | Val mIoU | Test mIoU | Test OA |
|---|---|---|---|---|---|
| A (single-date) | SegFormer | 9 | 0.5056 | **0.1937** | 0.4392 |
| A (single-date) | DeepLabV3+ | 9 | 0.4693 | 0.0511 | 0.3167 |
| B (multi-temporal) | SegFormer | 36 | 0.4879 | 0.1017 | 0.2657 |
| B (multi-temporal) | DeepLabV3+ | 36 | 0.4825 | 0.0757 | 0.3639 |
| C (band selection) | SegFormer | 25 | 0.5082 | 0.0977 | 0.1646 |
| C (band selection) | DeepLabV3+ | 25 | 0.4774 | **0.1045** | 0.3987 |

### Key Observations
- **Val mIoU >> Test mIoU** → temporal domain shift (train 2022–2023, test 2024)
- SegFormer best on single-date (0.1937) but degrades on multi-temporal input
- DeepLabV3+ benefits most from band selection: Exp C (0.1045) >> Exp A (0.0511)
- Exp C used 25 channels vs 36 for Exp B → **79.8% feature reduction** from full space

### Per-Class IoU (Test 2024, %)
| Class | Exp A SF | Exp A DLV3 | Exp C DLV3 |
|---|---|---|---|
| Rice | 39.47 | 30.11 | 22.59 |
| Almonds | 36.95 | 4.90 | 23.85 |
| Grapes | 34.51 | 0.00 | 9.98 |
| Alfalfa | 16.55 | 7.30 | 11.41 |
| Tomatoes | 15.47 | 3.40 | 7.21 |
| Winter Wheat | 14.14 | 3.28 | 10.50 |
| Walnuts | 24.17 | 1.55 | 9.84 |
| Sunflower | 9.71 | 0.01 | 7.55 |
| Other Hay | 1.07 | 0.57 | 0.57 |
| Plums | 1.68 | 0.00 | 0.99 |

---

## 8. Why Temporal Domain Shift Happens

Train on 2022–2023, test on 2024 → calendar shift between years.

**Stable crops** (consistent phenology): Rice, Sunflower, Winter Wheat → NDVI correlation r > 0.86
**Unstable crops** (variable phenology):
- Alfalfa: 3–5 cuts/year, cutting schedule varies → r = -0.025 (2022 vs 2023)
- Grapes: anomaly in 2022 → r = +0.115
- Plums: only 2.1% area → mixed-pixel noise dominates

---

## 9. Anticipated Defense Questions

**Q: Kenapa pakai GSI bukan hanya RF importance?**
A: GSI lebih murah (statistik langsung, tidak perlu training), dan bekerja per kelas per tanggal. RF bersifat global (semua kelas digabung) dan tidak sensitif terhadap konteks spasial segmentasi.

**Q: Kenapa tidak pakai CNN forward selection seperti v1?**
A: CNN forward selection mahal secara komputasi dan menggunakan proxy model (U-Net binary per crop), bukan model akhir yang sebenarnya. K-sweep lebih jujur — menggunakan model yang sama persis dengan evaluasi akhir, dan bisa dijalankan paralel.

**Q: Kenapa test mIoU jauh lebih rendah dari val mIoU?**
A: Temporal domain shift — model dilatih pada 2022–2023 dan diuji pada 2024. Tanaman seperti Alfalfa dan Grapes memiliki fenologi yang tidak stabil antar tahun, sehingga spektral signature pada 2024 berbeda dari yang dipelajari model.

**Q: Kenapa SegFormer unggul di single-date tapi turun di multi-temporal?**
A: Self-attention SegFormer sensitif terhadap noise tambahan dari banyak channel. Single-date (9ch) memberikan input yang bersih → konteks spasial global bekerja optimal. Multi-temporal (36ch) menambah noise temporal → attention terdistraksi.

**Q: Kenapa memilih Sacramento Valley?**
A: Ketersediaan label publik CDL (USDA), keragaman jenis tanaman, iklim kering di musim panas (sedikit awan → data berkualitas), dan salah satu kawasan agrikultur paling produktif di Amerika Serikat.

**Q: Apakah hasil bisa digeneralisasi ke area lain?**
A: Tidak diklaim. GSI ranking dan K optimal mungkin berbeda untuk area dengan distribusi tanaman dan iklim berbeda. Namun metode 2-tahap (GSI + K-sweep) dapat diterapkan ke dataset manapun.

**Q: Apa kelebihan Exp A_v2 dibanding Exp A?**
A: Exp A memilih Jul 30 berdasarkan kualitas data, bukan fenologi. Exp A_v2 menguji apakah pemilihan tanggal yang tepat secara fenologi (misal Mar untuk Winter Wheat) meningkatkan performa. Ini menjawab: *apakah kelemahan Exp A adalah pilihan tanggal yang buruk, atau memang single-date fundamentally terbatas?*

**Q: Mengapa Fallow/Idle Cropland dijadikan background?**
A: Fallow/Idle tidak memiliki identitas tanaman yang konsisten — bisa berupa lahan kosong antar musim dari berbagai jenis tanaman. Memasukkannya sebagai kelas tersendiri akan menciptakan label yang ambigu dan merusak pelatihan.

**Q: Apa kontribusi utama penelitian ini?**
A: (1) Metode seleksi band 2-tahap berbasis GSI + K-sweep yang lebih sederhana dan efisien dibanding CNN forward selection; (2) Analisis komprehensif pengaruh jendela fenologi (Exp A_v2); (3) Perbandingan DeepLabV3+CBAM vs SegFormer pada input multi-temporal dengan variasi K.

---

## 10. Key Numbers to Memorize

| Fact | Value |
|---|---|
| Study area | 2,038 km², Sacramento Valley CA |
| S2 image size | 5,596 × 4,684 px |
| S2 bands used | 9 (B2,B3,B4,B5,B6,B7,B8,B11,B12) |
| Acquisition interval | 15 days |
| Total dates (2022) | 24 valid (1 excluded: 27 Dec, 0% valid) |
| Total feature space | ~648 band-date combinations |
| Best single date | 30 Jul 2022, 99.6% valid |
| CDL classes → model classes | 134 → 10 crops + background |
| Best test mIoU | 0.1937 (Exp A, SegFormer) |
| Best DeepLabV3+ test mIoU | 0.1045 (Exp C) |
| Feature reduction (Exp C) | 79.8% (124 → 25 channels) |
| Train years | 2022 + 2023 |
| Test year | 2024 |
| Phase A channels at k=3 | 25 channels |
| Phase B channels at k=1 | 59 channels |
