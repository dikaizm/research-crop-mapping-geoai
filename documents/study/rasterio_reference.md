# Rasterio Reference — Crop Mapping Pipeline

Covers the rasterio APIs actually used in the notebooks and pipeline stages.

---

## 1. Opening Files

```python
import rasterio

# Read mode (context manager — always preferred)
with rasterio.open("file.tif") as src:
    data = src.read(1)          # band 1, returns np.ndarray (H, W)
    data = src.read()           # all bands, returns (bands, H, W)

# Write mode
with rasterio.open("out.tif", "w", **profile) as dst:
    dst.write(array, 1)         # write band 1
    dst.write(array)            # write all bands, array shape (bands, H, W)

# Keeping multiple files open (for tiled inference / patch loading)
srcs = [rasterio.open(p) for p in paths]
# ... use srcs ...
for s in srcs:
    s.close()
```

---

## 2. File Metadata

```python
with rasterio.open("file.tif") as src:
    src.crs          # CRS object, e.g. CRS.from_epsg(4326)
    src.transform    # Affine transform (pixel → coordinates)
    src.width        # number of columns (int)
    src.height       # number of rows (int)
    src.count        # number of bands (int)
    src.dtypes       # tuple of dtype per band, e.g. ('float32',)
    src.nodata       # nodata value or None
    src.res          # (pixel_width_deg, pixel_height_deg) tuple
    src.bounds       # BoundingBox(left, bottom, right, top)

    # bounds fields
    b = src.bounds
    b.left, b.right, b.bottom, b.top
```

### Profile dict (used for writing)

`src.profile` returns a dict with all metadata — copy and modify before writing:

```python
with rasterio.open("input.tif") as src:
    profile = src.profile.copy()

profile.update(
    dtype   = "float32",
    nodata  = -9999.0,
    count   = 1,
    compress= "lzw",
)

with rasterio.open("output.tif", "w", **profile) as dst:
    dst.write(array, 1)
```

Common profile keys:

| Key | Description |
|---|---|
| `driver` | File format, always `"GTiff"` |
| `dtype` | `"float32"`, `"uint8"`, `"int32"` |
| `nodata` | NoData sentinel value |
| `width` / `height` | Raster dimensions in pixels |
| `count` | Number of bands |
| `crs` | Coordinate reference system |
| `transform` | Affine pixel-to-coordinate transform |
| `compress` | `"lzw"`, `"deflate"` — reduces file size |
| `tiled` | `True` for tiled GeoTIFFs (better random access) |
| `blockxsize` / `blockysize` | Tile size, e.g. `512` |

---

## 3. Reading Windows (Patches)

Used in `train_segmentation.py` and `feature_analysis_v2.py` for tiled/patch-based reading without loading the full raster.

```python
from rasterio.windows import Window

# Window(col_off, row_off, width, height)
win = Window(col_offset, row_offset, patch_size, patch_size)

with rasterio.open("file.tif") as src:
    patch = src.read(window=win)          # all bands → (bands, H, W)
    patch = src.read(1, window=win)       # band 1   → (H, W)
```

### Patch loop pattern (used in pipeline)

```python
patch_size = 256
stride     = 256

with rasterio.open("file.tif") as src:
    for row in range(0, src.height - patch_size + 1, stride):
        for col in range(0, src.width - patch_size + 1, stride):
            win   = Window(col, row, patch_size, patch_size)
            patch = src.read(window=win)   # (bands, 256, 256)
```

---

## 4. Merging Tiles (rasterio.merge)

Used in `process_data_v2.py` to mosaic GEE multi-tile exports into one image per date.

```python
from rasterio.merge import merge as rio_merge

srcs = [rasterio.open(p) for p in tile_paths]
try:
    mosaic, transform = rio_merge(srcs)   # mosaic: (bands, H, W)
finally:
    for s in srcs:
        s.close()

# Write mosaic
profile = srcs[0].profile.copy()
profile.update(
    width    = mosaic.shape[2],
    height   = mosaic.shape[1],
    transform= transform,
)
with rasterio.open("merged.tif", "w", **profile) as dst:
    dst.write(mosaic)
```

- Handles overlapping tiles — first valid pixel wins by default
- Input tiles must share the same CRS and band count

---

## 5. Reprojection (rasterio.warp)

Used in `process_data_v2.py` to reproject CDL (EPSG:5070, 30m) onto the S2 grid (EPSG:4326, ~10m).

```python
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# Read target grid from S2 reference file
with rasterio.open(s2_ref_path) as s2_ref:
    target_crs       = s2_ref.crs
    target_transform = s2_ref.transform
    target_width     = s2_ref.width
    target_height    = s2_ref.height

# Reproject source into target grid
with rasterio.open(cdl_raw_path) as cdl_src:
    dst_data = np.zeros((1, target_height, target_width), dtype=np.uint8)
    reproject(
        source        = rasterio.band(cdl_src, 1),  # source band reference
        destination   = dst_data,
        src_transform = cdl_src.transform,
        src_crs       = cdl_src.crs,
        dst_transform = target_transform,
        dst_crs       = target_crs,
        resampling    = Resampling.nearest,          # nearest for categorical labels
    )
```

### Resampling methods

| Method | Use case |
|---|---|
| `Resampling.nearest` | Categorical data (CDL labels) — preserves class values |
| `Resampling.bilinear` | Continuous data (elevation, temperature) |
| `Resampling.cubic` | Higher quality continuous data |
| `Resampling.average` | Downsampling continuous data |

### `rasterio.band(src, band_index)`

Wraps an open dataset + band index into a `Band` object for use as `reproject()` source. `band_index` is 1-based.

```python
rasterio.band(src, 1)   # first band
```

---

## 6. NoData Assignment

Pattern used in `process_data_v2.py` to mark invalid pixels as `-9999` before saving:

```python
S2_NODATA = -9999.0

with rasterio.open(in_path) as src:
    profile = src.profile.copy()
    data    = src.read().astype(np.float32)   # (bands, H, W)

# Mask invalid pixels → nodata
data[data < 0]      = S2_NODATA
data[np.isnan(data)]= S2_NODATA

profile.update(dtype="float32", nodata=S2_NODATA, compress="lzw")

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(data)
```

---

## 7. Class Filtering (label raster)

Used in `utils/label.py` to zero-out non-target CDL classes:

```python
import numpy as np
import rasterio

keep_classes = [3, 6, 24, 36, 37, 54, 69, 75, 76, 210]

with rasterio.open(in_path) as src:
    profile  = src.profile.copy()
    data     = src.read(1)          # (H, W), uint8
    nodata   = src.nodata or 0

profile.update(nodata=nodata)

# Keep selected classes; everything else → nodata
filtered = np.where(np.isin(data, keep_classes), data, nodata)

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(filtered, 1)
```

---

## 8. Coverage Analysis

Used in `process_data_v2.py` after CDL reprojection to measure per-class pixel fraction:

```python
import numpy as np
import rasterio

with rasterio.open(cdl_reprojected_path) as src:
    data = src.read(1).flatten()    # (H*W,)

total   = data.size
ids, counts = np.unique(data, return_counts=True)
coverage = {int(cls): cnt / total for cls, cnt in zip(ids, counts)}
# e.g. {3: 0.066, 75: 0.115, ...}
```

---

## 9. Reading Specific Bands (S2 multi-band stack)

S2 processed files have 11 bands stacked (B1–B12 excl. B9/B10). Reading by band index:

```python
S2_BAND_NAMES = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]

with rasterio.open("S2H_2022_2022_07_30_processed.tif") as src:
    # Read specific bands by 1-based index
    rgb = src.read([4, 3, 2])       # B4, B3, B2 → (3, H, W)

    # Read selected band indices (0-based list → 1-based for rasterio)
    band_indices = [2, 4, 7]        # 0-based positions in S2_BAND_NAMES
    data = src.read([i + 1 for i in band_indices])   # (n_bands, H, W)
```

---

## 10. Corrupt File Check

Pattern used in `utils/check_corrupt_files.py` and `train_segmentation.py`:

```python
from rasterio.windows import Window

def is_corrupt(path):
    try:
        with rasterio.open(path) as src:
            src.read(1, window=Window(0, 0,
                         min(256, src.width),
                         min(256, src.height)))
        return False
    except Exception:
        return True
```

---

## Quick Reference

```python
import rasterio
from rasterio.merge  import merge as rio_merge
from rasterio.warp   import reproject, Resampling
from rasterio.windows import Window

# Open + inspect
with rasterio.open(path) as src:
    src.crs, src.transform, src.width, src.height, src.count, src.nodata, src.bounds

# Read all / one band / windowed
src.read()          # (bands, H, W)
src.read(1)         # (H, W)
src.read(window=Window(col, row, w, h))

# Write
with rasterio.open(path, "w", **profile) as dst:
    dst.write(array, 1)

# Merge tiles
mosaic, transform = rio_merge(srcs)

# Reproject
reproject(source=rasterio.band(src, 1), destination=dst_arr,
          src_transform=..., src_crs=...,
          dst_transform=..., dst_crs=...,
          resampling=Resampling.nearest)
```
