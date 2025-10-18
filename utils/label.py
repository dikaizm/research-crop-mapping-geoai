import rasterio
import numpy as np

def label_filtering(in_path, out_path, keep_classes=[]):
    # Open raster
    with rasterio.open(in_path) as src:
        profile = src.profile
        data = src.read(1)  # read first band

    # Create a mask: keep selected classes, others become 0
    filtered = np.where(np.isin(data, keep_classes), data, 0)

    # Save filtered raster
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    print("✅ Saved filtered raster:", out_path)