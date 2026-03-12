
import os
import sys
import torch
import numpy as np
import rasterio
from pathlib import Path

# Add project root to sys.path
DASHBOARD_ROOT = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from crop_mapping_pipeline.models import build_segformer
from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, KEEP_CLASSES, REMAP_LUT, 
    NUM_CLASSES, S2_NODATA, PATCH_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

def run_inference(model, s2_paths, band_indices, patch_size=256, stride=256):
    arrays = []
    profile = None
    for p in s2_paths:
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
            if profile is None:
                profile = dict(src.profile)
        arr[arr == S2_NODATA] = 0.0
        arrays.append(arr)
    
    stacked = np.concatenate(arrays, axis=0)
    selected = stacked[band_indices]
    K, H, W = selected.shape
    
    pred_map = np.zeros((H, W), dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                ph = min(patch_size, H - y)
                pw = min(patch_size, W - x)
                patch = selected[:, y:y+ph, x:x+pw]
                
                if ph < patch_size or pw < patch_size:
                    padded = np.zeros((K, patch_size, patch_size), dtype=np.float32)
                    padded[:, :ph, :pw] = patch
                    patch = padded
                
                t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                out = model(t).argmax(dim=1).squeeze().cpu().numpy()
                pred_map[y:y+ph, x:x+pw] = out[:ph, :pw]
    
    return pred_map, profile

def main():
    # Use root data folder
    S2_ROOT_DIR = PROJECT_ROOT / "data" / "processed" / "s2"
    s2_files = sorted(list(S2_ROOT_DIR.glob("*.tif")))
    
    if not s2_files:
        print(f"No S2 data found in {S2_ROOT_DIR}")
        return

    # Load model
    ckpt_path = PROJECT_ROOT / "ml_models" / "stage3_smallscale" / "exp_C_segformer" / "best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    in_channels = checkpoint['in_channels']
    band_indices = checkpoint['band_indices']
    
    # Checkpoint has 12 classes (background + 11 crops)
    inference_num_classes = 12
    
    model = build_segformer(
        encoder_name="mit_b2",
        in_channels=in_channels,
        num_classes=inference_num_classes
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Running inference on {len(s2_files)} files...")
    pred_map, profile = run_inference(model, s2_files, band_indices)
    
    # Save as TIF to dashboard/data
    output_path = DASHBOARD_ROOT / "data" / "demo_prediction_exp_C.tif"
    profile.update(count=1, dtype='uint8', nodata=0)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred_map, 1)
    
    print(f"Saved demo prediction to {output_path}")

    # Also save an RGB version for context
    with rasterio.open(s2_files[0]) as src:
        rgb = src.read([4, 3, 2])
        rgb_profile = dict(src.profile)
    
    rgb_output = DASHBOARD_ROOT / "data" / "demo_s2_rgb.tif"
    rgb_profile.update(count=3)
    # Clip and normalize for visualization
    rgb = np.clip(rgb, 0, 3000) / 3000 * 255
    with rasterio.open(rgb_output, 'w', **rgb_profile) as dst:
        dst.write(rgb.astype(np.uint8))
    print(f"Saved demo RGB to {rgb_output}")

if __name__ == "__main__":
    main()
