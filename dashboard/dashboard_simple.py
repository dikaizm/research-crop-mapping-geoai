
import os
import leafmap
from pathlib import Path

# Colors for each class index (model ID)
CROP_COLORS = [
    "#000000", # 0: Background
    "#00a9e6", # 1: Rice (3)
    "#ffff00", # 2: Sunflower (6)
    "#a87000", # 3: Winter Wheat (24)
    "#ffa8e3", # 4: Alfalfa (36)
    "#a5f58d", # 5: Other Hay (37)
    "#f5a27a", # 6: Tomatoes (54)
    "#704489", # 7: Grapes (69)
    "#00a884", # 8: Almonds (75)
    "#ebd6b0", # 9: Walnuts (76)
    "#ff91ab", # 10: Plums (220)
    "#bfbf7a", # 11: Fallow (61)
]

LABEL_LIST = ["Background", "Rice", "Sunflower", "Winter Wheat", "Alfalfa", "Other Hay", "Tomatoes", "Grapes", "Almonds", "Walnuts", "Plums", "Fallow"]

# Data paths (relative to dashboard/ folder)
DASHBOARD_ROOT = Path(__file__).parent
DATA_DIR = DASHBOARD_ROOT / "data"
PROJECT_ROOT = DASHBOARD_ROOT.parent

RGB_TIF = DATA_DIR / "demo_s2_rgb.tif"
PRED_TIF = DATA_DIR / "demo_prediction_exp_C.tif"
GT_TIF = PROJECT_ROOT / "data" / "processed" / "cdl" / "cdl_2022_study_area_filtered.tif"

def main():
    print("Creating Interactive Dashboard...")
    
    m = leafmap.Map(center=[39.15, -121.8], zoom=10)

    # Add Sentinel-2 RGB
    if RGB_TIF.exists():
        m.add_raster(str(RGB_TIF), layer_name="Sentinel-2 RGB (2022-01-01)")
    
    # Add Ground Truth (CDL)
    if GT_TIF.exists():
        m.add_raster(str(GT_TIF), palette=CROP_COLORS, layer_name="Ground Truth (CDL 2022)", opacity=0.7)
    
    # Add Prediction
    if PRED_TIF.exists():
        m.add_raster(str(PRED_TIF), palette=CROP_COLORS, layer_name="Prediction (Exp C - SegFormer)", opacity=0.7)

    m.add_legend(title="Crop Classes", colors=CROP_COLORS, labels=LABEL_LIST)
    
    output_html = DASHBOARD_ROOT / "crop_mapping_dashboard.html"
    m.to_html(str(output_html))
    print(f"Dashboard saved to {output_html}")

    # Create a split map dashboard as well
    m_split = leafmap.Map(center=[39.15, -121.8], zoom=12)
    m_split.split_map(
        left_layer=str(GT_TIF), 
        right_layer=str(PRED_TIF), 
        left_args={"palette": CROP_COLORS, "layer_name": "Ground Truth"}, 
        right_args={"palette": CROP_COLORS, "layer_name": "Prediction"}
    )
    split_html = DASHBOARD_ROOT / "crop_mapping_split_view.html"
    m_split.to_html(str(split_html))
    print(f"Split-view dashboard saved to {split_html}")

if __name__ == "__main__":
    main()
