
import streamlit as st
import leafmap.foliumap as leafmap
import os
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Crop Mapping Research Demo")

st.title("🌾 Crop Mapping Research Demo")
st.markdown("""
### 3-Stage Spectro-Temporal Band Selection for Crop Mapping
This dashboard showcases the result of **Experiment C (Proposed Method)** vs. the **USDA CDL Ground Truth** 
in Sacramento Valley, CA. The model used here is **SegFormer (MiT-B2)**.
""")

# --- Data Paths (relative to dashboard/ folder) ---
DASHBOARD_ROOT = Path(__file__).parent
DATA_DIR = DASHBOARD_ROOT / "data"
PROJECT_ROOT = DASHBOARD_ROOT.parent

RGB_TIF = DATA_DIR / "demo_s2_rgb.tif"
PRED_TIF = DATA_DIR / "demo_prediction_exp_C.tif"
# GT stays in project data folder to avoid duplication
GT_TIF = PROJECT_ROOT / "data" / "processed" / "cdl" / "cdl_2022_study_area_filtered.tif"

# --- Constants ---
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

# --- Sidebar ---
st.sidebar.title("Dashboard Controls")
show_rgb = st.sidebar.checkbox("Show Sentinel-2 RGB", True)
show_gt = st.sidebar.checkbox("Show Ground Truth (CDL)", True)
show_pred = st.sidebar.checkbox("Show Model Prediction (Exp C)", True)
opacity = st.sidebar.slider("Layer Opacity", 0.0, 1.0, 0.7)

st.sidebar.markdown("---")
st.sidebar.info("""
**Legend:**
- 🌾 Rice (Blue)
- 🌻 Sunflower (Yellow)
- 🌾 Winter Wheat (Brown)
- 🍀 Alfalfa (Pink)
- 🍅 Tomatoes (Orange)
- 🌳 Almonds/Walnuts (Green/Beige)
""")

# --- Main Map ---
col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("Interactive Map View")
    m = leafmap.Map(center=[39.15, -121.8], zoom=11)
    
    if show_rgb and RGB_TIF.exists():
        m.add_raster(str(RGB_TIF), layer_name="Sentinel-2 RGB")
    
    if show_gt and GT_TIF.exists():
        m.add_raster(str(GT_TIF), palette=CROP_COLORS, layer_name="Ground Truth (CDL)", opacity=opacity)
        
    if show_pred and PRED_TIF.exists():
        m.add_raster(str(PRED_TIF), palette=CROP_COLORS, layer_name="Prediction (SegFormer)", opacity=opacity)
    
    m.add_legend(title="Crop Classes", colors=CROP_COLORS, labels=LABEL_LIST)
    m.to_streamlit(height=600)

with col2:
    st.markdown("### Metrics")
    st.metric("Model Architecture", "SegFormer (MiT-B2)")
    st.metric("Input Config", "Experiment C (Band Selected)")
    st.metric("Target Region", "Sacramento Valley, CA")

# --- Split Map Comparison ---
st.markdown("---")
st.subheader("Comparison Slider: Ground Truth vs. Prediction")
st.write("Slide the vertical bar to compare the USDA CDL labels (Left) with our model's classification (Right).")

m_split = leafmap.Map(center=[39.15, -121.8], zoom=12)
m_split.split_map(
    left_layer=str(GT_TIF), 
    right_layer=str(PRED_TIF), 
    left_args={"palette": CROP_COLORS, "layer_name": "Ground Truth"}, 
    right_args={"palette": CROP_COLORS, "layer_name": "Prediction"}
)
m_split.to_streamlit(height=600)

st.markdown("""
---
*Demo created for Thesis Defense. Data sourced from Sentinel-2 SR and USDA CDL.*
""")
