
import streamlit as st
import leafmap.foliumap as leafmap
import os
import torch
import numpy as np
import rasterio
import time
from pathlib import Path
from PIL import Image

# --- Project Setup ---
DASHBOARD_ROOT = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_ROOT.parent
import sys
sys.path.append(str(PROJECT_ROOT))

from crop_mapping_pipeline.models import build_segformer
from crop_mapping_pipeline.config import NUM_CLASSES, S2_NODATA

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Crop Mapping Live Demo")

# --- Constants & Styling ---
CROP_COLORS = ["#000000", "#00a9e6", "#ffff00", "#a87000", "#ffa8e3", "#a5f58d", "#f5a27a", "#704489", "#00a884", "#ebd6b0", "#ff91ab", "#bfbf7a"]
LABEL_LIST = ["Background", "Rice", "Sunflower", "Winter Wheat", "Alfalfa", "Other Hay", "Tomatoes", "Grapes", "Almonds", "Walnuts", "Plums", "Fallow"]

SAMPLE_SITES = {
    "Rice Fields (Colusa)": {"y": 1500, "x": 2500, "desc": "Large scale rice cultivation area."},
    "Almond Orchards (Arbuckle)": {"y": 3500, "x": 800, "desc": "Dense nut tree orchards."},
    "Mixed Crops (Woodland)": {"y": 4200, "x": 1500, "desc": "Complex mosaic of tomatoes and hay."},
    "Walnut Groves": {"y": 2200, "x": 3200, "desc": "High-value permanent crops."}
}

EXP_CONFIGS = {
    "Exp A": {"name": "Single-date", "ckpt": "exp_A_segformer"},
    "Exp B": {"name": "Multi-temporal Naive", "ckpt": "exp_B_segformer"},
    "Exp C": {"name": "Proposed (Selected)", "ckpt": "exp_C_segformer"}
}

# --- Cache Model Loading ---
@st.cache_resource
def load_crop_model(exp_key):
    ckpt_path = PROJECT_ROOT / "ml_models" / "stage3_smallscale" / EXP_CONFIGS[exp_key]['ckpt'] / "best_model.pth"
    if not ckpt_path.exists():
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model = build_segformer(encoder_name="mit_b2", in_channels=checkpoint['in_channels'], num_classes=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['band_indices']

# --- Sidebar ---
st.sidebar.title("🚀 CropAI Control")
mode = st.sidebar.radio("Display Mode", ["Full Map Dashboard", "Live Patch Inference"])

# --- MODE 1: FULL MAP DASHBOARD ---
if mode == "Full Map Dashboard":
    st.title("🌾 Crop Mapping Research Dashboard")
    st.markdown("### Regional Scale Visualization")
    
    # (Existing full map logic updated for the new structure)
    exp_key = st.sidebar.selectbox("Experiment Configuration", ["Exp A", "Exp B", "Exp C"], index=2)
    opacity = st.sidebar.slider("Layer Opacity", 0.0, 1.0, 0.7)
    
    m = leafmap.Map(center=[39.15, -121.8], zoom=11)
    rgb_path = DASHBOARD_ROOT / "data" / "demo_s2_rgb.tif"
    pred_path = DASHBOARD_ROOT / "data" / f"demo_prediction_{exp_key.replace(' ', '_').lower()}.tif"
    gt_path = PROJECT_ROOT / "data" / "processed" / "cdl" / "cdl_2022_study_area_filtered.tif"
    
    if rgb_path.exists(): m.add_raster(str(rgb_path), layer_name="Sentinel-2 RGB")
    if gt_path.exists(): m.add_raster(str(gt_path), palette=CROP_COLORS, layer_name="Ground Truth", opacity=opacity)
    if pred_path.exists(): m.add_raster(str(pred_path), palette=CROP_COLORS, layer_name=f"{exp_key} Prediction", opacity=opacity)
    
    m.add_legend(title="Crop Classes", colors=CROP_COLORS, labels=LABEL_LIST)
    m.to_streamlit(height=700)

# --- MODE 2: LIVE PATCH INFERENCE ---
else:
    st.title("🔬 Live Inference Flow")
    st.markdown("Observe how the model processes a 256x256 patch from the multi-temporal stack.")

    col_ctrl, col_flow = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Settings")
        site_name = st.selectbox("Select Study Site", list(SAMPLE_SITES.keys()))
        exp_key = st.selectbox("Model Version", ["Exp A", "Exp B", "Exp C"], index=2)
        run_btn = st.button("▶️ Run Inference", use_container_state=True)
        st.info(SAMPLE_SITES[site_name]['desc'])

    if run_btn:
        model_data = load_crop_model(exp_key)
        if not model_data:
            st.error("Model checkpoint not found!")
        else:
            model, band_indices = model_data
            site = SAMPLE_SITES[site_name]
            
            # 1. Load Data Flow
            with st.status("Fetching Multi-temporal Data...", expanded=True) as status:
                st.write("Loading Sentinel-2 spectral-temporal stack...")
                s2_files = sorted(list((PROJECT_ROOT / "data" / "processed" / "s2").glob("*.tif")))
                
                # Extract patch
                patch_size = 256
                arrays = []
                for p in s2_files:
                    with rasterio.open(p) as src:
                        window = rasterio.windows.Window(site['x'], site['y'], patch_size, patch_size)
                        arr = src.read(window=window).astype(np.float32)
                        arr[arr == S2_NODATA] = 0.0
                        arrays.append(arr)
                
                full_stack = np.concatenate(arrays, axis=0)
                input_tensor = torch.from_numpy(full_stack[band_indices]).unsqueeze(0)
                
                # Show Input Visuals
                st.write(f"Input Shape: `{input_tensor.shape}` ({len(band_indices)} selected bands)")
                
                st.markdown("**Step 1: Spectro-Temporal Input**")
                # Show RGB from 3 different dates to show "Temporal" change
                t_cols = st.columns(3)
                for i, idx in enumerate([0, 1, 2]):
                    # B4, B3, B2 for each file
                    rgb = arrays[idx][[3, 2, 1], :, :]
                    rgb = np.clip(rgb, 0, 3000) / 3000
                    t_cols[i].image(rgb.transpose(1, 2, 0), caption=f"Date {i+1}", use_column_width=True)
                
                # 2. Model Processing
                st.write("Forward pass through SegFormer (MiT-B2) encoder-decoder...")
                time.sleep(0.5) # simulate processing for demo effect
                with torch.no_grad():
                    output = model(input_tensor)
                    prediction = output.argmax(dim=1).squeeze().numpy()
                
                # 3. Ground Truth
                with rasterio.open(PROJECT_ROOT / "data" / "processed" / "cdl" / "cdl_2022_study_area_filtered.tif") as src:
                    window = rasterio.windows.Window(site['x'], site['y'], patch_size, patch_size)
                    gt_patch = src.read(1, window=window)
                
                status.update(label="Inference Complete!", state="complete", expanded=False)

            # --- Results Visualization ---
            st.markdown("---")
            st.subheader("Inference Result")
            res_col1, res_col2 = st.columns(2)
            
            # Helper to colorize prediction
            def colorize(array):
                h, w = array.shape
                img = np.zeros((h, w, 3), dtype=np.uint8)
                for i, color in enumerate(CROP_COLORS):
                    c = [int(color[i:i+2], 16) for i in (1, 3, 5)]
                    img[array == i] = c
                return img

            res_col1.image(colorize(gt_patch), caption="Ground Truth (USDA CDL)", use_column_width=True)
            res_col2.image(colorize(prediction), caption=f"Model Prediction ({exp_key})", use_column_width=True)
            
            # Show Legend
            st.markdown("**Legend**")
            legend_html = "".join([f'<span style="background-color:{CROP_COLORS[i]}; padding: 2px 8px; margin-right: 5px; border-radius: 3px; border: 1px solid #444;">{LABEL_LIST[i]}</span>' for i in range(1, 12)])
            st.markdown(legend_html, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Crop Mapping Thesis Research © 2026")
