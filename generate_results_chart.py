import matplotlib.pyplot as plt
import numpy as np

# Data
classes = ["Rice", "Sunflower", "W. Wheat", "Alfalfa", "Other Hay", "Tomatoes", "Grapes", "Almonds", "Walnuts", "Plums"]
exp_a_sf = [21.74, 14.58, 27.13, 11.45, 4.35, 40.43, 24.58, 57.37, 35.03, 18.91]
exp_b_sf = [25.10, 17.93, 35.52, 18.48, 3.38, 35.18, 12.02, 56.43, 32.61, 15.25]
exp_c_dl = [22.59, 7.55, 10.50, 11.41, 0.57, 7.21, 9.98, 23.85, 9.84, 0.99]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, exp_a_sf, width, label='Exp A (SF - Single Date)', color='#aec7e8')
rects2 = ax.bar(x, exp_b_sf, width, label='Exp B (SF - Multi-Temp)', color='#ffbb78')
rects3 = ax.bar(x + width, exp_c_dl, width, label='Exp C (DLV3 - Proposed)', color='#2ca02c')

ax.set_ylabel('IoU (%)')
ax.set_title('Per-Crop IoU Comparison (2024 Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig('documents/thesis/figures/per_crop_iou_comparison.png', dpi=300)
print("Chart saved to documents/thesis/figures/per_crop_iou_comparison.png")
