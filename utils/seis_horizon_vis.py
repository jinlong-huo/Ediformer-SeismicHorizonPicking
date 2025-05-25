# Seismic Horizon Visualization Tool
# 
# Quick Start
# 1. Prepare your data:
#    - Place .npy files in a folder
#    - Required: F3_seismic.npy and horizon prediction files
#
# 2. Run the script:
#    ```bash
#    python utils/seis_horizon_vis.py
#    ```
#
# Output
# - Visualization of seismic data with overlaid horizons
# - Publication-ready figures (contour or overlay modes)

import numpy as np
import matplotlib.pyplot as plt

# Load background seismic data
background = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_seismic.npy')
background = background.reshape(-1, 951, 288)
background = np.swapaxes(background, -1, 1)
background = background[300]  # Select slice to visualize
print(f"Background shape: {background.shape}")

# Load horizon prediction data - uncomment the one you want to visualize
horizon_layer = np.load(r'D:\Pycharm Projects\Pytorch_Template\savgol_dformer_trace.npy')
# horizon_layer = np.load(r'D:\Pycharm Projects\Pytorch_Template\dformer_trace_predictions.npy')
# horizon_layer = np.load(r"D:\Pycharm Projects\Horizon_Picking\seg_patch_reshape_back.npy")
# horizon_layer = np.load(r"D:\Pycharm Projects\Pytorch_Template\EDIFormer_patch_pred_label_for_savgol.npy")

# Select slice to visualize
horizon_layer = horizon_layer[130]

# Display basic visualization
plt.figure()
plt.tight_layout()
plt.show()

# Colored overlay visualization
layer_labels = [0, 1, 2, 3, 4, 5, 6]  # Horizon layer labels
layer_cmap = plt.get_cmap('tab10', len(layer_labels))

plt.figure(figsize=(38.04, 11.52))
plt.imshow(background, cmap='YlGnBu')

# Overlay each horizon layer with transparency
for label in layer_labels:
    layer_mask = (horizon_layer == label)
    layer_data = np.where(layer_mask, horizon_layer, np.nan)
    plt.imshow(layer_data, cmap=layer_cmap, alpha=0.1, vmin=0, vmax=len(layer_labels) - 1)

plt.show()