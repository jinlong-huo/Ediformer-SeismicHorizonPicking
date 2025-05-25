# Seismic Patch to Volume Converter
# 
# Quick Start
# 1. Prepare your data:
#    - Original seismic data file (to determine patch structure)
#    - Patched prediction file from neural network output
#
# 2. Run the script:
#    ```bash
#    python utils/unpatch.py
#    ```
#    Or edit the file paths in the code
#
# Output
# - Reconstructed volume (.npy): Prediction patches reassembled into original 3D shape
# - CSV export (.csv): Flattened 2D representation for spreadsheet analysis
# All files saved in the current directory with timestamp.

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import csv
import imageio

# First you need the original data to (of how you patch to obtain the unfold shape(the patch size))
x = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy') # have something wrong at head

x = x.reshape(-1, 951, 288)
x = np.swapaxes((x),-1,1)

x = torch.tensor(x)
x = x[np.newaxis, :]
kc, kh, kw = 1, 288, 16  # kernel size
dc, dh, dw = 1, 288, 16  # stride

# Pad to multiples of 32
x = F.pad(x, [x.size(3) % kw // 2, x.size(3) % kw // 2,
              x.size(2) % kh // 2, x.size(2) % kh // 2,
              x.size(1) % kc // 2, x.size(1) % kc // 2])

patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
unfold_shape = patches.size()
print(patches.shape)

# change here to select your data and transform to the original shape

# patches = np.load(r'D:\Pycharm Projects\Pytorch_Template\dformer_predict_label_2023-09-08-12-55.npy')
# patches = np.load(r'D:\Pycharm Projects\Pytorch_Template\_pred_label.npy')

patches = np.load(r"D:\Pycharm Projects\Pytorch_Template\dformer_predict_label_2023-09-08-12-55.npy")
patches = torch.tensor(patches)
# patches = torch.argmax(patches,dim=1)
patches = patches.reshape(-1, 1, 288, 16)
print(patches.shape)
patches = torch.tensor(patches)
patches_orig = patches.view(unfold_shape)
output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]
patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.view(1, output_c, output_h, output_w)
patches_orig = np.squeeze(patches_orig)
np.save('9_8_reshape_back.npy', patches_orig)
# np.save('10_2_unet_prediction_reshape_back.npy', patches_orig)

print(patches_orig.shape)
patches_orig = patches_orig.reshape(-1, 288, 944)

#%%
#------------------------------ save txt file from npy----------------------------------
# save a total file may cost XX 3 minutes
# np.save('figure.npy', figure)

patches_orig = np.load('10_2_unet_prediction_reshape_back.npy') # torch.Size([601, 288, 944])
print(patches_orig.shape)
figure = np.swapaxes((patches_orig), -1, 1)
print(figure.shape)
figure = figure.reshape(-1, 288)
# Save the reshaped data as a CSV file
np.savetxt('10_2_unet_prediction_reshape_back.csv', figure, delimiter=',')
print(figure.shape)
