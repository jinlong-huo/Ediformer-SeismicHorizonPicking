# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# patch yong
# X = np.arange(0,288,1)
# Y = np.arange(0,960,1)
#
# TestData1 = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_crop_horizon_freq.npy') # 测试用数据和标签
# TestData1 = TestData1.reshape(-1, 288, 951)  # reshape 成为 三维数据
# TestData1 = TestData1[np.newaxis, :]  # 为了使用patch操作扩充一维数据
# TestData1 = torch.tensor(TestData1)
# kc, kh, kw = 1, 288, 64  # kernel size
# dc, dh, dw = 1, 64, 64  # stride
# TestData1 = F.pad(TestData1, [TestData1.size(3) % kw // 100, TestData1.size(2) % kw // 2,
#                               TestData1.size(2) % kh // 100, TestData1.size(1) % kh // 2,
#                               TestData1.size(1) % kc // 100, TestData1.size(0) % kc // 2])
# TestData1 = TestData1.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
# unfold_shape = TestData1.size()
# patches = TestData1.contiguous().view(-1, kc, kh, kw)
#
#
# patches_2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\1_25_phase_test_label_for_fusion.npy')
# print(patches.shape)
# patches_2 = torch.tensor(patches_2)
# patches_orig = patches_2.view(unfold_shape)
# output_c = unfold_shape[1] * unfold_shape[4]
# output_h = unfold_shape[2] * unfold_shape[5]
# output_w = unfold_shape[3] * unfold_shape[6]
# patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
# patches_orig = patches_orig.reshape(-1, output_c, output_h, output_w)
# figure = np.reshape(patches_orig, (-1, 288, 960))  # 601 951 288
# figure = figure[:,:,:951]


import numpy as np
import matplotlib.pyplot as plt
background = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_seismic.npy')
background = background.reshape(-1,951,288)
background = np.swapaxes((background),-1,1)
background = background[300]
# background = background[:, 10:941]
print(background.shape)


# horizon_layer = np.load(r'D:\Pycharm Projects\Pytorch_Template\dformer_trace_predictions.npy')                 # SegFormer trace 130
horizon_layer = np.load(r'D:\Pycharm Projects\Pytorch_Template\savgol_dformer_trace.npy')                 # SegFormer trace 130
# horizon_layer = np.load(r"D:\Pycharm Projects\Horizon_Picking\seg_patch_reshape_back.npy")                     # SegFormer patch 130
# horizon_layer = np.load(r"D:\Pycharm Projects\Pytorch_Template\EDIFormer_patch_pred_label_for_savgol.npy")     # DIFormer patch 130
# horizon_layer = np.load(r"D:\Pycharm Projects\Pytorch_Template\DIFormer_pahse_patch_pred_label.npy")           # DIFormer phase patch 130

# horizon_layer = np.load(r"D:\Pycharm Projects\Pytorch_Template\10_2_seismic_predicted_label_reshape_back.npy")  # DIFormer phase patch 130

# horizon_layer = np.load(r"D:\Pycharm Projects\Pytorch_Template\10_3_seg_patch_pred_label_for_savgol.npy")      # EDIFormer patch 130
# horizon_layer = np.load(r"D:\Pycharm Projects\Pytorch_Template\dformer_trace_pred_label_for_savgol.npy")       # DIFormer trace 130 （932， 587， 288）
# horizon_layer = np.load(r"D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy")       # DIFormer trace 130


# horizon_layer = horizon_layer.reshape(-1, 288, 932)
# horizon_layer = horizon_layer.reshape(-1, 288, 944)
# horizon_layer = horizon_layer.reshape(-1, 288, 928)
# horizon_layer = horizon_layer.reshape(-1, 283, 927)
# horizon_layer = horizon_layer.reshape(-1, 951, 288)


# horizon_layer = horizon_layer[:, 10: 941]
# fig = plt.figure(figsize=(38.04, 11.52))
# # plt.imshow(background, cmap='gray')
# # ax = plt.gca()
#
# horizon_layer = horizon_layer.reshape(-1, 951, 288)
# horizon_layer = np.swapaxes(horizon_layer, -1,1)
horizon_layer = horizon_layer[130]

# plt.colorbar()
# ax.invert_yaxis()
# plt.contour(horizon_layer, linewidths=4, cmap='winter')
# plt.ylabel('Time', fontsize=25)  # Increase the font size
# plt.xlabel('Inline', fontsize=25)  # Increase the font size
# plt.xticks(fontsize=25)  # Adjust the tick labels font size
# plt.yticks(fontsize=25)  # Adjust the tick labels font size

# Automatically adjust the layout to prevent overlap
plt.tight_layout()

# plt.savefig(r'C:\Users\hjl15\Desktop\SegFormer_patch.png', dpi=500)
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have label numbers for each layer in a list called 'layer_labels'
layer_labels = [0, 1, 2, 3, 4, 5, 6]  # Replace this with your actual label numbers

background = horizon_layer

# Create a color map for different layers
layer_cmap = plt.get_cmap('tab10', len(layer_labels))

fig = plt.figure(figsize=(38.04, 11.52))
plt.imshow(background, cmap='YlGnBu')#YlGnBu GnBu coolwarm 0.1
ax = plt.gca()


for label in layer_labels:

    horizon_layer = background  # Replace with actual layer data

    # Filter the specific layer
    layer_mask = (horizon_layer == label)
    layer_data = np.where(layer_mask, horizon_layer, np.nan)

    # Plot the layer with the specified color
    plt.imshow(layer_data, cmap=layer_cmap, alpha=0.1, vmin=0,
               vmax=len(layer_labels) - 1)  # Adjust alpha for transparency 透明度

plt.show()
#%%
# fig = plt.figure()
#
# # Plot the seismic wiggle trace
# plt.fill_betweenx(np.arange(len(trace_normalized)), trace_normalized, 0,
#                   trace_normalized > 0, color='k', linewidth=1)
#
# # Set the y-axis direction
# plt.gca().invert_yaxis()
#
# # Set labels and title
# plt.xlabel('Amplitude')
# plt.ylabel('Time (s)')
# plt.title('Seismic Wiggle Trace')
#
# # Show the plot
# plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

# Create a color map using YlGnBu
cmap = plt.get_cmap('YlGnBu')

# Create a gradient of values to visualize the colormap
gradient = np.linspace(0, 1, 7).reshape(1, -1)

# Create a figure and axis to display the colormap
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Display the colormap using a colorbar
cax = ax.matshow(gradient, cmap=cmap, aspect='auto')
ax.set_xticks([])  # Hide x-axis ticks
plt.show()
