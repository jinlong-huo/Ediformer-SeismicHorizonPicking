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
horizon_layer = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_seismic.npy')
horizon_layer = horizon_layer.reshape(-1,951,288)
horizon_layer = np.swapaxes((horizon_layer),-1,1)
horizon_layer = horizon_layer[300]

horizon_layer = horizon_layer[:, 10:941]

# background = np.load(r'D:\Pycharm Projects\Pytorch_Template\seg_trace_predictions.npy')
# background = np.load(r'D:\Pycharm Projects\Pytorch_Template\dformer_trace_predictions.npy')
# background = np.load(r'D:\Pycharm Projects\Pytorch_Template\seg_patch_predictions.npy')

# background = np.load('dformer_best_patch_reshape_back.npy')
background = np.load(r'D:\Pycharm Projects\Horizon_Picking\9_20_patch_reshape_back.npy')
# background = np.load('dformer_seis_98_reshape_back.npy')

background = background.reshape(-1, 288, 944)
background = np.swapaxes(background,-1,1)

print(background.shape)
background = np.swapaxes((background),-1,1)
background = background[0]
background = background[:, 10: 941]

fig = plt.figure(figsize=(38.04, 11.52))
plt.imshow(horizon_layer, cmap='gray')

ax = plt.gca()
# ax.invert_yaxis()

plt.contour(background, linewidths=6, cmap='winter')

plt.ylabel('inline')
plt.xlabel('Time')
# plt.savefig(r'C:\Users\hjl15\Desktop\overlap.png',dpi=500)
plt.show()

