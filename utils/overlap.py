import PIL.Image as Image
from PIL import ImageEnhance
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
# patch yong
X = np.arange(0,288,1)
Y = np.arange(0,960,1)

TestData1 = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_crop_horizon_freq.npy') # 测试用数据和标签
TestData1 = TestData1.reshape(-1, 288, 951)  # reshape 成为 三维数据
TestData1 = TestData1[np.newaxis, :]  # 为了使用patch操作扩充一维数据
TestData1 = torch.tensor(TestData1)
kc, kh, kw = 1, 288, 64  # kernel size
dc, dh, dw = 1, 64, 64  # stride
TestData1 = F.pad(TestData1, [TestData1.size(3) % kw // 100, TestData1.size(2) % kw // 2,
                              TestData1.size(2) % kh // 100, TestData1.size(1) % kh // 2,
                              TestData1.size(1) % kc // 100, TestData1.size(0) % kc // 2])
TestData1 = TestData1.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
unfold_shape = TestData1.size()
patches = TestData1.contiguous().view(-1, kc, kh, kw)


patches_2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\1_25_phase_test_label_for_fusion.npy')
print(patches.shape)
patches_2 = torch.tensor(patches_2)
patches_orig = patches_2.view(unfold_shape)
output_c = unfold_shape[1] * unfold_shape[4]
output_h = unfold_shape[2] * unfold_shape[5]
output_w = unfold_shape[3] * unfold_shape[6]
patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
patches_orig = patches_orig.reshape(-1, output_c, output_h, output_w)
figure = np.reshape(patches_orig, (-1, 288, 960))  # 601 951 288
figure = figure[:,:,:951]

# # dod yong
# z = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
#
# # z = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
# # z = z.reshape(-1,951,288)
# z = figure
# z = np.array(z)
# import imageio
# c=0
# for i in z:
#     c+=1
#     if c==500:
# #     # imageio.imwrite(r'D:/Pycharm Projects/Horizon_Picking/for_paper/'+str(c)+'freq.png',i)
#         fig = plt.figure(figsize=(38.04, 11.52))
#         ax = fig.add_subplot(1, 1, 1)
#         ax.invert_yaxis()
#         plt.contour(i,linewidths=4,cmap='winter')
#         ax.plot(i)
#
#         # cbar = plt.colorbar(y)
#         # cbar_ticks = np.linspace(0., 5., num=5, endpoint=True)
#         # cbar.set_ticks(cbar_ticks)
#
#         plt.xticks([])
#         plt.yticks([])
#         plt.show()

# in_data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
# in_data = in_data.reshape(-1, 951, 288)
# in_data = torch.tensor(in_data)
# in_data = in_data.permute(0, -1, 1)
# in_data = list(in_data)
# plt.xticks([])
# plt.yticks([])
#
# test_data1 = torch.tensor(in_data[191])
# print(test_data1.shape)
# fig = plt.figure(figsize=(38.04, 11.52))
# ## flip y axis first to give a plot
# ax = fig.add_subplot(1, 1, 1)
# ax.invert_yaxis()
# plt.xticks([])
# plt.yticks([])
# y = plt.imshow(test_data1,  cmap='gray')
# plt.savefig('test.png'.format(test_data1),dpi=500)
# plt.show()
#
import matplotlib.pyplot as plt

# # 重叠
# # layer1 = Image.open(r'D:\Pycharm Projects\Horizon_Picking\MCDL_inline100.png').convert('RGBA')   # 底图背景

# layer1 = Image.open(r"D:\Pycharm Projects\Horizon_Picking\for_paper\seismic.png").convert('RGBA')   # 底图背景
# layer2 = Image.open(r"D:\Pycharm Projects\Horizon_Picking\for_paper\transunet.png").convert('RGBA')    # mask
#
layer1 = Image.open(r"test.png").convert('RGBA')   # 底图背景
print(layer1.shape)
layer2 = Image.open(r"D:\Pycharm Projects\Horizon_Picking\myplot1.png").convert('RGBA')    # mask
print(layer2.shape)

r,g,b,a = layer2.split()
# opacity为透明度，范围(0,1)
opacity = 0.1
alpha = ImageEnhance.Brightness(a).enhance(opacity)
layer2.putalpha(alpha)

# 使用alpha_composite叠加，两者需相同size
final = Image.new("RGBA", layer1.size)
final = Image.alpha_composite(final, layer1)
final = Image.alpha_composite(final, layer2)
final=final.convert('RGB')
plt.savefig('test.png',dpi=700)
# final.save('2_1.jpg',dpi=1000)
final.show()




# # 使用paste叠加，无需相同大小，可调整box位置
# layer = Image.new('RGBA', layer1 .size, (0, 0, 0, 0))
# layer.paste(layer2, (100, 100))
# Image.composite(layer, layer1 , layer).convert('RGB')


#改变大小
# import cv2
# root = r'C:\Users\hjl15\Desktop\seism.png'
# crop_size = (1920, 757)
# img = cv2.imread(root)
# img_new = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
# cv2.imwrite('1_30.png', img_new)