from mayavi import mlab
import numpy as np

# # x, y = np.ogrid[0:951:951j, 0:651:651j] # x==>951, 1    y==>1, 651
# # z = figure[30]
# # pl = mlab.surf(x, y, z, warp_scale="auto",colormap='gist_earth')
# # mlab.axes(xlabel='x', ylabel='y', zlabel='z')
# # mlab.outline(pl)
# # mlab.show()
# # mlab.clf()

# figure = np.load(r'D:\Pycharm Projects\Pytorch_Template\seismic_prediction_feature_maps_predicted_label.npy')
# figure2 = np.load(r"D:\Pycharm Projects\Pytorch_Template\unet_9_28_data_for_petrel.npy")

# figure2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\9_26_unet_patch_reshape_back.npy')
# figure2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\convolved_horizons.npy')
# figure2 = np.load(r'D:\Pycharm Projects\Pytorch_Template\savgol_smoothed_data.npy')
# figure2 = np.load(r'D:\Pycharm Projects\Pytorch_Template\unet_9_28_data_for_petrel.npy')
# figure2 = np.load(r"D:\Pycharm Projects\Horizon_Picking\9_20_patch_reshape_back.npy")
# figure2 = np.load(r"D:\Pycharm Projects\Pytorch_Template\10_2_seismic_predicted_label_reshape_back.npy")
figure2 = np.load(r"D:\Pycharm Projects\Horizon_Picking\seg_patch_reshape_back.npy")
print(figure2.shape)
# figure2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\9_25_patch_reshape_back.npy')
# figure2 = np.load(r"D:\Pycharm Projects\Horizon_Picking\9_25_patch_reshape_back.npy")
# figure2 = np.load(r'D:\Pycharm Projects\Pytorch_Template\Gaussian_smoothed_data.npy')
# figure2 = np.load(r'D:\Pycharm Projects\Pytorch_Template\EMA_smoothed_data.npy')
# figure2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\9_26_unet_patch_reshape_back.npy')
# print((figure2.all()==figure1.all()))

figure = np.swapaxes(figure2, -1, 1)
x, y, z = np.ogrid[0:940:9510j, 0:590:6510j, 0:288:2880j]  # [0:601:601j, 0:951:951j, 0:288:288j]
figure = np.array(figure)
# values = figure[20:590, 10:940, :]
mlab.contour3d(figure)
mlab.axes()
mlab.show()

#%%
import numpy as np
from mayavi import mlab

# adjusted_predictions_3d = np.load(r'D:\Pycharm Projects\Pytorch_Template\unet_9_28_data_for_petrel.npy')
# adjusted_predictions_3d = np.load(r"D:\Pycharm Projects\Horizon_Picking\9_20_patch_reshape_back.npy")
adjusted_predictions_3d = np.load(r"D:\Pycharm Projects\Horizon_Picking\9_9_patch_reshape_back.npy")
adjusted_predictions_3d = np.load(r"D:\Pycharm Projects\Horizon_Picking\9_9_patch_reshape_back.npy")
# adjusted_predictions_3d = np.load(r"D:\Pycharm Projects\Horizon_Picking\9_20_patch_reshape_back.npy")

mlab.contour3d(adjusted_predictions_3d)
mlab.axes()
mlab.show()

# print(adjusted_predictions_3d.shape)

# file_path = "adjusted_predictions_3d_10_2.csv"
#
# with open(file_path, "w") as file:
#     for slice_2d in adjusted_predictions_3d:
#         np.savetxt(file, slice_2d, fmt="%d", delimiter="\t")
#         file.write("\n")  # Add a newline to separate slices
#
# print('done!')

