import matplotlib.pyplot as plt
import numpy as np

figure = np.load(r'D:\Pycharm Projects\Pytorch_Template\EDIFormer_patch_pred_label_for_savgol.npy')
print(figure.shape)
figures = np.swapaxes(figure, -1,1 )
print(figures.shape)

#%%
selected_figure = figures[100]

plt.plot(selected_figure)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Horizontal Section with Label')
plt.grid(True)
plt.show()
