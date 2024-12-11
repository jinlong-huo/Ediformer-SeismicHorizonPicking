import numpy as np


from mayavi import mlab

# seismic_data = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
# seismic_data = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_seismic.npy')
# seismic_data = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_rop_horizon_freq.npy')
# seismic_data = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_crop_horizon_phase.npy')
seismic_data = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_predict_MCDL_crossline.npy')
# seismic_data = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_RMSAmp.npy')

seismic_data = seismic_data.reshape(-1, 951, 288)
seismic_data = seismic_data[10:590,10:940,:288]


# Generate some sample data
data = seismic_data

# Create the Mayavi figure
fig = mlab.figure()

# Create the grid of image plane widgets for all slices
x_widget = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                            colormap='Blues', plane_orientation='x_axes')
y_widget = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                            colormap='Blues', plane_orientation='y_axes')
z_widget = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                            colormap='Blues', plane_orientation='z_axes')

# Create additional image plane widgets for opposite surfaces
x_widget_opposite = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                     colormap='Blues', plane_orientation='x_axes',
                                                     slice_index=data.shape[0]-1)
y_widget_opposite = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                     colormap='Blues', plane_orientation='y_axes',
                                                     slice_index=data.shape[1]-1)
z_widget_opposite = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                     colormap='Blues', plane_orientation='z_axes',
                                                     slice_index=data.shape[2]-1)

# Adjust the colormap range for all image plane widgets
data_min = data.min()
data_max = data.max()

x_widget.module_manager.scalar_lut_manager.data_range = (data_min, data_max)
y_widget.module_manager.scalar_lut_manager.data_range = (data_min, data_max)
z_widget.module_manager.scalar_lut_manager.data_range = (data_min, data_max)

x_widget_opposite.module_manager.scalar_lut_manager.data_range = (data_min, data_max)
y_widget_opposite.module_manager.scalar_lut_manager.data_range = (data_min, data_max)
z_widget_opposite.module_manager.scalar_lut_manager.data_range = (data_min, data_max)

# Adjust the camera clipping range to display the complete surface
fig.scene.camera.clipping_range = [data_min, data_max]

# colorbar = mlab.colorbar(orientation='vertical', label_fmt='%.1f', title='Colorbar Title')
# colorbar.scalar_bar_representation.position = [0.85, 0.1]  # Adjust position
# colorbar.scalar_bar_representation.position2 = [0.1, 0.8]  # Adjust size

# z_label = mlab.zlabel('Z')
# z_label.property.orientation = (0, 0, -90)  # Rotate counterclockwise

# Display the plot
# mlab.axes(xlabel='Xline', ylabel='Inline')
# mlab.outline()

mlab.show()


