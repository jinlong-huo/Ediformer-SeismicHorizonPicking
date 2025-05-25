# 3D Seismic Attribute Visualization Tool
# 
# Quick Start
# 1. Prepare your data:
#    - Place all .npy files in one folder
#    - Required files: F3_seismic.npy, F3_crop_horizon_freq.npy, F3_predict_MCDL_crossline.npy, 
#      F3_crop_horizon_phase.npy, F3_RMSAmp.npy, test_label_no_ohe.npy
#
# 2. Run the script:
#    ```bash
#    python utils/seis3dvis.py --mode train --attribute seismic
#    ```
#    Or edit the default parameters in the code
#
# Output
# - Interactive 3D visualization with Mayavi
# - Six orthogonal slice planes showing attribute data
# - Visualize any of the following attributes:
#   - seismic: original amplitude data
#   - frequency: instantaneous frequency
#   - phase: instantaneous phase
#   - dip: structural dip attribute
#   - rms: RMS amplitude
#   - label: horizon/class labels
# Use mouse to rotate, zoom and interact with the 3D volume.

# Please pay attention to the following:
# - Ensure you have the required libraries installed: mayavi, numpy, argparse

import numpy as np
import argparse

from mayavi import mlab

# seismic = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_seismic.npy').reshape((-1, 951, 288))
seismic = np.load(r"F:\Facies\FaciesData\NewZealand\data_train.npz")
seismic = seismic['data']


frequency = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_crop_horizon_freq.npy').reshape((-1, 951, 288))
phase = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_crop_horizon_phase.npy').reshape((-1, 951, 288))
rms = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_RMSAmp.npy').reshape((-1, 951, 288))
dip = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\F3_predict_MCDL_crossline.npy')
dip = np.swapaxes(dip, -1, 1)
label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy').reshape((-1, 951, 288))

# seismic_train = seismic[:188, 200:488, :]
seismic_train = seismic

frequency_train = frequency[:188, 200:488, :]
phase_train = phase[:188, 200:488, :]
rms_train = rms[:188, 200:488, :]
dip_train = dip[:188, 200:488, :]
label_train = label[:188, 200:488, :]

seismic_validation = seismic[300:488, 600:888, :]
frequency_validation = frequency[300:488, 600:888, :]
phase_validation = phase[300:488, 600:888, :]
rms_validation = rms[300:488, 600:888, :]
dip_validation = dip[300:488, 600:888, :]
label_validation = label[300:488, 600:888, :]

data_list_train = [seismic_train, frequency_train, phase_train, dip_train, rms_train, label_train]
data_list_validation = [seismic_validation, frequency_validation, phase_validation,
                        dip_validation, rms_validation, label_validation]


def plot(data, cm):
    fig = mlab.figure(bgcolor=(1, 1, 1))  # set white bg

    x_widget = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                colormap=cm, plane_orientation='x_axes')
    y_widget = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                colormap=cm, plane_orientation='y_axes')
    z_widget = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                colormap=cm, plane_orientation='z_axes')

    # Create additional image plane widgets for opposite surfaces
    x_widget_opposite = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                         colormap=cm, plane_orientation='x_axes',
                                                         slice_index=data.shape[0] - 1)
    y_widget_opposite = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                         colormap=cm, plane_orientation='y_axes',
                                                         slice_index=data.shape[1] - 1)
    z_widget_opposite = mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                                                         colormap=cm, plane_orientation='z_axes',
                                                         slice_index=data.shape[2] - 1)

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
    mlab.show()
    return


def generate_figure(mode, attribute):
    colormap_list = ['seismic', 'rainbow', 'Blues', 'coolwarm']

    if mode == 'train':
        if attribute == 'seismic':
            data = data_list_train[0]
            cm = colormap_list[0]

            plot(data, cm)
        elif attribute == 'frequency':
            data = data_list_train[1]
            cm = colormap_list[3]
            plot(data, cm)
        elif attribute == 'phase':
            cm = colormap_list[2]
            data = data_list_train[2]
            plot(data, cm)
        elif attribute == 'dip':
            cm = colormap_list[3]
            data = data_list_train[3]
            plot(data, cm)

        elif attribute == 'rms':
            cm = colormap_list[3]
            data = data_list_train[4]
            plot(data, cm)

        elif attribute == 'label':
            cm = colormap_list[1]
            data = data_list_train[5]
            plot(data, cm)


    elif mode == 'validation':

        if attribute == 'seismic':
            data = data_list_validation[0]
            cm = colormap_list[0]
            plot(data, cm)
        elif attribute == 'frequency':
            data = data_list_validation[1]
            cm = colormap_list[3]
            plot(data, cm)
        elif attribute == 'phase':
            cm = colormap_list[2]
            data = data_list_validation[2]
            plot(data, cm)
        elif attribute == 'dip':
            cm = colormap_list[3]
            data = data_list_validation[3]
            plot(data, cm)

        elif attribute == 'rms':
            cm = colormap_list[3]
            data = data_list_validation[4]
            plot(data, cm)

        elif attribute == 'label':
            cm = colormap_list[1]
            data = data_list_validation[5]
            plot(data, cm)


def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train or test a neural network')
    parser.add_argument('--mode', type=str, default=False, help='mode for train or validation')
    parser.add_argument('--attribute', type=str, default=False, help='train the network')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.mode = 'train'
    # args.mode = 'validation'
    args.attribute = 'seismic'
    # args.attribute = 'frequency'
    # args.attribute = 'dip'
    # args.attribute = 'phase'
    # args.attribute = 'rms'
    # args.attribute = 'label'

    generate_figure(args.mode, args.attribute)
