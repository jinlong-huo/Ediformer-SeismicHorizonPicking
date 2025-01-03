import numpy as np
import fast_glcm
import matplotlib.pyplot as plt

def main():
    pass

if __name__ == '__main__':
    main()
    data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
    data = data.reshape(-1, 288
    glcm_mean = fast_glcm.fast_glcm_mean(data[0])
    plt.imshow(glcm_mean)
    plt.tight_layout()
    plt.show()