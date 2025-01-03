import numpy as np
import fast_glcm
import matplotlib.pyplot as plt
from skimage import data

def main():
    pass

if __name__ == '__main__':
    main()
    # data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
    # data = data.reshape(-1, 951, 288)
    img =  data.camera()
    # data = np.swapaxes(data ,-1, 1)
    glcm_mean = fast_glcm.fast_glcm_dissimilarity(img)
    plt.imshow(glcm_mean)
    # plt.imshow(img)
    plt.tight_layout()
    plt.savefig('data_0_glcm_mean.png')
    plt.show()