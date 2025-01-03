import numpy as np
import fast_glcm
import matplotlib.pyplot as plt

def main():
    pass

if __name__ == '__main__':
    main()
    data = np.load()

    glcm_mean = fast_glcm.fast_glcm_mean(data)
    plt.imshow(glcm_mean)
    plt.tight_layout()
    plt.show()