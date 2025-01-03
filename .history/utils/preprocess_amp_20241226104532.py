import numpy as np

data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_amp.npy')

for i in range(1, data.shape[0]):
                for j in range(data.shape[1]):  # Iterate over columns
                    if np.isinf(data[i, j]):
                        data[i, j] = data[i, j-1]   