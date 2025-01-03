import numpy as np

# Load the data
data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_amp.npy')

# Find where the infinities are
inf_indices = np.isinf(data)

# Replace infinities with the previous value along each column
data[inf_indices] = np.roll(data, shift=1, axis=0)[inf_indices]

# Save the modified data
np.save('/home/dell/disk1/Jinlong/Horizontal-data/F3_amp_modified.npy', data)
