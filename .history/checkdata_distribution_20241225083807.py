import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
data = data.reshape(-1, 951, 288)
data = data[0]

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50)
plt.title('Histogram')
plt.show()

# KDE (Kernel Density Estimation) Plot
sns.kdeplot(data=data)
plt.title('KDE Plot')
plt.show()

# Box Plot - good for outlier detection
plt.boxplot(data)
plt.title('Box Plot')
plt.show()

# Q-Q Plot - good for checking normality
from scipy import stats
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()