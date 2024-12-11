import numpy as np
import matplotlib.pyplot as plt

segformer_dices = [0.9700994491577148, 0.9573017954826355, 0.9627752900123596, 0.8763929009437561, 0.8598835468292236, 0.8349288702011108, 0.987147331237793]
dformer_dices = [0.9916325211524963, 0.9617410898208618, 0.9522289037704468, 0.7963778972625732, 0.7999752163887024, 0.7628410458564758, 0.9885793328285217]
mdormer_dices = [0.9983333945274353, 0.996299684047699, 0.9943389892578125, 0.963957667350769, 0.9580138325691223, 0.9692031741142273, 0.9989721775054932]

# Combine the dice results for all models
all_dices = np.vstack((segformer_dices, dformer_dices, mdormer_dices))
num_classes = 7
class_labels = range(1, num_classes+1)

x = np.arange(num_classes)
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))
rects = []
colors = ['#CDC9C9', '#EE7621','#87CEFA']  # Colors for each model
# colors = ['#FFC107', '#3F51B5', '#4CAF50']  # New color codes


# colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']

for i in range(len(all_dices)):
    offset = (i - 1) * width / 2
    rect = ax.bar(x + offset, all_dices[i], width, color=colors[i])
    rects.append(rect)

# Customize plot
ax.set_xlabel('Class', fontsize=10)
ax.set_ylabel('Normalized DICE Coefficient', fontsize=10)
# ax.set_title('DICE Coefficients for Each Class in Three Models', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(class_labels, fontsize=10)
ax.legend(rects, ['Segformer', 'DFormer', 'MDormer'], fontsize=10, loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.5)

# Adjust y-axis limits
ax.set_ylim(0, 1.2)

plt.tight_layout()
plt.savefig('bar_plot.png', dpi=300)
plt.show()





















