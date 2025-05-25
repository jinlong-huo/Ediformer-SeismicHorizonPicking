from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

predicted_label1 = np.load(r'D:\Pycharm Projects\Horizon_Picking\seg_patch_reshape_back.npy')
predicted_label2 = np.load(r'D:\Pycharm Projects\Horizon_Picking\Dformer_phase_patch_reshape_back.npy')
predicted_label3 = np.load(r'D:\Pycharm Projects\Horizon_Picking\9_9_patch_reshape_back.npy')
true_label_all = np.load(r'D:\Pycharm Projects\Horizon_Picking\dformer_best_label_reshape_back.npy')
true_label_all = np.squeeze(true_label_all)

# print(predicted_label1.shape,predicted_label2.shape,predicted_label2.shape, true_label_all.shape)

# true_label_all = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy.npy')

predicted_label1 = np.swapaxes((predicted_label1), -1, 1)
predicted_label2 = np.swapaxes((predicted_label2), -1, 1)
predicted_label3 = np.swapaxes((predicted_label3), -1, 1)
true_label_all = np.swapaxes((true_label_all), -1, 1)

#
# # print(predicted_label1.shape, predicted_label2.shape, predicted_label3.shape, true_label_all.shape)
# #
predicted_label1 = predicted_label1.reshape(-1, 288)
predicted_label2 = predicted_label2.reshape(-1, 288)
predicted_label3 = predicted_label3.reshape(-1, 288)
true_label_all = true_label_all.reshape(-1, 288)
#
# # print(predicted_label1.shape, predicted_label2.shape, predicted_label3.shape, true_label_all.shape)
#
# # Save the reshaped data as a CSV file
# # np.savetxt('predicted_label1.csv', predicted_label1, delimiter=',')
# # np.savetxt('predicted_label2.csv', predicted_label2, delimiter=',')
# # np.savetxt('predicted_label3.csv', predicted_label3, delimiter=',')
# # np.savetxt('true_label_all.csv', true_label_all, delimiter=',')
#
#
#
predicted_label1 = predicted_label1[:]
predicted_label2 = predicted_label2[:]
predicted_label3 = predicted_label3[:]
true_label_all = true_label_all[ :]
#

# predicted_labels = [np.squeeze(predicted_label1), np.squeeze(predicted_label2), np.squeeze(predicted_label3)]
# true_labels_all = [np.squeeze(true_label_all), np.squeeze(true_label_all), np.squeeze(true_label_all)]
# # Calculate one figure
predicted_labels = [np.squeeze(predicted_label2)]
true_labels_all = [np.squeeze(true_label_all)]
num_files = len(predicted_labels)

fig, axes = plt.subplots(1, num_files, figsize=(18, 5))

# label = ['(a)' '(b)' '(c)']
label = ['(a)']
for i in range(num_files):
    predicted_label = predicted_labels[i]
    true_label_all = true_labels_all[i]
    pred_label_all_cm = predicted_label.flatten()
    true_label_all_cm = true_label_all.flatten()

    classes = ['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6']
    cm = confusion_matrix(true_label_all_cm, pred_label_all_cm, labels=range(len(classes)))
    class_counts = np.sum(cm, axis=1)
    normalized_cm = cm / class_counts[:, np.newaxis]

    ax = axes
    # ax = axes[i]
    im = ax.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes,
           yticklabels=classes,
           title='{}'.format(str(label[i])),
           ylabel='True Label',
           xlabel='Predicted Label')
    fig.tight_layout()

    thresh = normalized_cm.max() / 2.
    for j in range(len(classes)):
        for k in range(len(classes)):
            ax.text(k, j, format(normalized_cm[j, k], '.2f'),
                    ha="center", va="center",
                    color="white" if normalized_cm[j, k] > thresh else "black")

ax.figure.colorbar(im, ax=ax)
plt.savefig(r"C:\Users\Administrator\Desktop\SegFormer_matrix.png", dpi=500)
plt.show()