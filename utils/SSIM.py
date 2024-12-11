import torch
import matplotlib.pyplot as plt
from ignite.engine import *
from ignite.metrics import *
import numpy as np


target = np.load(r'D:\Pycharm Projects\Pytorch_Template\output\label_reshape_back.npy')
prediction = np.load(r'D:\Pycharm Projects\Pytorch_Template\output\fusion_reshape_back.npy')
target = target[:5]
prediction = prediction[:5]

target = torch.tensor(target)
prediction = torch.tensor(prediction)

target = torch.chunk(input=target,
                     chunks=100,   # 每个图片对应500个元素 对应的result的shape 300000 1 288 1
                     dim=1)

prediction = torch.chunk(input=prediction,
                     chunks=100,   # 每个图片对应500个元素 对应的result的shape 300000 1 288 1
                     dim=1)
print(len(target),len(prediction))
target = list(target)
prediction = list(prediction)
ssim_list = []

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

for i in range(100):
    target_i = torch.squeeze(target[i])  # 288 951
    target_i = torch.unsqueeze(target_i, dim=0)  # 1 288 951
    target_i = torch.unsqueeze(target_i, dim=1)  # 1 1 288 951

    prediction_i = torch.squeeze(prediction[i])  # 288 951
    prediction_i = torch.unsqueeze(prediction_i, dim=0)  # 1 288 951
    prediction_i = torch.unsqueeze(prediction_i, dim=1)  # 1 1 288 951
    # print(target_i.shape, prediction_i.shape)
    metric = SSIM(data_range=1.0)
    metric.attach(default_evaluator, 'ssim')

    preds = prediction_i.float()
    targets = target_i.float()

    state = default_evaluator.run([[preds, targets]])
    ssim_list.append(state.metrics['ssim'])

ssim_array = np.array(ssim_list)
np.savetxt('fusion_values.txt', ssim_array)
print("SSIM values saved to 'ssim_values.txt'")

# x1 = range(0, 100)
# y1 = ssim_list
# plt.plot(x1, y1, '.-')
# plt.xlabel('SSIM vs epochs')
# plt.ylabel('SSIM_point')
# plt.savefig("DenseNetSSIM.jpg")
# plt.show()