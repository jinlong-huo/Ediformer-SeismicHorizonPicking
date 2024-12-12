# import torch, time, os
# import numpy as np
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import torch.utils.data.dataset as Dataset
# import torch.utils.data.dataloader as DataLoader
# import argparse
#
# import torch.nn.functional as F
# from tqdm import tqdm
#
# from torch.autograd import Variable
# from torch.cuda.amp import autocast as autocast
# from DODmodel import DexiNed
# from torch import nn
#
#
# torch.set_printoptions(profile="full")  # 全输出
#
# os.environ['CUDA_VISIBLE_DEVICES']='0' # gpu1 keyi
# # np.set_printoptions(profile="full")    # 全输出
#
#
# # train dataset
# class MyDataset(Dataset.Dataset):
#     def __init__(self, Data1,Label):
#         self.Data1 = Data1
#         self.Label = Label
#
#     def __getitem__(self, idx):
#         data1 = torch.Tensor(self.Data1[idx])  # .permute(0, 2, 1)
#         label = torch.Tensor(self.Label[idx])  # .permute(1, 0)   # (7, 288) 读出来是有问题的 index的值和batch_size的值相等了
#
#         return data1, label
#
#     def __len__(self):
#         return len(self.Data1)
#
# # train dataset
# class MyDataset2(Dataset.Dataset):
#     def __init__(self, Data2,Labe2):
#         self.Data2 = Data2
#         self.Label2 = Labe2
#
#     def __getitem__(self, idx):
#         data2 = torch.Tensor(self.Data2[idx-4])  # .permute(0, 2, 1)
#         label2 = torch.Tensor(self.Label2[idx-4])  # .permute(1, 0)   # (7, 288) 读出来是有问题的 index的值和batch_size的值相等了
#
#         return data2, label2
#
#     def __len__(self):
#         return len(self.Data2)
#
# class TestDataset(Dataset.Dataset):
#     def __init__(self, TestData1, TestLabel):
#         self.TestData1 = TestData1
#         self.TestLabel = TestLabel
#
#     def __getitem__(self, idx):
#         test_data1 = torch.Tensor(self.TestData1[idx])  # .permute(0, 2, 1)
#         test_label = torch.Tensor(self.TestLabel[idx])  # .permute(1, 0)
#         if idx>1:
#             test_data1 = torch.Tensor(self.TestData1[idx])  # .permute(0, 2, 1)
#             test_label = torch.Tensor(self.TestLabel[idx])
#         print(idx,test_data1[idx].shape)
#         return test_data1, test_label
#
#     # else:
#     #     def __getitem__(self, idx):
#     #         test_data1 = torch.Tensor(self.TestData1[idx])  # .permute(0, 2, 1)
#     #         test_label = torch.Tensor(self.TestLabel[idx])  # .permute(1, 0)
#     #         print(idx)
#     #         return test_data1, test_label
#
#     def __len__(self):
#         return len(self.TestData1)
#
#
#
# def train_one_epoch(epoch, dataloader, model, criterion, optimizer, tb_writer):
#     batch_size = 20
#     correct_total = 0
#     total = 0
#     correct = 0
#
#     model.train()
#     loss_avg = []
#     Loss_list = []
#     accuracy_c = []
#     accuracy = []
#     train_label_for_fusion = None
#     l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]  # 最后一个loss改成了2.3   12.6--75% 7.8--65%
#     scaler = torch.cuda.amp.GradScaler()
#     model = model.cuda()  # 放到cuda里
#
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     predicted_label = None
#     train_tq = tqdm(enumerate(dataloader))
#     for batch_id, item in enumerate(dataloader):
#         data1, label_a = item  # (20, 1, 288, 1)  (20, 288, 7)
#
#         data1 = Variable(data1).cuda()
#
#         label_a = Variable(label_a).cuda()
#
#         optimizer.zero_grad()
#         with autocast():
#             # with torch.no_grad():
#             # print(data1.shape,data2.shape)
#             output = model(data1)  # output[0-5]是 (20, 1, 1, 288) output[6]是(20, 7, 1, 288)
#             label = Variable(label_a).cuda()
#             l_weight = torch.tensor(l_weight, requires_grad=True)
#             l_weight = Variable(l_weight).cuda()
#             criterion = nn.CrossEntropyLoss(weight=l_weight)  # 损失函数加了权重
#             label = torch.squeeze(label)
#             loss = criterion(output[6], label.long())
#
#             _, output = torch.max(output[6], 1)
#             predicted_label = np_extend(predicted_label,output.cpu())
#             correct += (output == label).sum().item()
#
#             # 差一个梯度消除
#             # loss.requires_grad_(True)
#
#
#
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         loss_avg.append(loss.item())
#
#         if epoch == 0 and (batch_id == 100 and tb_writer is not None):
#             tmp_loss = np.array(loss_avg).mean()
#             tb_writer.add_scalar('loss', tmp_loss, epoch)
#
#         train_tq.desc = str(time.ctime()) + 'Epoch: {0} Sample {1}/{2} Loss: {3} '.format(epoch, batch_id,
#                                                                                           len(dataloader), loss.item())
#         total += label_a.size(0)
#
#     correct_total += correct
#     final = 100 * (correct_total / (total * 288 * 64))
#
#     loss_avg = np.array(loss_avg).mean()
#     Loss_list.append(loss_avg)
#     # print('train Loss ',loss_avg)
#     accuracy.append(final)
#     print(time.ctime(), 'Epoch: {0}  Loss: {1} '
#           .format(epoch, loss_avg))
#     return loss_avg, accuracy  # 加上了acc
#
# device = torch.device('cuda')
# print(device)
# def validate(model, val_loader, val_dataset, criterion):
#     print('Validating')
#     model.eval()
#     val_running_loss = 0.0
#     val_running_correct = 0
#     counter = 0
#     total = 0
#     prog_bar = tqdm(enumerate(val_loader), total=int(len(val_dataset) / val_loader.batch_size))
#     with torch.no_grad():
#         for i, data in prog_bar:
#             counter += 1
#             data, target = data[0].to(device), data[1].to(device)
#             total += target.size(0)
#             target = torch.tensor(target, dtype=torch.int64).to(device)
#             target = target.clone().detach()
#             target = torch.squeeze(target)
#             outputs = model(data)
#             # print(target.shape,outputs[6].shape)
#             loss = criterion(outputs[6], target)
#             val_running_loss += loss.item()
#             _, preds = torch.max(outputs[6].data, 1)
#             val_running_correct += (preds == target).sum().item()
#         print(val_running_correct)
#         val_loss = val_running_loss / counter
#
#         val_accuracy = 100. * val_running_correct / (total * 288 * 64)
#         # final = 100 * (correct_total / (total * 128 * 128))
#         return val_loss, val_accuracy
# # 数组拼接
# def np_extend(a, b, axis=0):
#     if a is None:
#         shape = list(b.shape)
#         shape[axis] = 0
#         a = np.array([]).reshape(tuple(shape))
#     return np.append(a, b, axis)
#
#
# def test(checkpoint_path, dataloader, model, device, output_dir):
#     model.load_state_dict(torch.load(checkpoint_path,
#                                      map_location=device))
#     # Put model in evaluation mode
#     predicted_label = None
#     predicted_data = None
#     seismic = None  # 临时变量存储上一次变量值
#     model = model.cuda()
#     model.eval()
#     test_loss_avg = []
#     correct = 0
#     total = 0
#     l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 415.8]  # 初始化权重
#
#     n_classes = 7
#     target_num = torch.zeros((1, n_classes))
#     predict_num = torch.zeros((1, n_classes))
#     acc_num = torch.zeros((1, n_classes))
#
#     with torch.no_grad():
#
#         for batch_id, item in enumerate(dataloader):
#
#             test_data1, test_label = item  # (571551, 1, 288, 1)  (571551, 288, 7)
#             seismic = np_extend(seismic,test_label)
#             test_data1 = Variable(test_data1).cuda()  #
#             test_label = Variable(test_label).cuda()
#             # test_label = torch.unsqueeze(test_label, dim=-2)  # (571551, 288, 7)-->(571551, 288, 1, 7)
#             pred_label = model(test_data1)
#             # print(pred_label[6].shape)
#             _, predicted = torch.max(pred_label[6].data, 1)  # (20, 1 ,288)
#
#             temp = predicted
#             temp_data = pred_label[6]
#             # 下面这句话是把所有的数组都拼接起来 如果要快速test记得注释！！！！！！
#             predicted_label = np_extend(predicted_label, temp.cpu())
#             predicted_data = np_extend(predicted_data, temp_data.cpu())
#             # print(predicted_label.shape,predicted_data.shape)
#                   # torch.Size([100, 1, 288, 1]) torch.Size([100, 288])
#
#             # 加上weight看下
#             l_weight = torch.tensor(l_weight, requires_grad=True)
#             l_weight = Variable(l_weight).cuda()
#             criterion = nn.CrossEntropyLoss(l_weight)
#             test_label = torch.squeeze(test_label,dim=1)
#             # print(test_label.shape,pred_label[6].shape)
#             test_loss = criterion(pred_label[6], test_label.long())  # label和data的size不一样
#             test_loss_avg.append(test_loss.item())
#             total += test_label.size(0)
#             correct += (predicted == test_label).sum().item()
#
#             test_loss = np.array(test_loss_avg).mean()
#             pred_label[6] = pred_label[6].view(-1, n_classes)
#
#             pre_mask = torch.zeros(pred_label[6].size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
#             predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
#             test_label = test_label.long()
#             # print(type(predicted), type(test_data))
#             tar_mask = torch.zeros(pred_label[6].size()).scatter_(1, test_label.data.cpu().view(-1, 1), 1.)
#             target_num += tar_mask.sum(0)  # 得到数据中每类的数量
#             acc_mask = pre_mask * tar_mask
#             acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量
#         recall = acc_num / target_num
#         precision = acc_num / predict_num + float('1e-8')
#         F1 = 2 * recall * precision / (recall + precision)
#         accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
#
#         print('Test Acc {}'.format(accuracy))
#         print('recall {}'.format(recall))
#         print('precision {}'.format(precision))
#         print('F1-score {}'.format(F1))
#
#         # if batch_id > 0 and batch_id % 2500 == 0:at
#         print('Test loss:', test_loss)
#         print('Accuracy of the network:', (100 * (correct / (total * 288))))
#         print('Begin Writing!')
#         # np.save('seismic_train_label_for_fusion.npy',seismic)
#         # np.save('1_25_seismic_test_data_for_fusion.npy', predicted_data)
#         np.save('2_17_seismic_test_label_for_fusion_64.npy', predicted_label)
#         # np.save('1_25_seismic_train_data_for_fusion.npy', predicted_data)
#         # np.save('1_25_seismic_train_predicted_label_for_fusion.npy', predicted_label)
#         print(predicted_data.shape,predicted_label.shape)
#         # np.save('1_9_seismic_nocon_data_for_fusion.npy', predicted_data)
#         return correct
#
#
# def main(args):
#     stime = time.ctime()
#     num_epoch = 3501
#     Loss_list = []
#     accuracy = []
#     count = 0
#     # 训练过程记录地址 Dexined
#     training_dir = r'D:\Pycharm Projects\DexiNed_horizon\process'
#     output_dir = r'D:\Pycharm Projects\DexiNed_horizon\process'
#     # D:\Pycharm Projects\DexiNed_horizon\checkpoints\1_model.pth
#     # checkpoints加载地址 Dexined # D:\Pycharm Projects\Dexi_horizon\Dex_Horizon_model.pth
#     checkpoint_dir = r'D:\Pycharm Projects\Horizon_Picking\checkpoints\1_25_seismic_1200_model.pth'
#     from torch.utils.tensorboard import SummaryWriter  # for torch 1.4 or greather
#     tb_writer = SummaryWriter(log_dir=training_dir)
#     device = torch.device('cpu' if torch.cuda.device_count() == 0
#                           else 'cuda')
#     model = DexiNed().to(device)
#     checkpoint_path = checkpoint_dir
#
#     if not args.is_testing:
#         print('*********is training *********')
#         # model.load_state_dict(torch.load(checkpoint_path,  # 要调用model生成的checkpoints
#         #                                  map_location=device))
#         # 训练集数据地址Wu
#
#         Data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
#         Label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
#
#         Data_1 = Data.reshape(-1, 951, 288)  # reshape 成为 三维数据
#         Data_1 = Data_1[::10, :, :]  # 再使用等间隔采样抽取五分之一数据
#         Data1 = Data_1[np.newaxis, :]  # 为了使用patch操作扩充一维数据
#         Data1 = torch.tensor(Data1)
#         Data1 = Data1.permute(0,1,-1,2)
#
#
#         Label = torch.tensor(Label)
#         Label = Label.reshape(-1, 951, 288)
#         Label = Label[::10, :, :]
#         Label = Label[np.newaxis, :]
#         Label = torch.tensor(Label).clone().detach()
#         Label = Label.permute(0,1,-1,2)
#
#         kc, kh, kw = 1, 288, 64  # kernel size
#         dc, dh, dw = 1, 64, 64  # stride
#         # Pad to multiples of 64
#
#         # no ovlerlapping
#         Data1 = F.pad(Data1, [Data1.size(3) % kw // 100, Data1.size(3) % kw // 2,
#                               Data1.size(2) % kh // 100, Data1.size(2) % kh // 2,
#                               Data1.size(1) % kc // 100, Data1.size(1) % kc // 2])
#         Data1 = Data1.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
#         # unfold_shape = Data1.size()
#         Data1 = Data1.contiguous().view(-1, kc, kh, kw)
#
#
#         Label = F.pad(Label, [Label.size(3) % kw // 100, Label.size(3) % kw // 2,
#                               Label.size(2) % kh // 100, Label.size(2) % kh // 2,
#                               Label.size(1) % kc // 100, Label.size(1) % kc // 2])
#         Label = Label.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
#         Label = Label.contiguous().view(-1, kc, kh, kw)
#         # Label = list(Label)
#         # import imageio
#         # for i in range(3630):
#         #     Label[i] = torch.squeeze(Label[i])
#         #     imageio.imwrite(str(i)+'_for_patches_orig_label.png', Label[i])
#         # unfold_shape = Label.size()
#         # print(Data1.shape, Label.shape, '********')
#         Data1=Data1[:1760,:]
#         Label=Label[:1760,:]
#         Label = torch.tensor(Label, dtype=torch.float)
#         print(Data1.shape, Label.shape, '********')
#         train_dataset = MyDataset(Data1, Label)
#
#         # train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)  # false效果比较好
#         length = len(train_dataset)
#
#         # change the coefficient to change the train dataset size
#         train_size, val_size = int(0.8 * length), int(0.2 * length)
#         # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [210000, 90000])
#         train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(0))
#

#         train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=10)
#         val_loader = DataLoader.DataLoader(dataset=val_dataset, batch_size=10)
#         # val_dataloader = DataLoader.DataLoader(dataset=val_dataset,batch_size=10, shuffle=True)
#
#
#
#     if args.is_testing:
#         print('_______ is testing _______')
#         # 测试集数据地址Wu
#         TestData1 = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
#         TestLabel = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
#
#         np_add = np.zeros((601, 9, 288))
#
#         TestData1 = TestData1.reshape(-1, 951, 288)  # reshape 成为 三维数据
#         TestData1 = np.concatenate((TestData1, np_add), axis=1)
#         TestData1 = TestData1[np.newaxis, :]  # 为了使用patch操作扩充一维数据
#         TestData1 = torch.tensor(TestData1)
#         TestData1 = TestData1.permute(1, 0, -1, 2)
#
#         TestLabel = TestLabel.reshape(-1, 951, 288)
#         TestLabel = np.concatenate((TestLabel, np_add), axis=1)
#         TestLabel = TestLabel[np.newaxis, :]
#         TestLabel = torch.tensor(TestLabel)
#         TestLabel = TestLabel.permute(1, 0, -1, 2)
#         TestData1 = torch.tensor(TestData1, dtype=torch.float)
#         TestLabel = torch.tensor(TestLabel, dtype=torch.float)
#
#         kc, kh, kw = 1, 288, 64  # kernel size
#         dc, dh, dw = 1, 64, 64  # stride
#         # Pad to multiples of 64
#
#         # no ovlerlapping
#         TestData1 = F.pad(TestData1, [TestData1.size(3) % kw // 100, TestData1.size(2) % kw // 2,
#                                       TestData1.size(2) % kh // 100, TestData1.size(1) % kh // 2,
#                                       TestData1.size(1) % kc // 100, TestData1.size(0) % kc // 2])
#         TestData1 = TestData1.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
#         TestData1 = TestData1.contiguous().view(-1, kc, kh, kw)
#         TestLabel = F.pad(TestLabel, [TestLabel.size(3) % kw // 100, TestLabel.size(2) % kw // 2,
#                                       TestLabel.size(2) % kh // 100, TestLabel.size(1) % kh // 2,
#                                       TestLabel.size(1) % kc // 100, TestLabel.size(0) % kc // 2])
#
#         TestLabel = TestLabel.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
#         # unfold_shape = TestLabel.size()
#         TestLabel = TestLabel.contiguous().view(-1, kc, kh, kw)
#         TestLabel = torch.tensor(TestLabel, dtype=torch.float)
#         test_dataset = TestDataset(TestData1, TestLabel)
#         # print(TestData1.shape,TestLabel.shape)
#         test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
#         test(checkpoint_path, test_loader, model, device, output_dir)
#
#         return
#
#     l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 4.2]
#     l_weight = torch.tensor(l_weight, requires_grad=True)
#     l_weight = Variable(l_weight).cuda()
#     criterion = nn.CrossEntropyLoss(l_weight)  # 加入权重和adamw和每一轮都去修正学习率看看效果
#     seed = 1021  # 1021 4.19改的521输出0.0002稳定
#     val_loss, val_accuracy = [], []
#     for epoch in range(num_epoch):  # 更改训练轮数
#
#         optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)  # YYDS!
#         # optimizer = optim.Adam(model.parameters(), lr=0.00001)
#
#         if epoch % 20 == 0:
#             seed = seed + 500
#             np.random.seed(seed)
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)
#             print("------ Random seed applied-------------")
#         # Create output directories
#
#
#         output_dir_epoch = 'D:/Pycharm Projects/Horizon_Picking/checkpoints/'
#         os.makedirs(output_dir_epoch, exist_ok=True)
#
#         avg_loss = train_one_epoch(epoch,
#                                    train_loader,
#                                    model,
#                                    criterion,
#                                    optimizer,
#                                    tb_writer)
#         y1, y2 = avg_loss
#         Loss_list.append(y1)
#         accuracy.append(y2)
#         val_epoch_loss, val_epoch_accuracy = validate(model,
#                                                         val_loader,
#                                                         val_dataset,
#                                                         criterion)
#
#         val_loss.append(val_epoch_loss)
#         val_accuracy.append(val_epoch_accuracy)
#
#         # print(f"Train Loss: {y1:.4f}, Train Acc: {y2:.2f}")
#         print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
#
#
#
#
#         if epoch>0 and epoch % 100 == 0:
#             torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
#                        os.path.join(output_dir_epoch, '1_25_seismic_{0}_model.pth'.format(epoch)))
#         # tb_writer.add_scalar('loss',
#
#     etime = time.ctime()
#     print(stime, etime)
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='DexHorizon_trainer.')
#     is_testing = False  # Ture  False
#     parser.add_argument('--is_testing', type=bool,
#                         default=is_testing,
#                         help='Script in testing mode')
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     print(args)
#     main(args)




import torch, time, os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import argparse

import torch.nn.functional as F
from tqdm import tqdm

from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from DODmodel import DexiNed
from torch import nn
# from Early_LR import EarlyStopping, LRScheduler, EarlyStopper

torch.set_printoptions(profile="full")  # 全输出

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # gpu1 keyi


# np.set_printoptions(profile="full")    # 全输出


# train dataset
class MyDataset(Dataset.Dataset):
    def __init__(self, Data1, Label):
        self.Data1 = Data1
        self.Label = Label

    def __getitem__(self, idx):
        data1 = torch.Tensor(self.Data1[idx])  # .permute(0, 2, 1)
        label = torch.Tensor(self.Label[idx])  # .permute(1, 0)   # (7, 288) 读出来是有问题的 index的值和batch_size的值相等了

        return data1, label

    def __len__(self):
        return len(self.Data1)


class TestDataset(Dataset.Dataset):
    def __init__(self, TestData1, TestLabel):
        self.TestData1 = TestData1

        self.TestLabel = TestLabel

    def __getitem__(self, idx):
        test_data1 = torch.Tensor(self.TestData1[idx])  # .permute(0, 2, 1)
        test_label = torch.Tensor(self.TestLabel[idx])  # .permute(1, 0)   (7, 288) 读出来是有问题的 index的值和batch_size的值相等了

        return test_data1, test_label

    def __len__(self):
        return len(self.TestData1)


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, tb_writer):
    batch_size = 20
    correct_total = 0
    total = 0
    correct = 0

    model.train()
    loss_avg = []
    Loss_list = []
    accuracy_c = []
    accuracy = []
    train_label_for_fusion = None
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]  # 最后一个loss改成了2.3   12.6--75% 7.8--65%
    scaler = torch.cuda.amp.GradScaler()
    model = model.cuda()  # 放到cuda里

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    predicted_label = None
    train_tq = tqdm(enumerate(dataloader))
    for batch_id, item in enumerate(dataloader):
        data1, label_a = item  # (20, 1, 288, 1)  (20, 288, 7)

        data1 = Variable(data1).cuda()

        label_a = Variable(label_a).cuda()

        optimizer.zero_grad()
        with autocast():
            # with torch.no_grad():
            # print(data1.shape,data2.shape)
            output = model(data1)  # output[0-5]是 (20, 1, 1, 288) output[6]是(20, 7, 1, 288)
            label = Variable(label_a).cuda()
            l_weight = torch.tensor(l_weight, requires_grad=True)
            l_weight = Variable(l_weight).cuda()
            criterion = nn.CrossEntropyLoss(weight=l_weight)  # 损失函数加了权重
            label = torch.squeeze(label)

            print(label.shape, output[6].shape, '*****(((((')

            loss = criterion(output[6], label.long())

            _, output = torch.max(output[6], 1)
            predicted_label = np_extend(predicted_label, output.cpu())
            correct += (output == label).sum().item()

            # 差一个梯度消除
            # loss.requires_grad_(True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_avg.append(loss.item())

        if epoch == 0 and (batch_id == 100 and tb_writer is not None):
            tmp_loss = np.array(loss_avg).mean()
            tb_writer.add_scalar('loss', tmp_loss, epoch)

        train_tq.desc = str(time.ctime()) + 'Epoch: {0} Sample {1}/{2} Loss: {3} '.format(epoch, batch_id,
                                                                                          len(dataloader), loss.item())
        total += label_a.size(0)

    correct_total += correct
    final = 100 * (correct_total / (total * 288 * 64))

    loss_avg = np.array(loss_avg).mean()
    Loss_list.append(loss_avg)
    print('train Loss ', loss_avg)
    accuracy.append(final)
    print(time.ctime(), 'Epoch: {0}  Loss: {1} '
          .format(epoch, loss_avg))
    return loss_avg, accuracy  # 加上了acc



device = torch.device('cuda')
print(device)


def validate(model, val_loader, val_dataset, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(val_loader), total=int(len(val_dataset) / val_loader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            total += target.size(0)
            target = torch.tensor(target, dtype=torch.int64).to(device)
            target = target.clone().detach()
            target = torch.squeeze(target)
            outputs = model(data)

            loss = criterion(outputs[6], target)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs[6].data, 1)
            val_running_correct += (preds == target).sum().item()
        val_loss = val_running_loss / counter

        val_accuracy = 100. * val_running_correct / (total * 288 * 64)
        # final = 100 * (correct_total / (total * 128 * 128))
        return val_loss, val_accuracy


# 数组拼接
def np_extend(a, b, axis=0):
    if a is None:
        shape = list(b.shape)
        shape[axis] = 0
        a = np.array([]).reshape(tuple(shape))
    return np.append(a, b, axis)


def test(checkpoint_path, dataloader, model, device, output_dir):
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location='cpu'))
    # Put model in evaluation mode
    predicted_label = None
    predicted_data = None
    temp = None  # 临时变量存储上一次变量值
    model = model.cuda()
    model.eval()
    test_loss_avg = []
    correct = 0
    total = 0
    freq=  None
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 415.8]  # 初始化权重

    n_classes = 7
    target_num = torch.zeros((1, n_classes))
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))

    with torch.no_grad():
        for batch_id, item in enumerate(dataloader):
            test_data1, test_label = item  # (571551, 1, 288, 1)  (571551, 288, 7)
            # freq = np_extend(freq,test_label)
            test_data1 = Variable(test_data1).cuda()  #

            test_label = Variable(test_label).cuda()

            test_label = torch.unsqueeze(test_label, dim=-2)  # (571551, 288, 7)-->(571551, 288, 1, 7)

            pred_label = model(test_data1)
            _, predicted = torch.max(pred_label[6].data, 1)
            print(predicted.shape)
            temp = predicted
            predicted_label = np_extend(predicted_label, temp.cpu())
            l_weight = torch.tensor(l_weight, requires_grad=True)
            l_weight = Variable(l_weight).cuda()
            criterion = nn.CrossEntropyLoss(l_weight)
            test_label = torch.squeeze(test_label)
            test_loss = criterion(pred_label[6], test_label.long())  # label和data的size不一样
            test_loss_avg.append(test_loss.item())
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()

            test_loss = np.array(test_loss_avg).mean()
            pred_label[6] = pred_label[6].view(-1, n_classes)

            pre_mask = torch.zeros(pred_label[6].size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            test_label = test_label.long()
            # print(type(predicted), type(test_data))
            tar_mask = torch.zeros(pred_label[6].size()).scatter_(1, test_label.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量
        recall = acc_num / target_num
        precision = acc_num / predict_num + float('1e-8')
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
        print('Test Acc {}'.format(accuracy))
        print('recall {}'.format(recall))
        print('precision {}'.format(precision))
        print('F1-score {}'.format(F1))

        # if batch_id > 0 and batch_id % 2500 == 0:at
        print('Test loss:', test_loss)
        print('Accuracy of the network:', (100 * (correct / (total * 288))))
        print('Begin Writing!')

        # np.save('1_25_freq_test_data_for_fusion.npy', predicted_data)
        # np.save('1_25_freq_test_label_for_fusion.npy', predicted_label)
        # np.save('2_22_freq_train_data_for_fusion.npy', predicted_data)
        np.save('2_22_freq_train_predicted_label_for_fusion_100_1000.npy', predicted_label)
        # np.save('1_9_RMS_Amp_nocon_label_for_fusion.npy', predicted_label)
        # np.save('1_9_RMS_Amp_nocon_data_for_fusion.npy', predicted_data)
        print(predicted_data.shape)
        return correct


def main(args):
    stime = time.ctime()
    num_epoch = 2501
    Loss_list = []
    accuracy = []
    count = 0
    # 训练过程记录地址 Dexined
    # training_dir = r'D:\Pycharm Projects\DexiNed_horizon\process'
    # output_dir = r'D:\Pycharm Projects\DexiNed_horizon\process'
    # # D:\Pycharm Projects\DexiNed_horizon\checkpoints\1_model.pth
    # # checkpoints加载地址 Dexined # D:\Pycharm Projects\Dexi_horizon\Dex_Horizon_model.pth
    # checkpoint_dir = r'D:\Pycharm Projects\Horizon_Picking\checkpoints\1_25_seismic_1200_model.pth'
    # from torch.utils.tensorboard import SummaryWriter  # for torch 1.4 or greather
    # tb_writer = SummaryWriter(log_dir=training_dir)
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    model = DexiNed().to(device)
    # checkpoint_path = checkpoint_dir
    # early_stopping = EarlyStopping()
    if not args.is_testing:
        print('*********is training *********')
        # model.load_state_dict(torch.load(checkpoint_path,  # 要调用model生成的checkpoints
        #                                  map_location=device))
        # 训练集数据地址Wu


        # Data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        # Label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        Data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy')
        Label = np.load('/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy')

        # Data = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\train\RMS_Amp_train_data_no_black.npy')
        # Label = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\train\train_label_no_black.npy')
        Data = torch.tensor(Data, dtype=torch.float)
        Label = torch.tensor(Label, dtype=torch.float)
        Data = Data.reshape(-1, 951, 288)

        Data = Data[np.newaxis, :]
        Data = torch.tensor(Data)
        Data = Data.permute(0, 1,  -1, 2)

        Label = Label.reshape(-1, 951, 288)

        Label = Label[np.newaxis, :]
        Label = torch.tensor(Label).clone().detach()
        Label = Label.permute(0, 1, -1, 2)
        kc, kh, kw = 1, 288, 64  # kernel size
        dc, dh, dw = 1, 64, 32  # stride
        # Pad to multiples of 64


        Data = F.pad(Data, [Data.size(3) % kw // 2, Data.size(3) % kw // 2,
                              Data.size(2) % kh // 2, Data.size(2) % kh // 2,
                              Data.size(1) % kc // 2, Data.size(1) % kc // 2])
        Data = Data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        # unfold_shape = Data1.size()
        Data = Data.contiguous().view(-1, kc, kh, kw)

        Label = F.pad(Label, [Label.size(3) % kw // 100, Label.size(3) % kw // 2,
                              Label.size(2) % kh // 100, Label.size(2) % kh // 2,
                              Label.size(1) % kc // 100, Label.size(1) % kc // 2])
        Label = Label.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        Label = Label.contiguous().view(-1, kc, kh, kw)
        Label = torch.tensor(Label, dtype=torch.float)

        train_dataset = MyDataset(Data, Label)
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [210000, 90000])
        # train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)  # false效果比较好
        length = len(train_dataset)
        print(length)
        # change the coefficient to change the train dataset size
        train_size, val_size = int(0.8 * length), int(0.2 * length)

        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [1500, 269],generator=torch.Generator().manual_seed(0))

        train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=50)
        val_loader = DataLoader.DataLoader(dataset=val_dataset, batch_size=50)
        # val_dataloader = DataLoader.DataLoader(dataset=val_dataset,batch_size=1, shuffle=True)

    if args.is_testing:
        print('_______ is testing _______')
        # # 测试集数据地址Wu
        # TestData1 = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\F3_crop_horizon_freq.npy')  # 测试用数据和标签
        # TestLabel = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy') # 现在的test是所有的标签
        TestData1 = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        TestLabel = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')

        # print(TestData1.shape, TestLabel.shape)
        # TestData1 = np.load( r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\test\F3_rms_test_data.npy')  # 测试用数据和标签
        # TestLabel = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\test\test_label_no_con.npy')

        TestData1 = torch.tensor(TestData1, dtype=torch.float)
        TestLabel = torch.tensor(TestLabel, dtype=torch.float)
        TestData1 = TestData1.reshape(-1, 951, 288)  # reshape 成为 三维数据
        # TestData1 = TestData1[:100,]
        TestData1 = TestData1[np.newaxis, :]  # 为了使用patch操作扩充一维数据
        TestData1 = torch.tensor(TestData1)
        TestData1 = TestData1.permute(0, 1, -1, 2)

        TestLabel = TestLabel.reshape(-1, 951, 288)
        # TestLabel = TestLabel[:100,]
        TestLabel = TestLabel[np.newaxis, :]
        TestLabel = torch.tensor(TestLabel)
        TestLabel = TestLabel.permute(0, 1, -1, 2)
        print(TestData1.shape, TestLabel.shape)
        kc, kh, kw = 1, 288, 64  # kernel size
        dc, dh, dw = 1, 64, 64  # stride
        # Pad to multiples of 64

        # no ovlerlapping
        TestData1 = F.pad(TestData1, [TestData1.size(3) % kw // 100, TestData1.size(2) % kw // 2,
                                      TestData1.size(2) % kh // 100, TestData1.size(1) % kh // 2,
                                      TestData1.size(1) % kc // 100, TestData1.size(0) % kc // 2])
        TestData1 = TestData1.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = TestData1.size()
        print(unfold_shape)
        TestData1 = TestData1.contiguous().view(-1, kc, kh, kw)

        TestLabel = F.pad(TestLabel, [TestLabel.size(3) % kw // 100, TestLabel.size(2) % kw // 2,
                                      TestLabel.size(2) % kh // 100, TestLabel.size(1) % kh // 2,
                                      TestLabel.size(1) % kc // 100, TestLabel.size(0) % kc // 2])

        TestLabel = TestLabel.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        # unfold_shape = TestLabel.size()
        TestLabel = TestLabel.contiguous().view(-1, kc, kh, kw)
        TestLabel = torch.tensor(TestLabel, dtype=torch.float)
        TestData1= TestData1[100:1000,:]
        TestLabel = TestLabel[100:1000,:]
        print(TestData1.shape, TestLabel.shape)
        test_dataset = TestDataset(TestData1, TestLabel)
        test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=50, shuffle=False)

        # Data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        # Label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        #
        # Data = torch.tensor(Data, dtype=torch.float)
        # Label = torch.tensor(Label, dtype=torch.float)
        # Data_1 = Data.reshape(-1, 951, 288)  # reshape 成为 三维数据
        # Data_1 = Data_1[::10, :, :]  # 再使用等间隔采样抽取五分之一数据
        # Data1 = Data_1[np.newaxis, :]  # 为了使用patch操作扩充一维数据
        # Data1 = torch.tensor(Data1)
        # Data1 = Data1.permute(0, 1, -1, 2)
        #
        # Label = Label.reshape(-1, 951, 288)
        # Label = Label[::10, :, :]
        # Label = Label[np.newaxis, :]
        # Label = torch.tensor(Label).clone().detach()
        # Label = Label.permute(0, 1, -1, 2)
        #
        # kc, kh, kw = 1, 288, 64  # kernel size
        # dc, dh, dw = 1, 64, 64  # stride
        # # Pad to multiples of 64
        #
        # # no ovlerlapping
        # Data1 = F.pad(Data1, [Data1.size(3) % kw // 100, Data1.size(3) % kw // 2,
        #                       Data1.size(2) % kh // 100, Data1.size(2) % kh // 2,
        #                       Data1.size(1) % kc // 100, Data1.size(1) % kc // 2])
        # Data1 = Data1.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        # # unfold_shape = Data1.size()
        # Data1 = Data1.contiguous().view(-1, kc, kh, kw)
        #
        # Label = F.pad(Label, [Label.size(3) % kw // 100, Label.size(3) % kw // 2,
        #                       Label.size(2) % kh // 100, Label.size(2) % kh // 2,
        #                       Label.size(1) % kc // 100, Label.size(1) % kc // 2])
        # Label = Label.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        # Label = Label.contiguous().view(-1, kc, kh, kw)
        # print(Data1.shape, Label.shape)
        #
        # # unfold_shape = Label.size()
        # Data1 = Data1[:3500, :, :, :]
        # Label = Label[:3500, :, :, :]
        #
        # Label = torch.tensor(Label, dtype=torch.float)
        # train_dataset = MyDataset(Data1, Label)
        # # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [210000, 90000])
        # # train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)  # false效果比较好
        # length = len(train_dataset)
        # # change the coefficient to change the train dataset size
        # train_size, val_size = int(0.8 * length), int(0.2 * length)
        #
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(0))
        #
        # train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=50)
        # output_dir = 'E:/Cal_HJL/Pycharm Projects/Horizon_Picking/output/'

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!---------------------------------------
        # change the dataloader to trainloader to make fusion train input data
        # change the dataloader to testloader to make fusion test input data
        test(checkpoint_path, test_loader, model, device, output_dir)

        return

    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 4.2]
    l_weight = torch.tensor(l_weight, requires_grad=True)
    l_weight = Variable(l_weight).cuda()
    criterion = nn.CrossEntropyLoss(l_weight)
    seed = 1021
    val_loss, val_accuracy = [], []
    for epoch in range(num_epoch):

        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)
        # optimizer = optim.Adam(model.parameters(), lr=0.0001)
        # lr_scheduler = LRScheduler(optimizer)
        if epoch % 20 == 0:
            seed = seed + 500
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")

        output_dir_epoch = 'E:/Cal_HJL/Pycharm Projects/Horizon_Picking/checkpoints/'
        os.makedirs(output_dir_epoch, exist_ok=True)

        avg_loss = train_one_epoch(epoch,
                                   train_loader,
                                   model,
                                   criterion,
                                   optimizer,
                                   tb_writer)
        y1, y2 = avg_loss
        Loss_list.append(y1)
        accuracy.append(y2)
        val_epoch_loss, val_epoch_accuracy = validate(model,
                                                      val_loader,
                                                      val_dataset,
                                                      criterion)

        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        # print(f"Train Loss: {y1:.4f}, Train Acc: {y2:.2f}")
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')

        # early_stopper = EarlyStopper(patience=1, min_delta=1)
        # print(early_stopper.early_stop(val_epoch_loss))

        if epoch > 0 and epoch % 20 == 0:
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                       os.path.join(output_dir_epoch, '1_25_freq_{0}_model.pth'.format(epoch)))

        # tb_writer.add_scalar('loss',
        #                          avg_loss,
        #                          epoch + 1)
    # plt.figure(figsize=(10, 7))
    # plt.plot(y1, color='green', label='train accuracy')
    # plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.figure(figsize=(10, 7))
    # plt.plot(y2, color='orange', label='train loss')
    # plt.plot(val_loss, color='red', label='validataion loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # if not args.is_testing:
    #     x1 = range(0, num_epoch)  # 和训练轮数一样
    #     y1 = Loss_list
    #     plt.plot(x1, y1, '.-')
    #     plt.xlabel('Train loss vs. epochs')
    #     plt.ylabel('Train loss')
    #     plt.figure(1)
    #     plt.savefig('E:/Cal_HJL/Pycharm Projects/Horizon_Picking/output/S_Train_loss.png', dpi=1000)
    #     plt.show()
    #     plt.close()
    #
    #     x2 = range(0, num_epoch)  # 和训练轮数一样
    #     y2 = accuracy
    #     # print(y2)
    #     plt.plot(x2, y2, '.-')
    #     plt.xlabel('accuracy vs. epochs')
    #     plt.ylabel('accuracy ')
    #     plt.figure(2)
    #     plt.savefig('E:/Cal_HJL/Pycharm Projects/Horizon_Picking/output/S_Train_acc.png', dpi=1000)
    #     plt.show()
    #     plt.close()
    #     # print(type(accuracy))
    #     # accuracy = [i for j in range(len(accuracy)) for i in accuracy[j]]
    # accuracy = sum(accuracy, [])
    # print(accuracy, '\n', Loss_list)

    etime = time.ctime()
    print(stime, etime)


def parse_args():
    parser = argparse.ArgumentParser(description='DexHorizon_trainer.')
    is_testing = True  # 默认是Ture 测试模式 False 训练
    parser.add_argument('--is_testing', type=bool,
                        default=is_testing,
                        help='Script in testing mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)


