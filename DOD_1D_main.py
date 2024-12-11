import torch, time, os
import numpy as np
import torch.optim as optim
# import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import argparse
import csv
# from tqdm import tqdm

from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from model import Diformer
from torch import nn

torch.set_printoptions(profile="full")   # 全输出
# np.set_printoptions(profile="full")    # 全输出


# train dataset
class MyDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __getitem__(self, idx):
        data = torch.Tensor(self.Data[idx])  
        label = torch.IntTensor(self.Label[idx])  

        return data, label

    def __len__(self):
        return len(self.Data)


# test dataset
class TestDataset(Dataset.Dataset):
    def __init__(self, TestData, TestLabel):
        self.TestData = TestData
        self.TestLabel = TestLabel

    def __getitem__(self, idx):
        test_data = torch.Tensor(self.TestData[idx])  # .permute(0, 2, 1)
        test_label = torch.IntTensor(self.TestLabel[idx])  # .permute(1, 0)   (7, 288) 读出来是有问题的 index的值和batch_size的值相等了

        return test_data, test_label

    def __len__(self):
        return len(self.TestData)


def train_one_epoch(epoch, dataloader, model, criterion, optimizer):

    batch_size = 20
    correct_total = 0
    total = 0
    correct = 0

    model.train()
    loss_avg = []
    Loss_list = []
    accuracy_c = []
    accuracy = []
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]  # 最后一个loss改成了2.3   12.6--75% 7.8--65%
    scaler = torch.cuda.amp.GradScaler()
    model = model.cuda()  # 放到cuda里

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # train_tq = tqdm(enumerate(dataloader))
    for batch_id, item in enumerate(dataloader):
        data, label_a = item                            # (20, 1, 288, 1)  (20, 288, 7)
        data = Variable(data).cuda()
        label_a = Variable(label_a).cuda()


        label_a = torch.unsqueeze(label_a, dim=-2)            # (20, 7, 288)-->(20，7，1，288)
        data = torch.unsqueeze(data, dim=-2)            # (20, 7, 288)-->(20，7，1，288)

        # data = data.permute(0, 1, 3, 2)                     # (20, 1, 288, 1)-->(20, 1, 1, 288)

        # data = data.view(-1, 7)
        # with torch.no_grad():                             # 这个是为了减少gpu内存解决了 RuntimeError: CUDA out of memory.
        optimizer.zero_grad()
        with autocast():
            # with torch.no_grad():
            output = model(data)                           # output[0-5]是 (20, 1, 1, 288) output[6]是(20, 7, 1, 288

            label = label_a           # (20, 7, 1, 288)-----> (20, 7, 288) (50, 288)----(50, 1, 288)
            label = torch.squeeze(label, dim=-2)        # (20, 7, 1, 288)-----> (20, 7, 288)
            label = Variable(label).cuda()
            l_weight = torch.tensor(l_weight, requires_grad=True)
            l_weight = Variable(l_weight).cuda()
            criterion = nn.CrossEntropyLoss(weight=l_weight)
            loss = criterion(output[6], label.long())
            # print(output[6])
            _, output = torch.max(output[6], 1)
            # print(output[6])
            correct += (output == label).sum().item()

            # 差一个梯度消除
            # loss.requires_grad_(True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_avg.append(loss.item())

        total += label_a.size(0)
    correct_total += correct
    final = 100 * (correct_total / (total * 288))
    loss_avg = np.array(loss_avg).mean()
    Loss_list.append(loss_avg)
    accuracy.append(final)
    print(time.ctime(), 'Epoch: {0}  Loss: {1}  ACC:{2}'
          .format(epoch, loss_avg, accuracy))
    return loss_avg, accuracy


def validate_one_epoch(dataloader, model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_losses = []
    correct = 0
    val_loss = []
    val_acc = []

    with torch.no_grad():
        for batch_id, (val_data, val_label) in enumerate(dataloader):
            # print(val_data.shape, val_label.shape)
            val_data = torch.unsqueeze(val_data,dim=1)
            val_label = torch.squeeze(val_label)
            val_data = val_data.cuda()
            val_label = val_label.cuda()

            # val_data = val_data.permute(0, 1, 3, 2).cuda()
            # val_label = val_label.permute(0, -1, 1).cuda()

            val_preds = model(val_data)
            val_preds = torch.squeeze(val_preds[6], dim=-2)

            loss = criterion(val_preds, val_label.long())
            val_loss.append(loss.item())

            _, predicted = torch.max(val_preds, 1)
            correct = (predicted == val_label).sum().item()
            val_acc.append(correct / (val_data.size(0) * 288))

    num_samples = len(dataloader.dataset)
    print(num_samples)
    # val_losses.append(val_loss / num_samples * 288)
    val_loss = np.array(val_loss).mean()
    val_acc = np.array(val_acc).mean() * 100

    print('Test set: Average loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        val_loss, val_acc))
    return


def np_extend(a, b, axis = 0):
    if a is None:
        shape = list(b.shape)
        shape[axis] = 0
        a = np.array([]).reshape(tuple(shape))
    return np.append(a, b, axis)


def test(checkpoint_path, dataloader, model, device, output_dir):
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))
    # Put model in evaluation mode
    predicted_label = None
    temp = None  # 临时变量存储上一次变量值
    model = model.cuda()
    model.eval()
    test_loss_avg = []
    correct = 0
    total = 0
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 415.8] # 初始化权重

    n_classes = 7
    target_num = torch.zeros((1, n_classes))
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))

    with torch.no_grad():
        for batch_id, item in enumerate(dataloader):
            test_data, test_label = item                      # (571551, 1, 288, 1)  (571551, 288, 7)

            test_data = Variable(test_data).cuda()            # 放入variable 可以用gpu跑
            test_label = Variable(test_label).cuda()

            test_data = test_data.permute(0, 1, 3, 2)         # (571551, 1, 288, 1)-->(571551, 1, 1, 288)
            test_data = torch.squeeze(test_data, -1)          # 改为test
            test_label = torch.unsqueeze(test_label, dim=-2)  # (571551, 288, 7)-->(571551, 288, 1, 7)
            # test_label = test_label.permute(0, 3, 2, 1)       # (571551, 288, 1, 7)-->(571551, 7, 1, 288)
            # test_label = torch.argmax(test_label, 1)          # (571551, 7, 1, 288)-->(571551, 1 ,1, 288)
            pred_label = model(test_data)
            _, predicted = torch.max(pred_label[6].data, 1)   # (20, 1 ,288)
            
            temp = predicted
            
            predicted_label = np_extend(predicted_label, temp.cpu())
            
            
            l_weight = torch.tensor(l_weight, requires_grad=True)
            l_weight = Variable(l_weight).cuda()
            criterion = nn.CrossEntropyLoss(l_weight)
            test_loss = criterion(pred_label[6], test_label.long())  # label和data的size不一样
            test_loss_avg.append(test_loss.item())
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()

            test_loss = np.array(test_loss_avg).mean()
            pred_label[6] = pred_label[6].view(-1,n_classes)

            pre_mask = torch.zeros(pred_label[6].size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0) 
            test_label = test_label.long()
            
            tar_mask = torch.zeros(pred_label[6].size()).scatter_(1, test_label.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)  
        recall = acc_num / target_num
        precision = acc_num / predict_num + float('1e-8')
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
        print('Test Acc {}'.format(accuracy))
        print('recall {}'.format(recall))
        print('precision {}'.format(precision))
        print('F1-score {}'.format(F1))

    
        print('Test loss:', test_loss)
        print('Accuracy of the network:', (100*(correct / (total*288))))
        print('Begin Writing!')
        row_len = len(predicted_label)
        
        with open(output_dir+'test_pred_5_12_1_label.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for i in range(row_len):
                data_list = predicted_label[i]
                
                writer.writerow(data_list[0])
        return correct


def main(args):
    stime = time.time()
    num_epoch = 5
    Loss_list = []
    accuracy = []

    # checkpoint_dir = r'D:\Pycharm Projects\DexiNed_horizon\checkpoints\180_model.pth'    # 生成的model地址

    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    model = Diformer().to(device)
    # checkpoint_path = checkpoint_dir
    if not args.is_testing:
        print('*********is training *********')
        # model.load_state_dict(torch.load(checkpoint_path,
        #                                  map_location=device))

        # Data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy') # 这里改成了训练测试数据
        # Label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')

        Data = np.load(r'D:\Pycharm Projects\Horizon_Picking\9_20_patch_reshape_back.npy')
        Label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        
        print(Data.shape, Label.shape)
        Data = Data.reshape(-1, 1, 288)
        Label = Label.reshape(-1, 1, 288)
        trainData = Data[::1000, :, :]
        trainLabel = Label[::1000, :, :]
        trainLabel = trainLabel[:568, :, :]
        print(trainData.shape, trainLabel.shape, '*********************')
        train_dataset = MyDataset(trainData, trainLabel)
        val_data = Data[::50]
        val_label = Label[::50]
        val_dataset = MyDataset(val_data, val_label)
        train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
        val_loader = DataLoader.DataLoader(dataset=val_dataset,batch_size=10, shuffle=True)

    if args.is_testing:
        print('_______ is testing _______')
        # 测试集数据地址Wu
        TestData = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        TestLabel = np.load(r'D:\Pycharm Projects\DexiNed_horizon\output\test_label_no_ohe.npy')
        test_dataset = TestDataset(TestData, TestLabel)
        test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=300, shuffle=False)
        # 预测输出地址Dexined
        output_dir = 'D:/Pycharm Projects/DexiNed_horizon/output/'
        # test(checkpoint_path, test_loader, model, device, output_dir)
        return  # 如果没有这个空return就会报错

    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 4.2]
    l_weight = torch.tensor(l_weight, requires_grad=True)
    l_weight = Variable(l_weight).cuda()
    criterion = nn.CrossEntropyLoss(l_weight)
    # torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1, last_epoch=-1)
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')  和pytorch原生的torch.cuda.amp一个性质之不过是插件
    seed = 1021

    for epoch in range(num_epoch):  # 更改训练轮数
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)  # lr=0.001
        optimizer = optim.AdamW(model.parameters(),
                               lr=0.01,  # 改一下
                               weight_decay=1e-4)
        if epoch % 20 == 0:
            seed = seed + 500
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("------ Random seed applied-------------")
        output_dir_epoch = r'D:\Pycharm Projects\DexiNed_horizon\checkpoints'
        os.makedirs(output_dir_epoch, exist_ok=True)

        avg_loss, accuracy = train_one_epoch(epoch,
                                   train_loader,
                                   model,
                                   criterion,
                                   optimizer,
                                   )

        # y1, y2 = avg_loss, accuracy
        # Loss_list.append(y1)
        # accuracy.append(y2)

        validate_one_epoch(val_loader, model)

        # y1, y2 = avg_loss, accuracy
        # Loss_list.append(y1)
        # accuracy.append(y2)



        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))

    # if not args.is_testing:
    #     x1 = range(0, num_epoch) # 和训练轮数一样
    #     y1 = Loss_list
    #     plt.plot(x1, y1, '.-')
    #     plt.xlabel('Test loss vs. epoches')
    #     plt.ylabel('Test loss')
    #     plt.figure(1)
    #     plt.show()
    #     plt.savefig("Test loss.jpg")
    #
    #     x2 = range(0, num_epoch)  # 和训练轮数一样
    #     y2 = accuracy
    #     # print(y2)
    #     plt.plot(x2, y2, '.-')
    #     plt.xlabel('accuracy vs. epoches')
    #     plt.ylabel('accuracy ')
    #     plt.figure(2)
    #     plt.show()
    #     plt.savefig("accuracy.jpg")
    #     # print(type(accuracy))
    #     # accuracy = [i for j in range(len(accuracy)) for i in accuracy[j]]
    # accuracy = sum(accuracy,[])
    # print(accuracy, '\n', Loss_list)
    # endtime = time.time()
    # print(endtime-stime)

def parse_args():
    parser = argparse.ArgumentParser(description='DexHorizon_trainer.')
    is_testing = False  #  默认是Ture 测试模式 False 训练
    parser.add_argument('--is_testing', type=bool,
                        default=is_testing,
                        help='Script in testing mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


