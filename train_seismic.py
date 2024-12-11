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
from model import Diformer
from torch import nn

torch.set_printoptions(profile="full")  # 全输出
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
# np.set_printoptions(profile="full")    # 全输出


class MyDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __getitem__(self, idx):
        data = torch.Tensor(self.Data[idx])  
        label = torch.Tensor(self.Label[idx])  
        return data, label

    def __len__(self):
        return len(self.Data)


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
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]  
    scaler = torch.cuda.amp.GradScaler()
    model = model.cuda() 

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    predicted_label = None
    train_tq = tqdm(enumerate(dataloader))
    for batch_id, item in enumerate(dataloader):
        data1, label_a = item  

        data1 = Variable(data1).cuda()

        label_a = Variable(label_a).cuda()

        optimizer.zero_grad()
        with autocast():
            output = model(data1) 
            label = Variable(label_a).cuda()
            l_weight = torch.tensor(l_weight, requires_grad=True)
            l_weight = Variable(l_weight).cuda()
            criterion = nn.CrossEntropyLoss(weight=l_weight)  
            label = torch.squeeze(label)

            # print(label.shape, output[6].shape)

            loss = criterion(output[6], label.long())

            _, output = torch.max(output[6], 1)
            predicted_label = np_extend(predicted_label, output.cpu())
            correct += (output == label).sum().item()

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
    return loss_avg, accuracy  

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
    temp = None  
    model = model.cuda()
    model.eval()
    test_loss_avg = []
    correct = 0
    total = 0
    
    l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 415.8] 

    n_classes = 7
    target_num = torch.zeros((1, n_classes))
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))

    with torch.no_grad():
        for batch_id, item in enumerate(dataloader):
            test_data1, test_label = item  # (571551, 1, 288, 1)  (571551, 288, 7)
            
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

        # if batch_id > 0 and batch_id % 2500 == 0:at
        print('Test loss:', test_loss)
        print('Accuracy of the network:', (100 * (correct / (total * 288))))
        print('Begin Writing!')

        np.save('2_22_freq_train_predicted_label_for_fusion_100_1000.npy', predicted_label)
       
        print(predicted_data.shape)
        
        return correct


def main(args):
    stime = time.ctime()
    num_epoch = 2501
    Loss_list = []
    accuracy = []
    count = 0
    
    training_dir = r'D:\Pycharm Projects\DexiNed_horizon\process'
    output_dir = r'D:\Pycharm Projects\DexiNed_horizon\process'
    
    checkpoint_dir = r'D:\Pycharm Projects\Horizon_Picking\checkpoints\1_25_seismic_1200_model.pth'
    from torch.utils.tensorboard import SummaryWriter  # for torch 1.4 or greather
    tb_writer = SummaryWriter(log_dir=training_dir)
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    model = Diformer().to(device)
    checkpoint_path = checkpoint_dir
    # early_stopping = EarlyStopping()
    if not args.is_testing:
        print('*********is training *********')
        # model.load_state_dict(torch.load(checkpoint_path,  # 要调用model生成的checkpoints
        #                                  map_location=device))
        
        Data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        Label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
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
        
        length = len(train_dataset)
        
        # change the coefficient to change the train dataset size for randomsplit
        train_size, val_size = int(0.8 * length), int(0.2 * length)
        # we didn't applied the randomsplit with train and val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [1500, 269],generator=torch.Generator().manual_seed(0))

        train_loader = DataLoader.DataLoader(dataset=train_dataset, batch_size=50)
        val_loader = DataLoader.DataLoader(dataset=val_dataset, batch_size=50)
       

    if args.is_testing:
        print('_______ is testing _______')
        
        TestData1 = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        TestLabel = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        
        TestData1 = torch.tensor(TestData1, dtype=torch.float)
        TestLabel = torch.tensor(TestLabel, dtype=torch.float)
        TestData1 = TestData1.reshape(-1, 951, 288)  
        # TestData1 = TestData1[:100,]
        TestData1 = TestData1[np.newaxis, :]  
        TestData1 = torch.tensor(TestData1)
        TestData1 = TestData1.permute(0, 1, -1, 2)

        TestLabel = TestLabel.reshape(-1, 951, 288)
        # TestLabel = TestLabel[:100,]
        TestLabel = TestLabel[np.newaxis, :]
        TestLabel = torch.tensor(TestLabel)
        TestLabel = TestLabel.permute(0, 1, -1, 2)
        print(TestData1.shape, TestLabel.shape)
        kc, kh, kw = 1, 288, 64  # padding kernel size
        dc, dh, dw = 1, 64, 64  # padding stride
        
        # no ovlerlapping Pad to multiples of 64
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
        # TestData1= TestData1[100:1000,:]
        # TestLabel = TestLabel[100:1000,:]
        # print(TestData1.shape, TestLabel.shape)
        test_dataset = MyDataset(TestData1, TestLabel)
        test_loader = DataLoader.DataLoader(dataset=test_dataset, batch_size=50, shuffle=False)
    
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


    etime = time.ctime()
    print(stime, etime)


def parse_args():
    parser = argparse.ArgumentParser(description='DiFormer_trainer.')
    is_testing = True  
    parser.add_argument('--is_testing', type=bool,
                        default=is_testing,
                        help='Script in testing mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)


