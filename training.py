import torch
import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from torch import nn
from model_trace_origin import Diformer
from utils.extend import np_extend
import numpy as np
import time


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, tb_writer):
    correct_total = 0
    total = 0
    correct = 0

    model.train()
    loss_avg = []
    Loss_list = []
    accuracy = []
    
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


def validate(model, val_loader, criterion):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(val_loader), total=int(len(val_loader.val_dataset) / val_loader.batch_size))
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
