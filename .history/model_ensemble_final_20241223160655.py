import os
# import csv
import time
import torch
# import imageio
# import datetime
import argparse
import numpy as np
# import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from DOD import DexiNed
from models.DOD_ensemble import DexiNed
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
# from skimage.measure import compare_ssim as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from ignite.metrics import MeanSquaredError, SSIM, PSNR

def generate_crossline(data):
    """return: dataset after patched with shape of 601*N, 1, kh, kw where N is decided by kh and kw"""
    kc, kh, kw = 1, 288, 16
    dc, dh, dw = 1, 288, 16

    data = data[np.newaxis,]
    data = torch.tensor(data).clone().detach()
    data = F.pad(data, [
                        data.size(3) % kw // 2, data.size(3) % kw // 2,
                        data.size(2) % kh // 2, data.size(2) % kh // 2,
                        data.size(1) % kc // 2, data.size(1) % kc // 2])
    data = data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    data = data.contiguous().view(-1, kc, kh, kw)

    data = data.reshape(601, -1, kh, kw)[::20, ::2]

    data = data.reshape(-1, 1, kh, kw)

    return data


def generate_inline(data):
    """return: dataset after patched with shape of 601*N, 1, kh, kw where N is decided by kh and kw
    only difference between the crossline is to swap axes to generate the inline data
    """

    kc, kh, kw = 1, 288, 16
    dc, dh, dw = 1, 288, 16
    '''By swapping the dimensions below we obtain the inline slices'''
    data = np.swapaxes(data,-1, 0)
    data = data[np.newaxis,]
    data = torch.tensor(data).clone().detach()
    data = F.pad(data, [data.size(3) % kw // 2, data.size(3) % kw // 2,
                        data.size(2) % kh // 2, data.size(2) % kh // 2,
                        data.size(1) % kc // 2, data.size(1) % kc // 2])
    data = data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    data = data.contiguous().view(-1, kc, kh, kw)
    data = data.reshape(451, -1, kh, kw)[::30, ::3]
    data = data.reshape(-1, 1, kh, kw)

    return data

def generate_test(data):
    """return: dataset after patched with shape of 601*N, 1, kh, kw where N is decided by kh and kw"""
    kc, kh, kw = 1, 288, 16
    dc, dh, dw = 1, 288, 16
    data = data[np.newaxis,]
    data = torch.tensor(data).clone().detach()
    data = F.pad(data, [data.size(3) % kw // 2, data.size(3) % kw // 2,
                        data.size(2) % kh // 2, data.size(2) % kh // 2,
                        data.size(1) % kc // 2, data.size(1) % kc // 2])
    data = data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    data = data.contiguous().view(-1, kc, kh, kw)
    # data = data.reshape(601, -1, kh, kw)
    # print(data.shape)
    data = data.reshape(-1, 1, kh, kw)
    # data = data[::100, ]
    # print(data.shape)
    return data

def np_extend(a, b, axis=0):
    if a is None:
        return b
    else:
        return np.concatenate((a, b), axis=axis)

def generate_val_crossline(data):
    """return: dataset after patched with shape of 601*N, 1, kh, kw where N is decided by kh and kw"""
    kc, kh, kw = 1, 288, 16
    dc, dh, dw = 1, 288, 16
    # data = data[100:110, :, :640]
    print(data.shape)
    data = data[np.newaxis,]
    data = torch.tensor(data).clone().detach()
    data = F.pad(data, [data.size(3) % kw // 2, data.size(3) % kw // 2,
                        data.size(2) % kh // 2, data.size(2) % kh // 2,
                        data.size(1) % kc // 2, data.size(1) % kc // 2])
    data = data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    data = data.contiguous().view(-1, kc, kh, kw)

    data = data.reshape(601, -1, kh, kw)[::20][::5] # here we change into 10 due to the limitation
    data = data.reshape(-1, 1, kh, kw)

    return data


def generate_val_inline(data):
    """return: dataset after patched with shape of 601*N, 1, kh, kw where N is decided by kh and kw
    only difference between the crossline is to swap axes to generate the inline data
    """

    kc, kh, kw = 1, 288, 16
    dc, dh, dw = 1, 288, 16

    '''By swapping the dimensions below we obtain the inline slices'''
    data = np.swapaxes(data, -1, 0)
    # data = data[100:110, :, :576]
    data = data[np.newaxis, ]
    data = torch.tensor(data).clone().detach()
    data = F.pad(data, [data.size(3) % kw // 2, data.size(3) % kw // 2,
                        data.size(2) % kh // 2, data.size(2) % kh // 2,
                        data.size(1) % kc // 2, data.size(1) % kc // 2])
    data = data.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    data = data.contiguous().view(-1, kc, kh, kw)
    data = data.reshape(451, -1, kh, kw)[::20][::5] # here we change to 10 due to the limitation
    data = data.reshape(-1, 1, kh, kw)

    return data

# def calculate_metrics(pred, target, num_classes):
#     """
#     Calculates PSNR, SSIM, MSE, and DICE coefficient for each class in a multiclass segmentation task.
#     """
#     psnr_metric = PSNR(data_range=1.0)
#     ssim_metric = SSIM(data_range=1.0)
#     mse_metric = MeanSquaredError()
#
#     psnrs = []
#     ssims = []
#     mses = []
#     dices = []
#
#     for i in range(num_classes):
#         pred_i = (pred == i).float()
#         target_i = (target == i).float()
#         # print(pred_i.shape, target_i.shape)
#         mse_metric.update((pred_i, target_i))
#         psnr_metric.update((pred_i, target_i))
#         ssim_metric.update((pred_i, target_i))
#
#         intersection = (pred_i * target_i).sum()
#         dice_i = (2. * intersection) / (pred_i.sum() + target_i.sum())
#
#         mses.append(mse_metric.compute())
#         psnrs.append(psnr_metric.compute())
#         ssims.append(ssim_metric.compute())
#         dices.append(dice_i.item())
#
#     return psnrs, ssims, mses, dices


class dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx])
        label = torch.Tensor(self.label[idx])

        return data, label

    def __len__(self):
        return len(self.data)

    # def train_meta_models(self, classifier, train_loader, val_loader, optimizer, criterion, attr_name, epoch, early_stopping, mm_path):
    #     """Train individual meta-models"""
    #     # Training phase
    #     classifier.train()
    #     total_loss = 0
    #     accuracy = []
        
    #     for batch_x, batch_y in train_loader:
    #         optimizer.zero_grad()
    #         batch_y = torch.squeeze(batch_y.long())
            
    #         with torch.cuda.amp.autocast():
    #             outputs, _ = classifier(batch_x)
    #             loss = criterion(outputs[6], batch_y)
    #             loss.backward()
    #             optimizer.step()
                
    #         total_loss += loss.item()
            
    #         with torch.no_grad():
    #             _, predicted = torch.max(outputs[6], 1)
    #             correct = (predicted == batch_y).sum().item()
    #             accuracy.append(correct / (batch_x.size(0) * batch_x.size(-1) * 288))
        
    #     train_loss = total_loss / len(train_loader)
    #     train_acc = np.array(accuracy).mean() * 100
        
    #     # Validation phase
    #     val_metrics = self.validate_meta_model(classifier, val_loader, criterion)
        
    #     print(f"Meta-Model {attr_name} - Epoch {epoch+1}:")
    #     print(f"Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}%")
    #     print(f"Validation Loss: {val_metrics['loss']:.4f}, Validation Acc: {val_metrics['accuracy']:.4f}%")
        
    #     early_stopping(-val_metrics['loss'], classifier, mm_path, attr_name, stage_name='meta')
        
    #     return train_loss, train_acc, val_metrics
    
    def train_meta_models(model, train_loader, optimizer,  patch_size):
        """
        Basically speaking, we use the train function to train the patched data.
        """
        model.train()
        total_loss = 0
        accuracy = []
        l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]

        l_weight_tensor = torch.tensor(l_weight, requires_grad=False).cuda()
        scaler = torch.cuda.amp.GradScaler()

        for data, target in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            target = torch.squeeze(target.long())
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output, _ = model(data) # remember the mode generates two
                criterion = nn.CrossEntropyLoss(weight=l_weight_tensor)
                loss = criterion(output[6], target)

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss_avg.append(loss.item())

            with torch.no_grad():
                _, predicted = torch.max(output[6], 1)
                correct = (predicted == target).sum().item()
                accuracy.append(correct /(data.size(0)*patch_size*288))
                
        train_loss = total_loss / len(train_loader)
        train_acc = np.array(accuracy).mean() * 100

        return train_loss, train_acc

    def validate_meta_model(self, classifier, val_loader, criterion):
        """Validate individual meta-models"""
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_y = torch.squeeze(batch_y.long())
                outputs, _ = classifier(batch_x)
                loss = criterion(outputs[6], batch_y.long())
                val_loss += loss.item()
                
                _, predicted = outputs[6].max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': 100 * correct / (total * batch_x.shape[2] * batch_x.shape[3])
        }
        return metrics
    
    def validate(flag,  model, val_loader, device, patch_size):

        """
        Accordingly, the validation is used to verify the training results.
        By saving the model who has the least validation loss.
        """
        model.eval()
        
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in enumerate(val_loader):
                
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                batch_y = torch.squeeze(batch_y.long(),dim=1)
                criterion = nn.CrossEntropyLoss()
                outputs, _ = model(batch_x)

                loss = criterion(outputs[6], batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs[6].data, 1)
                
                total += batch_y.size(0)
                correct = (predicted == batch_y).sum().item()
                
        metrics = {
            'loss': val_loss / len(val_loader),
            'accuracy': 100 * correct / (total * batch_x.shape[2] * batch_x.shape[3])
        }

        return metrics

    def train_fusion_model(self, attribute_dataloaders, fusion_optimizer, criterion, epoch):
        """Train fusion model"""
        self.fusion_model.train()
        all_features = []
        all_labels = []
        
        # Extract features from meta-models
        for classifier, (attr_name, train_loader) in zip(self.classifiers, attribute_dataloaders):
            classifier_features = []
            classifier_labels = []
            
            with torch.no_grad():  # Don't accumulate gradients for feature extraction
                for batch_x, batch_y in train_loader:
                    features = classifier(batch_x, extract_features=True)
                    classifier_features.append(features)
                    classifier_labels.append(batch_y)
            
            all_features.append(torch.cat(classifier_features, dim=0))
            all_labels.append(torch.cat(classifier_labels, dim=0))
        
        total_features = torch.cat(all_features, dim=1)
        final_labels = torch.squeeze(all_labels[0])
        
        # Train fusion model
        fusion_optimizer.zero_grad()
        fusion_outputs = self.fusion_model(total_features)
        fusion_loss = criterion(fusion_outputs, final_labels.long())
        fusion_loss.backward()
        fusion_optimizer.step()
        
        return fusion_loss.item()

    def validate_fusion_model(self, validation_dataloaders, criterion):
        """Validate fusion model"""
        self.fusion_model.eval()
        val_features = []
        val_labels = []
        attr_names = []
        
        with torch.no_grad():
            for classifier, (attr_name, val_loader) in zip(self.classifiers, validation_dataloaders):
                classifier_features = []
                classifier_labels = []
                
                for batch_x, batch_y in val_loader:
                    features = classifier(batch_x, extract_features=True)
                    classifier_features.append(features)
                    classifier_labels.append(batch_y)
                
                val_features.append(torch.cat(classifier_features, dim=0))
                val_labels.append(torch.cat(classifier_labels, dim=0))
                attr_names.append(attr_name)
        
        total_val_features = torch.cat(val_features, dim=1)
        final_val_labels = torch.squeeze(val_labels[0])
        
        val_outputs = self.fusion_model(total_val_features)
        val_loss = criterion(val_outputs, final_val_labels.long())
        
        _, predicted = val_outputs.max(1)
        total = final_val_labels.size(0)
        correct = predicted.eq(final_val_labels).sum().item()
        accuracy = 100. * correct / total
        
        return {
            'loss': val_loss.item(),
            'accuracy': accuracy,
            'attr_name': '_'.join(attr_names)
        }

    def train_ensemble(self, attribute_dataloaders, validation_dataloaders, epochs=2, learning_rate=1e-1):
        """Main training loop"""
        classifier_optimizers = [optim.AdamW(classifier.parameters(), lr=learning_rate) 
                               for classifier in self.classifiers]
        fusion_optimizer = optim.AdamW(self.fusion_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        mm_path = os.path.join(self.mmp)
        fm_path = os.path.join(self.fmp)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Stage 1: Train and validate meta-models
            for classifier, ((attr_name, train_loader), (attr_name, val_loader)), optimizer in \
                zip(self.classifiers, zip(attribute_dataloaders, validation_dataloaders), classifier_optimizers):
                self.train_meta_models(classifier, train_loader, val_loader, optimizer, 
                                     criterion, attr_name, epoch, early_stopping, mm_path)
            
            # Stage 2: Train and validate fusion model
            fusion_loss = self.train_fusion_model(attribute_dataloaders, fusion_optimizer, criterion, epoch)
            fusion_val_metrics = self.validate_fusion_model(validation_dataloaders, criterion)
            
            print(f"\nFusion Model - Epoch {epoch+1}:")
            print(f"Training Loss: {fusion_loss:.4f}")
            print(f"Validation Loss: {fusion_val_metrics['loss']:.4f}, Validation Acc: {fusion_val_metrics['accuracy']:.4f}%")
            
            early_stopping(-fusion_val_metrics['loss'], self.fusion_model, fm_path, 
                         fusion_val_metrics['attr_name'], stage_name='fusion')
                    
    def predict(self, attribute_test_loaders, meta_model_path, fusion_model_path):
        """
        Make predictions using feature fusion
        """
        # Extract features from meta-models
        all_features = []
        
        self.classifiers = torch.load(meta_model_path)
        self.fusion_model = torch.load(fusion_model_path)
        
        for classifier, dataloader in zip(self.classifiers, attribute_test_loaders):
            classifier_features = []
            
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    features = classifier(batch_x, extract_features=True)
                    classifier_features.append(features)
            
            all_features.append(torch.cat(classifier_features, dim=0))
        # Concatenate features
        total_features = torch.cat(all_features, dim=1)
        
        # Final prediction
        with torch.no_grad():
            self.fusion_model.eval()
            return self.fusion_model(total_features)




def test(checkpoint_path, test_loader, model, device, output_dir):
    """
    While, it should be noted that in the test process, we're using the OverlapDataset who will give the overlap
    result of the input test data. To make the results more smoother.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device)
    model.eval()

    test_loss_avg = []
    correct = 0
    total = 0
    num_classes = 7
    target_num = torch.zeros((1, num_classes))
    predict_num = torch.zeros((1, num_classes))
    acc_num = torch.zeros((1, num_classes))
    predicted_label = None
    pred_label_all = None
    pred_label_ensemble_all = None
    true_label_all = None

    with torch.no_grad():
        for batch_id, (test_data, test_label) in enumerate(test_loader):

            test_data = test_data.to(device)
            test_label = test_label.to(device)
            test_data= test_data.reshape((-1,1,288,16))
            test_label = test_label.reshape((-1, 1, 288, 16))
            test_label = torch.squeeze(test_label, dim=1)
            # pred_label,pred_label_ensemble = model(test_data)
            pred_label,_ = model(test_data)
            _, predicted = torch.max(pred_label[6].data, 1)

            # save after max that is integer number
            # predicted_label = np.concatenate(
            #     (predicted_label, predicted.cpu())) if predicted_label is not None else predicted.cpu()

            # save the original feature map after model training this should be sent to the meta model
            pred_label_all = np.concatenate(
                (pred_label_all, pred_label[6].cpu())) if pred_label_all is not None else pred_label[6].cpu()

            # save the train label
            true_label_all = np.concatenate(
                (true_label_all, test_label.cpu())) if true_label_all is not None else test_label.cpu()

            # pred_label_ensemble_all = np.concatenate(
            #     (pred_label_ensemble_all, pred_label_ensemble.cpu())) if pred_label_ensemble_all is not None else pred_label_ensemble.cpu()

            criterion = nn.CrossEntropyLoss()
            test_loss = criterion(pred_label[6], test_label.long())
            test_loss_avg.append(test_loss.item())

            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()

            pred_label[6] = pred_label[6].view(-1, num_classes)

            pre_mask = torch.zeros(pred_label[6].size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            test_label = test_label.long()

            tar_mask = torch.zeros(pred_label[6].size()).scatter_(1, test_label.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)

        np.save("seismic_prediction_feature_maps_data.npy", pred_label_all)
        np.save("seismic_prediction_feature_maps_label.npy", true_label_all)

        

    test_loss = np.mean(test_loss_avg)
    accuracy = 100. * correct / (total * 288 * 16)
    recall = acc_num / target_num
    precision = acc_num / predict_num + float('1e-8')
    F1 = 2 * recall * precision / (recall + precision)


    return test_loss, accuracy, recall, precision, F1



def main():

    # Parse arguments
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DexiNed().to(device)

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    if args.train:
        print('-------------Training------------')

        # checkpoint_path = r'D:\Pycharm Projects\Pytorch_Template\checkpoints\fold_5_epoch_21_model.pth'
        checkpoint_path = r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\output\DFormer_seismic_patch_fold_1_epoch_28_model.pth'
        # model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
        
        # add data normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean) / std

        '''if data is not dip'''
        data = data.reshape((-1, 951, 288))
        data = data[:, 500:, :]
        data = np.swapaxes(data, -1, 1)
        '''if data is dip'''
        print(data.shape) # (601, 288, 451) --> [434, 1, 288, 16]
        data_cross = generate_crossline(data)
        data_in = generate_inline(data)

        train_data = np.concatenate((data_cross, data_in), axis=0)
        # data =data[::2]

        # label = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        label = np.load(r'/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy')
        # label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        label = label.reshape((-1, 951, 288))
        label = label[:, 500:, :]
        label = np.swapaxes(label, -1, 1)

        label_cross = generate_crossline(label)
        label_in = generate_inline(label)
        train_label = np.concatenate((label_cross, label_in), axis=0)
   
        print(data.shape, label.shape)

        train_dataset_augmentated = dataset(train_data,train_label)
        val_crossline_data = generate_val_crossline(data)
        val_crossline_label = generate_val_crossline(label)

        val_inline_data = generate_val_inline(data)
        val_inline_label = generate_val_inline(label)

        val_data = np.concatenate((val_crossline_data, val_inline_data), axis=0)
        val_label = np.concatenate((val_crossline_label, val_inline_label), axis=0)
        print(train_data.shape, train_label.shape, val_data.shape, val_label.shape)
        val_dataset_augmentated = dataset(val_data, val_label)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.9, verbose=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        best_val_loss = float('inf')

        save_fold = 0
        save_epoch = 0
        flag = 0

        for fold, (train_idx, val_idx) in enumerate(kfold.split(val_dataset_augmentated)):
            print('------------fold #---------{}----------------'.format(fold+1), time.ctime())

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_loader = torch.utils.data.DataLoader(train_dataset_augmentated, batch_size=10, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(val_dataset_augmentated, batch_size=10, sampler=val_subsampler)


            for epoch in range(args.num_epochs):
                train_loss, train_acc = train(model, train_loader, optimizer, epoch, args.patch_size)
                val_loss, val_acc = validate(flag,epoch+1, fold+1, model, val_loader, device, args.patch_size)

                # Update the learning rate based on validation loss
                scheduler.step(val_loss)
                # Check if this is the best validation loss so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_fold = fold
                    save_epoch = epoch
                # Print the results for this epoch
                print(f'Epoch [{epoch + 1}/{args.num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f' Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {val_acc:.4f}, '
                      f'Best Val loss: {best_val_loss:.4f}, '
                      f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")
            print('')
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(args.output_dir, 'DFormer_seismic_patch_fold_' + str(save_fold+1) + '_epoch_{}_model.pth'.format(save_epoch + 1)))
        print(f'Best model saved at fold: {save_fold+1} with epoch: {save_epoch + 1}')


    if args.test:
        print('-------------Testing------------')
        # checkpoint_path = r'D:\Pycharm Projects\Pytorch_Template\checkpoints\DFormer_seismic_patch_fold_5_epoch_14_model.pth'
        # checkpoint_path = r'D:\Pycharm Projects\Pytorch_Template\checkpoints\DFormer_seismic_patch_fold_5_epoch_14_model.pth'
        checkpoint_path = r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\output\DFormer_seismic_patch_fold_5_epoch_25_model.pth'

        # test_data = np.load(r'D:\Pycharm Projects\Wu_unet\DL_horizon_demo\data\test_data.npy')
        test_data = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\test_data.npy')

        mean = np.mean(test_data, axis=0)
        std = np.std(test_data, axis=0)
        test_data = (test_data - mean) / std

        '''dip do not need reshape'''
        test_data = test_data.reshape((-1, 951, 288))
        test_data = test_data[300:310]
        test_data = np.swapaxes((test_data), -1, 1)

        test_data = generate_test(test_data)
        test_data = torch.tensor(test_data, dtype=torch.float).clone().detach()
        # test_label = np.load(r'D:\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        test_label = np.load(r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\data\test_label_no_ohe.npy')
        test_label = test_label.reshape((-1, 951, 288))
        test_label = test_label[300:310]
        test_label = np.swapaxes((test_label), -1, 1)
        test_label = generate_test(test_label)

        test_label = torch.tensor(test_label, dtype=torch.float).clone().detach()
        print(test_data.shape, test_label.shape)


        test_dataset = dataset(test_data, test_label)
        test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)
        # np.save('test_labe_for_ensemble.npy', test_label)
        # print('DONE!')

        test_loss, test_acc, test_precision, test_recall, test_F1 = \
            test(checkpoint_path, test_loader, model, device, args.output_dir)

        print('Test Loss: {:.4f},\n'
              'Test Accuracy: {:.2f}%,\n'
              'Recall: {},\n'
              'Precision: {},\n'
              'F1-score: {}'.format(test_loss, test_acc, test_recall, test_precision, test_F1))


def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train or test a neural network')
    parser.add_argument('--train',  type=bool,default=False, help='train the network')
    parser.add_argument('--test',  type=bool,default=False, help='test the network')
    parser.add_argument('--data_dir', type=str, default='data', help='directory containing the data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training and testing')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size for training ')
    parser.add_argument('--stride', type=int, default=32, help='stride for patch in training')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--sample_interval', type=int, default=42, help='data size used to train')
    parser.add_argument('--checkpoint_path', type=str, default=r'D:\Pycharm Projects\Pytorch_Template\checkpoints\31_model.pth', help='checkpoints where you load and save')
    # parser.add_argument('--output_dir', type=str, default=r'D:\Pycharm Projects\Pytorch_Template\ouput', help='output_dir where you save the predicted labels')
    parser.add_argument('--output_dir', type=str, default=r'E:\Cal_HJL\Pycharm Projects\Horizon_Picking\output', help='output_dir where you save the predicted labels')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args.num_epochs = 50
    args.train = True  # switch to testing mode
    # args.test = True  # switch to testing mode
    main()
