import argparse
import os
# import csv
import time
from typing import Dict, List

import numpy as np
import torch
# import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
# from skimage.measure import compare_ssim as ssim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from models.DOD_ensemble import DexiNed
from models.fusion_model import UNetFusionModel
from utils.datafactory import HorizonDataFactory
from utils.tools import EarlyStopping


class Ensemblemodel:
       
    def train_meta_model(model, train_loader, optimizer,  patch_size):
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

    def validate_meta_model(flag,  model, val_loader, device, patch_size):

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

    def train_fusion_model(self, attribute_dataloaders, fusion_optimizer, criterion):
        """
        Basically speaking, we use the train function to train the patched data.
        """
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

    def validate_fusion_model(self, validation_dataloaders,criterion):

        """
        Accordingly, the validation is used to verify the training results.
        By saving the model who has the least validation loss.
        """
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



def main():
    stime = time.ctime()
    # freq phase seismic dip amp
    # attribute_names = ['freq', 'phase', 'seismic', 'dip', 'amp', 'complex', 'coherence','average_zero','azimuth']
    # attribute_names = ['seismic', 'dip', 'amp', 'complex', 'coherence','average_zero','azimuth']
    attribute_names = ['seismic', 'dip']
    
    data_factory = HorizonDataFactory(attr_dirs=args.attr_dirs, kernel_size=(1, 288, 16), stride=(1, 16, 32), batch_size=args.batch_size) # the resulting 
    
    attribute_dataloaders = data_factory.get_dataloaders(attribute_names)
    
    attribute_keys = list(attribute_dataloaders.keys())  # Extract keys from the dictionary
    
    train_values = [loaders["train"] for _, loaders in attribute_dataloaders.items()]  # Extract corresponding "train" loaders
    attribute_train_loaders = list(zip(attribute_keys, train_values))
    
    val_values = [loaders["val"] for _, loaders in attribute_dataloaders.items()]  # Extract corresponding "train" loaders
    attribute_val_loaders = list(zip(attribute_keys, val_values))
    
    test_values = [loaders["test"] for _, loaders in attribute_dataloaders.items()]  # Extract corresponding "train" loaders
    attribute_test_loaders = list(zip(attribute_keys, test_values))
    
    # embed_dims = [72, 36, 36, 36]
    # embed_dims = [4, 2, 2, 2]
    # embed_dims = [16, 8, 8, 8]
    # heads = 2
    
    model = Ensemblemodel(
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        num_meta_models=args.num_meta_models,
        meta_model_path='path/to/save/meta_models',
        fusion_model_path='path/to/save/fusion_model',
        patience=5  # Early stopping patience
    )
    
    if args.is_training:
        print("============ Begin training ============\n")  
        # Train ensemble
        model.train_ensemble(attribute_train_loaders, attribute_val_loaders, epochs=args.num_epochs)
        
        return 
    
    if args.is_testing:
        print("============ Begin testing ============\n") 
        # Predict
        ensemble_learner = Ensemblemodel()
        predictions = ensemble_learner.predict(attribute_test_loaders)
        pred_results = np.save(predictions)
        return pred_results    
    
    etime = time.ctime()
    print(stime, etime)
       


def parse_args():
    parser = argparse.ArgumentParser(description='Diformer_trainer.')
    
    parser.add_argument('--is_training', type=bool, default=True, 
                        help='Script in training mode')
    
    parser.add_argument('--num_epochs', type=int, default=3, 
                        help='Overall training epochs')
    
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Training batch size')
    
    parser.add_argument('--height', type=int, default=288, 
                        help='data height size')
    
    parser.add_argument('--width', type=int, default=16, 
                        help='data width size')

    parser.add_argument('--mm_ckpt_path', type=str, default='/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/process/output/meta_model_ckpt', 
                        help='checkpoint saving/loading path of meta model')
    
    parser.add_argument('--fm_ckpt_path', type=str, default='/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/process/output/fusion_model_ckpt', 
                        help='checkpoint saving/loading path of fusion model')

    parser.add_argument('--is_testing', type=bool, default=False,
                        help='Script in testing mode')
    
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='device configuration')
    
    parser.add_argument('--embed_dims', type=list, default=[16, 8, 8, 8],
                        help='Script in testing mode')
    
    parser.add_argument('--heads', type=int,  default=2,
                        help='Script in testing mode')
    
    parser.add_argument('--num_epoch', type=int,  default=20,
                        help='Overall training epochs')
    
    parser.add_argument('--training_dir', type=str,  default='./process/training',
                        help='training log dir')
    
    parser.add_argument('--output_dir', type=str,  default='./process/output',
                        help='output log dir')
    
    parser.add_argument('--attr_dirs', type=dict,  default = {
        # "freq": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy", 
        #          "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "phase": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_phase.npy", 
                #   "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        "seismic": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy", 
                    "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        "dip": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_predict_MCDL_crossline.npy", 
                "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "amp": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_amp.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "complex": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_complex_trace.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "coherence": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_coherence.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "average_zero": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_Average_zero_crossing.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "azimuth": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_Azimuth.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"}
        
        # ['seismic', 'dip', 'amp', 'complex', 'coherence','average_zero','azimuth']
        
    }, help='attr names and paths')
    
    args = parser.parse_args()
        
    return args

if __name__ == '__main__':
    args = parse_args()
    main()
