import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from models.diformer_patch import Diformer this is not the problem of model it's about training and validation
from models.DOD_ensemble import DexiNed
from models.fusion_model import UNetFusionModel
from utils.datafactory import HorizonDataFactory
from utils.tools import EarlyStopping

class FeatureFusionModel(nn.Module):
    """
    Final model to fuse features from meta-models
    """
    def __init__(self, total_feature_dim: int, num_classes: int, fusion_height, fusion_width):
        super().__init__()
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # map the last layer for final prediction
            nn.Linear(256, num_classes * fusion_height * fusion_width) 
        )
    
    def forward(self, fused_features):
        
        x = self.fusion_network(fused_features)
        
        return x.view(-1, self.num_classes, self.height, self.width)



class AdvancedEnsembleLearner:
    """
    Advanced Ensemble Learning with Feature Fusion
    """
    def __init__(self, dim: List[Dict], num_heads: List[Dict], meta_model_path, fusion_model_path, num_classes: int, num_classifiers:int, height: int = 288, width: int = 1, patience: int=7):
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.mmp = meta_model_path
        self.fmp = fusion_model_path
        self.patience = patience
        
        # self.classifiers = nn.ModuleList([
        #     Diformer(
        #         dim=dim, 
        #         num_heads=num_heads,
        #         feature_projection_dim=288 # define projection dim regarding model_patch
        #     ) for _ in range(num_classifiers)
        # ])
        
        self.classifiers = nn.ModuleList([
            DexiNed() for _ in range(num_classifiers)
        ])
    
        # total_feature_dim = num_classifiers * self.classifiers[0].feature_fuse_projection.out_features
        total_feature_dim = num_classifiers * 7

        self.fusion_model = UNetFusionModel(
            total_feature_dim=total_feature_dim, 
            num_classes=num_classes,
            fusion_height=height,
            fusion_width=width
        )
        
        # Classifier weights
        # self.classifier_weights = torch.ones(len(self.classifiers)) / len(self.classifiers)
        
        
    def train_meta_models(self, classifier, train_loader, optimizer, criterion, attr_name, epoch):
        classifier.train()
        total_loss = 0
        accuracy = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_y = torch.squeeze(batch_y.long())
            with torch.cuda.amp.autocast():
                outputs, _ = classifier(batch_x)
                loss = criterion(outputs[6], batch_y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                _, predicted = torch.max(outputs[6], 1)
                correct = (predicted == batch_y).sum().item()
                accuracy.append(correct / batch_y.numel())
        print(f"Meta-Model {attr_name} - Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader)} Training Acc: {np.mean(accuracy)*100:.2f}%")

    def val_meta_models(self, classifier, val_loader, criterion, attr_name, epoch, early_stopping, save_path):
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
        print(f"Meta-Model {attr_name} - Epoch {epoch+1}: Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {100. * correct / total:.2f}%")
        early_stopping(-val_loss, classifier, save_path, attr_name, stage_name='meta')

    def train_fusion_model(self, attribute_dataloaders, optimizer, criterion):
        self.fusion_model.train()
        all_features = []
        all_labels = []
        for classifier, (attr_name, train_loader) in zip(self.classifiers, attribute_dataloaders):
            for batch_x, batch_y in train_loader:
                features = classifier(batch_x, extract_features=True)
                all_features.append(features)
                all_labels.append(batch_y)
        total_features = torch.cat(all_features, dim=1)
        final_labels = torch.cat(all_labels, dim=0)
        optimizer.zero_grad()
        final_labels = torch.squeeze(final_labels)
        fusion_outputs = self.fusion_model(total_features)
        loss = criterion(fusion_outputs, final_labels.long())
        loss.backward()
        optimizer.step()
        return loss

    def validate_fusion_model(self, validation_dataloaders, criterion, fm_path, early_stopping):
        
        self.fusion_model.eval()
        val_features = []
        val_labels = []
        with torch.no_grad():
            for classifier, (attr_name, val_loader) in zip(self.classifiers, validation_dataloaders):
                for batch_x, batch_y in val_loader:
                    features = classifier(batch_x, extract_features=True)
                    val_features.append(features)
                    val_labels.append(batch_y)
                    
        total_val_features = torch.cat(val_features, dim=1)
        final_val_labels = torch.cat(val_labels, dim=0)
        final_val_labels = torch.squeeze(final_val_labels)
        val_outputs = self.fusion_model(total_val_features)
        
        val_loss = criterion(val_outputs, final_val_labels.long())
        _, predicted = val_outputs.max(1)
        total = final_val_labels.size(0)
        correct = predicted.eq(final_val_labels).sum().item()
        
        early_stopping(-val_loss, self.fusion_model, fm_path, attr_name, stage_name='fusion')
        
        
        return {"loss": val_loss.item(), "accuracy": 100. * correct / total}

    
    def train_ensemble(self, 
                       attribute_dataloaders: List, 
                       validation_dataloaders: List, 
                       epochs: int = 2, 
                       learning_rate: float = 1e-1                       
                       ):
        """
        Train meta-models and fusion model
        """
        
        # Individual optimizers for classifiers and fusion model
        classifier_optimizers = [
            optim.AdamW(classifier.parameters(), lr=learning_rate) 
            for classifier in self.classifiers
        ]
        fusion_optimizer = optim.AdamW(self.fusion_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Init early stopping
        mm_path = os.path.join(self.mmp)
        fm_path = os.path.join(self.fmp)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        for epoch in range(epochs):
            # Stage 1: Individual meta-models for each attribute 
            for idx, (classifier, ((attr_name, train_loader), (attr_name, val_loader)), optimizer) in enumerate(zip(self.classifiers, zip(attribute_dataloaders, validation_dataloaders), classifier_optimizers)):
                
                self.train_meta_models(classifier, train_loader, optimizer, criterion, attr_name, epoch)
               
                self.val_meta_models(classifier, train_loader,  criterion, attr_name,  epoch, early_stopping, mm_path)
                
            print('\n')     
            # Stage 2: Fusion model
            # Stage 2.1: Train funsion model
            self.fusion_model.train()
            
            fusion_loss = self.train_fusion_model(attribute_dataloaders, fusion_optimizer, criterion)
            
            fusion_val_metrics = self.validate_fusion_model(validation_dataloaders, criterion,fm_path,early_stopping)
            
            print(f"Fusion Model Epoch {epoch+1}: Loss: {fusion_loss.item()} Val Loss: {fusion_val_metrics['loss']:.4f}, Val Acc: {fusion_val_metrics['accuracy']:.4f}%")

           
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
    embed_dims = [16, 8, 8, 8]
    heads = 2
    
    # Initialize Advanced Ensemble Learner
    ensemble_learner = AdvancedEnsembleLearner(
        dim=embed_dims,
        num_heads=heads,
        num_classes=7,
        num_classifiers=len(attribute_dataloaders),
        height=args.height,
        width=args.width,
        meta_model_path=args.mm_ckpt_path,
        fusion_model_path=args.fm_ckpt_path
    )
    
    if args.is_training:
        print("============ Begin training ============\n")  
        # Train ensemble
        ensemble_learner.train_ensemble(attribute_train_loaders, attribute_val_loaders, epochs=args.num_epochs)
        
        return 
    
    if args.is_testing:
        print("============ Begin testing ============\n") 
        # Predict
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
    
    
