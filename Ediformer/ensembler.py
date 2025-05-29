# Multi-Attribute Ensemble Model for Seismic Horizon Picking
# 
# Quick Start
# 1. Prepare your data:
#    - Configure data paths in --attr_dirs parameter
#    - Required files for each attribute: data file (.npy) and label file (.npy)
#    - Default attributes: seismic, dip (configurable in attribute_names)
#
# 2. Training:
#    ```bash
#    python Diformer_final/ensembler.py --is_training True --num_epochs 20
#    ```
#
# 3. Testing:
#    ```bash
#    python Diformer_final/ensembler.py --is_testing True
#    ```
#
# Output
# - Meta-models: Individual Diformer models trained on each attribute
# - Fusion model: Combines features from all meta-models
# - Predictions: Final horizon picks with improved accuracy
# All models saved to specified checkpoint directories (--mm_ckpt_path, --fm_ckpt_path).

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.diformer import Diformer
from models.memfusion import MemoryEfficientUNetFusion
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
        
        self.classifiers = nn.ModuleList([
            Diformer(
                dim=dim, 
                num_heads=num_heads,
                feature_projection_dim=16 # define projection dim regarding model_patch
            ) for _ in range(num_classifiers)
        ])
        
        # total_feature_dim = sum(
        #     classifier.feature_projection.out_features  # 288 aligned with feature projection dim
        #     for classifier in self.classifiers
        # )
        
        # total_feature_dim = num_classifiers * self.classifiers[0].feature_projection.out_features
        total_feature_dim = num_classifiers * self.classifiers[0].feature_fuse_projection.out_features

        self.fusion_model = MemoryEfficientUNetFusion(
            total_feature_dim=total_feature_dim, 
            num_classes=num_classes,
            fusion_height=height,
            fusion_width=width
        )
        
        # Classifier weights
        self.classifier_weights = torch.ones(len(self.classifiers)) / len(self.classifiers)
    
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
            # Stage 1: Individual meta-models
            for idx, (classifier, ((attr_name, train_loader), (attr_name, val_loader)), optimizer) in enumerate(zip(self.classifiers, zip(attribute_dataloaders, validation_dataloaders), classifier_optimizers)):
            
                # Stage 1.1 meta-models Training
                classifier.train()
                total_loss = 0
                stage_name = 'meta'
                
                for batch_x, batch_y in train_loader:
                    # print(f"Training {attr_name} with dataloader")
                    optimizer.zero_grad()
                    # squeeze the first dimension to calculate loss
                    batch_y = torch.squeeze(batch_y.long())
                    # Compute loss for individual classifier batch_x should have the shape of 288 as height and others as width
                    outputs = classifier(batch_x) 
                    # outputs / projected features shape torch.Size([16, 7, 64, 288])
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                print(f"Meta-Model {attr_name} - Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader)}")
                
                # Stage 1.2 meta-models Validation
                classifier.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_y = torch.squeeze(batch_y.long())
                        outputs = classifier(batch_x) # outputs: torch.Size([16, 7, 64, 288])
                        loss = criterion(outputs, batch_y.long())
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += batch_y.size(0)
                        correct += predicted.eq(batch_y).sum().item()
                
                print(f"Meta-Model {attr_name} - Epoch {epoch + 1}: Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * correct / (total * batch_x.shape[2] * batch_x.shape[3]):.4f}%")
                # attr_name
                early_stopping(-val_loss, classifier, mm_path, attr_name, stage_name=stage_name)     
                
            print('\n')     
            # Stage 2: Fusion model
            # Stage 2.1: Train funsion model
            self.fusion_model.train()
            all_features = []
            all_labels = []
            stage_name = 'fusion'
            
            # Extract features from meta-models
            for classifier, ((attr_name, train_loader), (attr_name, val_loader)) in zip(self.classifiers, zip(attribute_dataloaders, validation_dataloaders)):
            
            # Extract features from meta-models
            # for classifier, train_loader in zip(self.classifiers, attribute_dataloaders):
                classifier_features = []
                classifier_labels = []
                
                for batch_x, batch_y in train_loader:

                    features = classifier(batch_x, extract_features=True)
                    classifier_features.append(features)
                    classifier_labels.append(batch_y)
                
                all_features.append(torch.cat(classifier_features, dim=0))
                all_labels.append(torch.cat(classifier_labels, dim=0))
            
            # Concatenate features from all meta-models concatenate 2nd dimension to map channel into probs
            total_features = torch.cat(all_features, dim=1)
            
            final_labels = all_labels[0]  # Assume consistent labels across attributes
            
            # Train fusion model
            fusion_optimizer.zero_grad()
            
            # ================
            # Simplify the fusion model for running all the process by letting total features as fusion model outputs
            
            fusion_outputs = self.fusion_model(total_features) # channel num_classifiers * num_class --> num_class 
            
            fusion_outputs = total_features
            # squeeze and convert into long dtype
            final_labels = torch.squeeze(final_labels)
            fusion_loss = criterion(fusion_outputs, final_labels.long()) 
            
            fusion_loss.backward()
            fusion_optimizer.step()
            
            print(f"Fusion Model Epoch {epoch+1}: Loss: {fusion_loss.item()}")
            
            # Stage 2.2: Funsion model validation
            self.fusion_model.eval()
            with torch.no_grad():
                val_features = []
                val_labels = []
                attr_name_full = []
                for classifier, (attr_name, val_loader) in zip(self.classifiers, (validation_dataloaders)):
                # for classifier, val_loader in zip(self.classifiers, validation_dataloaders):
                    classifier_features = []
                    classifier_labels = []
                    
                    for batch_x, batch_y in val_loader:
                        # features = classifier(batch_x, extract_features=True)
                        features = classifier(batch_x)
                        classifier_features.append(features)
                        classifier_labels.append(batch_y)
                    
                    val_features.append(torch.cat(classifier_features, dim=0))
                    val_labels.append(torch.cat(classifier_labels, dim=0))
                    # generate entire output
                    attr_name_full.append(attr_name)
                
            # Concatenate validation features
            total_val_features = torch.cat(val_features, dim=1)
            final_val_labels = val_labels[0]  # Assume consistent labels across attributes
            
            val_outputs = self.fusion_model(total_val_features)
            val_outputs = total_val_features
            final_val_labels = torch.squeeze(final_val_labels)
            val_loss = criterion(val_outputs, final_val_labels.long())
            
            _, predicted = val_outputs.max(1)
            total = final_val_labels.size(0)
            correct = predicted.eq(final_val_labels).sum().item()
            
            print(f"Fusion Model Epoch {epoch+1}, Val Loss: {val_loss.item():.4f}, Val Acc: {100. * correct / (total * batch_x.shape[2] * batch_x.shape[3]):.4f}%")
            attr_name = '_'.join(attr_name_full)
            early_stopping(-val_loss, self.fusion_model, fm_path, attr_name, stage_name)   
            print('\n')   
            
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
    
    
    data_factory = HorizonDataFactory(attr_dirs=args.attr_dirs, kernel_size=(1, 288, 16), stride=(1, 64, 32), batch_size=args.batch_size) # the resulting 
    
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
    
    