
from typing import Dict, List

import torch
import argparse
import time
import torch.nn as nn
import torch.optim as optim

from model import Diformer
from utils.datafactory import HorizonDataFactory


class FeatureFusionModel(nn.Module):
    """
    Final model to fuse features from meta-models
    """
    def __init__(self, total_feature_dim: int, num_classes):
        super().__init__()
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, fused_features):
        return self.fusion_network(fused_features)

class AdvancedEnsembleLearner:
    """
    Advanced Ensemble Learning with Feature Fusion
    """
    def __init__(self, dim: List[Dict], num_heads: List[Dict], attribute_configs: List[Dict], num_classes: int = 7, num_classifiers:int = 4):
        self.num_classes = num_classes
        
        self.classifiers = nn.ModuleList([
            Diformer(
                dim=dim, 
                num_heads=num_heads,
                feature_projection_dim=288
            ) for _ in range(num_classifiers)
        ])
        
        # Initialize fusion model we define for 4 but only give 3
        total_feature_dim = sum(
            classifier.feature_projection.out_features  # 288 aligned with feature projection dim
            for classifier in self.classifiers
        )
        self.fusion_model = FeatureFusionModel(
            total_feature_dim=total_feature_dim, 
            num_classes=num_classes
        )
        
        # Classifier weights
        self.classifier_weights = torch.ones(len(self.classifiers)) / len(self.classifiers)
    
    def train_ensemble(self, 
                       attribute_dataloaders: List, 
                       validation_dataloaders: List, 
                       epochs: int = 50, 
                       learning_rate: float = 1e-3):
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
        
        for epoch in range(epochs):
            # Stage 1: Individual meta-models
                
            for idx, (classifier, train_loader, val_loader, optimizer) in enumerate(
                zip(self.classifiers, attribute_dataloaders, validation_dataloaders, classifier_optimizers)
            ):
                # Stage 1.1 meta-models Training
                classifier.train()
                total_loss = 0
                
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    # Compute loss for individual classifier
                    outputs = classifier(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Meta-Model {idx} Epoch {epoch}, Loss: {total_loss/len(train_loader)}")
                
                # Stage 1.2 meta-models Validation
                classifier.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = classifier(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        total += batch_y.size(0)
                        correct += predicted.eq(batch_y).sum().item()
                print(f"Meta-Model {idx} Epoch {epoch}, Val Loss: {val_loss / len(val_loader)}, Val Acc: {100. * correct / total}%") 
                       
            # Stage 2: Fusion model
            # Stage 2.1: Train funsion model
            self.fusion_model.train()
            all_features = []
            all_labels = []
            
            # Extract features from meta-models
            for classifier, train_loader in zip(self.classifiers, attribute_dataloaders):
                classifier_features = []
                classifier_labels = []
                
                for batch_x, batch_y in train_loader:
                    # Extract features
                    features = classifier(batch_x, extract_features=True)
                    classifier_features.append(features)
                    classifier_labels.append(batch_y)
                
                all_features.append(torch.cat(classifier_features, dim=0))
                all_labels.append(torch.cat(classifier_labels, dim=0))
            
            # Concatenate features from all meta-models
            total_features = torch.cat(all_features, dim=1)
            final_labels = all_labels[0]  # Assume consistent labels across attributes
            
            # Train fusion model
            fusion_optimizer.zero_grad()
            fusion_outputs = self.fusion_model(total_features)
            fusion_loss = criterion(fusion_outputs, final_labels)
            
            fusion_loss.backward()
            fusion_optimizer.step()
            
            print(f"Fusion Model Epoch {epoch}, Loss: {fusion_loss.item()}")
            
            # Stage 2.2: Funsion model validation
            self.fusion_model.eval()
            with torch.no_grad():
                val_features = []
                val_labels = []
                
                for classifier, val_loader in zip(self.classifiers, validation_dataloaders):
                    classifier_features = []
                    classifier_labels = []
                    
                    for batch_x, batch_y in val_loader:
                        features = classifier(batch_x, extract_features=True)
                        classifier_features.append(features)
                        classifier_labels.append(batch_y)
                    
                    val_features.append(torch.cat(classifier_features, dim=0))
                    val_labels.append(torch.cat(classifier_labels, dim=0))
                
            # Concatenate validation features
            total_val_features = torch.cat(val_features, dim=1)
            final_val_labels = val_labels[0]  # Assume consistent labels across attributes
            
            val_outputs = self.fusion_model(total_val_features)
            val_loss = criterion(val_outputs, final_val_labels)
            
            _, predicted = val_outputs.max(1)
            total = final_val_labels.size(0)
            correct = predicted.eq(final_val_labels).sum().item()
            
            print(f"Fusion Model Epoch {epoch}, Val Loss: {val_loss.item()}, Val Acc: {100. * correct / total}%")

    def predict(self, attribute_test_loaders):
        """
        Make predictions using feature fusion
        """
        # Extract features from meta-models
        all_features = []
        
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

# Example Usage
def main():
    stime = time.ctime()
    
    # attribute_names = ['seismic', 'freq', 'dip', 'rms', 'phase']
    attribute_names = ['freq', 'phase']
    data_factory = HorizonDataFactory(attr_dirs=args.attr_dirs, kernel_size=(1, 288, 64), stride=(1, 64, 32))
    
    # Example usage for the AdvancedEnsembleLearner
    attribute_dataloaders = data_factory.get_dataloaders(attribute_names)
    
    attribute_train_loaders = [loaders["train"] for _, loaders in attribute_dataloaders.items()]
    attribute_val_loaders = [loaders["val"] for _, loaders in attribute_dataloaders.items()]
    attribute_test_loaders = [loaders["test"] for _, loaders in attribute_dataloaders.items()]
    
    attribute_configs = [
        {'input_dim': 288, 'feature_dim': 288},  # Attribute 1
        {'input_dim': 288, 'feature_dim': 288},  # Attribute 2
        {'input_dim': 288, 'feature_dim': 288}   # Attribute 3
    ]
    embed_dims = [72, 36, 36, 36]
    heads = 2
    # Initialize Advanced Ensemble Learner
    ensemble_learner = AdvancedEnsembleLearner(
        dim=embed_dims,
        num_heads=heads,
        attribute_configs=attribute_configs, 
        num_classes=2,
        num_classifiers=len(attribute_configs)
    )
    
    if args.is_training:
        print('*********is training *********')    
        # Train ensemble
        ensemble_learner.train_ensemble(attribute_train_loaders, attribute_val_loaders)
    
    if args.is_testing:
        print('********* is testing *********')
        # Predict
        predictions = ensemble_learner.predict(attribute_test_loaders)
    
    etime = time.ctime()
    print(stime, etime)
       

def parse_args():
    parser = argparse.ArgumentParser(description='DexHorizon_trainer.')
    
    parser.add_argument('--is_training', type=bool, default=True, 
                        help='Script in training mode')
    
    parser.add_argument('--data_dir', type=str,  default='/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy', help='data dir')
    
    parser.add_argument('--attributes', type=list,  default=['seisimc', 'freq', 'rms', 'dip'], help='attributes chosen for model')
    
    parser.add_argument('--label_dir', type=str,  default='/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy',
                        help='label dir')
    
    parser.add_argument('--is_testing', type=bool, default=False,
                        help='Script in testing mode')
    
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='device configuration')
    
    parser.add_argument('--embed_dims', type=list, default=[72, 36, 36, 36],
                        help='Script in testing mode')
    
    parser.add_argument('--heads', type=int,  default=2,
                        help='Script in testing mode')
    
    parser.add_argument('--num_epoch', type=int,  default=2,
                        help='Overall training epochs')
    
    parser.add_argument('--training_dir', type=str,  default='./process/training',
                        help='training log dir')
    
    parser.add_argument('--output_dir', type=str,  default='./process/output',
                        help='output log dir')
    parser.add_argument('--attr_dirs', type=dict,  default = {
        "freq": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy", "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        "phase": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_phase.npy", "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"}
    }, help='attr names and paths')
    
    args = parser.parse_args()
        
    return args

if __name__ == '__main__':
    args = parse_args()
    main()
    
    
