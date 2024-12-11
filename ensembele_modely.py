import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataset as Dataset
from torch.autograd import Variable

from model import Diformer
from training import train_one_epoch, validate
from utils.datafactory import HorizonDataFactory


class SimpleDataset(Dataset.Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __getitem__(self, idx):
        data = torch.Tensor(self.Data[idx])  
        label = torch.IntTensor(self.Label[idx])  

        return data, label

    def __len__(self):
        return len(self.Data)


class FeatureExtractorBase(nn.Module):
    """Base class for feature extractors with different data attributes"""
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class FeatureExtractorBase(nn.Module):
    """Base class for feature extractors with different data attributes"""
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)
    
class EnsembleModel(nn.Module):
    """Advanced Ensemble Model with Feature Fusion"""
    def __init__(self, 
                 feature_extractors: List[FeatureExtractorBase], 
                 num_classes: int,
                 fusion_strategy: str = 'concat'):
        super().__init__()
        
        # Store feature extractors
        self.feature_extractors = nn.ModuleList(feature_extractors)
        
        # Calculate total feature dimension
        total_feature_dim = sum(extractor.feature_extractor[-1].out_features 
                                for extractor in feature_extractors)
        
        # Fusion strategy
        self.fusion_strategy = fusion_strategy
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Extract features from different attribute sets
        extracted_features = [
            extractor(input_tensor) 
            for extractor, input_tensor in zip(self.feature_extractors, inputs)
        ]
        
        # Concatenate features
        fused_features = torch.cat(extracted_features, dim=1)
        
        # Final classification
        return self.fusion_layer(fused_features)

class ModelEnsemblePipeline:
    """Comprehensive Ensemble Training Pipeline"""
    def __init__(self, 
                 attribute_configs: List[Dict], 
                 num_classes: int,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Create feature extractors for each attribute set
        self.feature_extractors = [
            FeatureExtractorBase(
                input_dim=config['input_dim'], 
                feature_dim=config.get('feature_dim', 128)
            ).to(self.device)
            for config in attribute_configs
        ]
        
        # Create ensemble model
        self.ensemble_model = EnsembleModel(
            feature_extractors=self.feature_extractors,
            num_classes=num_classes
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.ensemble_model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-5
        )
    
    def train_step(self, inputs: List[torch.Tensor], labels: torch.Tensor) -> float:
        # Move inputs and labels to device
        inputs = [x.to(self.device) for x in inputs]
        labels = labels.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.ensemble_model(inputs)
        
        # Compute loss
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimize
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Move inputs to device
        inputs = [x.to(self.device) for x in inputs]
        
        # Disable gradient computation
        with torch.no_grad():
            outputs = self.ensemble_model(inputs)
            return torch.softmax(outputs, dim=1)



def main(args):
    stime = time.ctime()
    Loss_list = []
    accuracy = []
    seed = 0
    
    # Configuration for different attribute sets
    attribute_configs = [
        {'input_dim': 100, 'feature_dim': 128},  # First attribute set
        {'input_dim': 150, 'feature_dim': 128},  # Second attribute set
        {'input_dim': 200, 'feature_dim': 128}   # Third attribute set
    ]
    
    # Initialize pipeline
    pipeline = ModelEnsemblePipeline(
        attribute_configs=attribute_configs, 
        num_classes=7
    )
    
    model = Diformer().to(args.device)
    
    if args.is_training:
        print('*********is training *********')
       
        data_factory = HorizonDataFactory(
            data_path=args.train_data_path, 
            label_path=args.train_label_path,
            kernel_size=(1, 288, 64),
            stride=(1, 64, 32),
            train_ratio=0.8,
            batch_size=50
        )

        # Get dataloaders
        train_loader, val_loader = data_factory.get_dataloaders()

        # Optional: Get dataset sizes
        train_size, val_size = data_factory.get_dataset_sizes()

        l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 4.2]
        l_weight = torch.tensor(l_weight, requires_grad=True)
        l_weight = Variable(l_weight).cuda()
        criterion = nn.CrossEntropyLoss(l_weight)
        val_loss, val_accuracy = [], []
        
        for epoch in range(args.num_epoch):  
            # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)  # lr=0.001
            optimizer = optim.AdamW(model.parameters(),
                                lr=0.01,  
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
            
            y1, y2 = avg_loss
            Loss_list.append(y1)
            accuracy.append(y2)
            val_epoch_loss, val_epoch_accuracy = validate(model,
                                                        val_loader,
                                                        criterion)

            inputs = [
                torch.randn(32, config['input_dim']) 
                for config in attribute_configs
            ]
            
            labels = torch.randint(0, 10, (32,))
            
            # Training step
            loss = pipeline.train_step(inputs, labels)
            
            # validate_one_epoch(val_loader, model)

            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))

            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)

            print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')


            if epoch > 0 and epoch % 20 == 0:
                torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        os.path.join(output_dir_epoch, '1_25_freq_{0}_model.pth'.format(epoch)))

    if args.is_testing:
        print('_______ is testing _______')
        
        checkpoint_path = args.ckpt
        
        data_factory = HorizonDataFactory(
            data_path=args.test_data_path, 
            label_path=args.test_label_path
        )

        # Get dataloaders
        test_loader,  = data_factory.get_dataloaders()

        # Optional: Get dataset sizes
        test_size, _ = data_factory.get_dataset_sizes()
        
        test(checkpoint_path, test_loader, model, args.device, args.output_dir)
        
        return
        
    
    etime = time.ctime()
    print(stime, etime)


def parse_args():
    parser = argparse.ArgumentParser(description='DexHorizon_trainer.')
    
    parser.add_argument('--is_training', type=bool, default=True, 
                        help='Script in training mode')
    
    parser.add_argument('--data_dir', type=str,  default='/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy', help='data dir')
    
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
    
    
    
    args = parser.parse_args()
        
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

