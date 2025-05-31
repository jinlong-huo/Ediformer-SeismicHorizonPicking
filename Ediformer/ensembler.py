import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.diformer_patch_attn import Diformer
from models.memfusion import MemoryEfficientUNetFusion
from utils.datafactory import HorizonDataFactory

# Try to import EarlyStopping, otherwise define a simple one
try:
    from utils.tools import EarlyStopping
except ImportError:
    class EarlyStopping:
        """Simple EarlyStopping implementation"""
        def __init__(self, patience=7, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta

        def __call__(self, val_loss, model, path, model_name, stage_name=''):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, model_name, stage_name)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path, model_name, stage_name)
                self.counter = 0

        def save_checkpoint(self, val_loss, model, path, model_name, stage_name=''):
            '''Saves model when validation loss decrease.'''
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), os.path.join(path, f'{stage_name}_{model_name}_best.pth'))
            self.val_loss_min = val_loss

class AdvancedEnsembleLearner:
    """
    Advanced Ensemble Learning with Feature Fusion
    """
    def __init__(self, dim: List[int], num_heads: List[int], meta_model_path, fusion_model_path, 
                 num_classes: int, num_classifiers: int, height: int = 288, width: int = 1, patience: int = 7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.mmp = meta_model_path
        self.fmp = fusion_model_path
        self.patience = patience
        
        # Initialize classifiers with proper device placement
        self.classifiers = nn.ModuleList([
            Diformer(
                dim=dim, 
                num_heads=num_heads,
                feature_projection_dim=16
            ).to(self.device) for _ in range(num_classifiers)
        ])
        
        # Calculate total feature dimension
        total_feature_dim = num_classifiers * num_classes
        print(f"Total feature dimension: {total_feature_dim}")
        
        self.fusion_model = MemoryEfficientUNetFusion(
            total_feature_dim=total_feature_dim, 
            num_classes=num_classes,
            fusion_height=height,
            fusion_width=width
        ).to(self.device)
        
        # Classifier weights
        self.classifier_weights = torch.ones(len(self.classifiers)) / len(self.classifiers)
    
    def train_ensemble(self, 
                       attribute_dataloaders: List, 
                       validation_dataloaders: List, 
                       epochs: int = 2, 
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
        
        # Init early stopping - one for each classifier and one for fusion
        mm_path = os.path.join(self.mmp)
        fm_path = os.path.join(self.fmp)
        
        # Create separate early stopping instances for each model
        classifier_early_stoppings = [EarlyStopping(patience=self.patience, verbose=True) 
                                     for _ in self.classifiers]
        fusion_early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        for epoch in range(epochs):
            # Stage 1: Individual meta-models
            for idx, (classifier, ((attr_name, train_loader), (attr_name_val, val_loader)), optimizer, early_stopping) in enumerate(
                zip(self.classifiers, zip(attribute_dataloaders, validation_dataloaders), classifier_optimizers, classifier_early_stoppings)):
            
                # Stage 1.1 meta-models Training
                classifier.train()
                total_loss = 0
                stage_name = 'meta'
                
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    # Move to device and squeeze
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device).squeeze(1).long()
                    
                    outputs = classifier(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                print(f"Meta-Model {attr_name} - Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader):.4f}")
                
                # Stage 1.2 meta-models Validation
                classifier.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device).squeeze(1).long()
                        
                        outputs = classifier(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        _, predicted = outputs.max(1)
                        total += batch_y.numel()
                        correct += predicted.eq(batch_y).sum().item()
                
                accuracy = 100 * correct / total
                print(f"Meta-Model {attr_name} - Epoch {epoch + 1}: Validation Loss: {val_loss / len(val_loader):.4f}, "
                      f"Validation Accuracy: {accuracy:.4f}%")
                
                # Check if early stopping has __call__ method, otherwise use step()
                if hasattr(early_stopping, '__call__'):
                    early_stopping(-val_loss, classifier, mm_path, attr_name, stage_name=stage_name)
                elif hasattr(early_stopping, 'step'):
                    early_stopping.step(-val_loss, classifier, mm_path, attr_name, stage_name=stage_name)
                else:
                    # If the interface is different, just save the model
                    torch.save(classifier.state_dict(), os.path.join(mm_path, f'{stage_name}_{attr_name}_epoch{epoch+1}.pth'))
                
                # Check if we should stop early
                if hasattr(early_stopping, 'early_stop') and early_stopping.early_stop:
                    print(f"Early stopping triggered for {attr_name}")
                    break     
                
            print('\n')     
            # Stage 2: Fusion model training
            # self.fusion_model.train()
            # stage_name = 'fusion'
            
            # # Collect all features and labels
            # all_train_features = []
            # all_train_labels = []
            
            # with torch.no_grad():
            #     for classifier, (attr_name, train_loader) in zip(self.classifiers, attribute_dataloaders):
            #         classifier.eval()
                    
            #         for batch_x, batch_y in train_loader:
            #             batch_x = batch_x.to(self.device)
            #             batch_y = batch_y.to(self.device)
                        
            #             # Get classifier outputs (logits)
            #             outputs = classifier(batch_x)
            #             all_train_features.append(outputs)
            #             all_train_labels.append(batch_y)
            
            # # Concatenate all features along channel dimension
            # combined_features = torch.cat(all_train_features, dim=1)
            # combined_labels = torch.cat(all_train_labels, dim=0).squeeze(1).long()
            
            # # Train fusion model
            # fusion_optimizer.zero_grad()
            # fusion_outputs = self.fusion_model(combined_features)
            # fusion_loss = criterion(fusion_outputs, combined_labels)
            # fusion_loss.backward()
            # fusion_optimizer.step()
            
            # print(f"Fusion Model Epoch {epoch+1}: Loss: {fusion_loss.item():.4f}")
            # Stage 2: Fusion model training
            self.fusion_model.train()
            stage_name = 'fusion'

            # Process batches synchronously from all dataloaders
            fusion_loss_total = 0
            batch_count = 0

            # Get iterators for each dataloader
            train_iterators = [iter(loader) for _, loader in attribute_dataloaders]

            # Process batches
            for batch_idx in range(len(next(iter(attribute_dataloaders))[1])):
                batch_features = []
                batch_labels = None
                
                # Collect features from each classifier for this batch
                for classifier, iterator in zip(self.classifiers, train_iterators):
                    try:
                        batch_x, batch_y = next(iterator)
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        with torch.no_grad():
                            classifier.eval()
                            outputs = classifier(batch_x)  # Shape: [batch, 7, H, W]
                            batch_features.append(outputs)
                        
                        if batch_labels is None:
                            batch_labels = batch_y
                    except StopIteration:
                        break
                
                if len(batch_features) != len(self.classifiers):
                    break
                    
                # Concatenate features for this batch only
                combined_features = torch.cat(batch_features, dim=1)  # Shape: [batch, 14, H, W]
                combined_labels = batch_labels.squeeze(1).long()
                
                # Train fusion model on this batch
                fusion_optimizer.zero_grad()
                fusion_outputs = self.fusion_model(combined_features)
                fusion_loss = criterion(fusion_outputs, combined_labels)
                fusion_loss.backward()
                fusion_optimizer.step()
                
                fusion_loss_total += fusion_loss.item()
                batch_count += 1

            print(f"Fusion Model Epoch {epoch+1}: Loss: {fusion_loss_total/batch_count:.4f}")
            # Stage 2.2: Fusion model validation
            self.fusion_model.eval()
            val_loss_total = 0
            correct_total = 0
            total_samples = 0

            with torch.no_grad():
                # Get iterators for validation
                val_iterators = [iter(loader) for _, loader in validation_dataloaders]
                
                for batch_idx in range(len(next(iter(validation_dataloaders))[1])):
                    batch_features = []
                    batch_labels = None
                    
                    # Collect features from each classifier for this batch
                    for classifier, iterator in zip(self.classifiers, val_iterators):
                        try:
                            batch_x, batch_y = next(iterator)
                            batch_x = batch_x.to(self.device)
                            batch_y = batch_y.to(self.device)
                            
                            classifier.eval()
                            outputs = classifier(batch_x)
                            batch_features.append(outputs)
                            
                            if batch_labels is None:
                                batch_labels = batch_y
                        except StopIteration:
                            break
                    
                    if len(batch_features) != len(self.classifiers):
                        break
                        
                    # Concatenate features for this batch
                    combined_features = torch.cat(batch_features, dim=1)  # [batch, 14, H, W]
                    combined_labels = batch_labels.squeeze(1).long()
                    
                    # Validate on this batch
                    val_outputs = self.fusion_model(combined_features)
                    val_loss = criterion(val_outputs, combined_labels)
                    
                    val_loss_total += val_loss.item()
                    
                    _, predicted = val_outputs.max(1)
                    correct_total += predicted.eq(combined_labels).sum().item()
                    total_samples += combined_labels.numel()

            accuracy = 100 * correct_total / total_samples
            avg_val_loss = val_loss_total / batch_idx if batch_idx > 0 else 0

            print(f"Fusion Model Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {accuracy:.4f}%")

            attr_name_combined = '_'.join([name for name, _ in attribute_dataloaders])
            # Early stopping code remains the same...
            
    def predict(self, attribute_test_loaders, meta_model_path, fusion_model_path):
        """
        Make predictions using feature fusion
        """
        # Load models
        for idx, (classifier, (attr_name, _)) in enumerate(zip(self.classifiers, attribute_test_loaders)):
            model_path = os.path.join(meta_model_path, f'meta_{attr_name}_best.pth')
            if os.path.exists(model_path):
                classifier.load_state_dict(torch.load(model_path))
                print(f"Loaded meta-model for {attr_name}")
            else:
                print(f"Warning: No saved model found for {attr_name}")
        
        # Load fusion model
        fusion_model_path_full = os.path.join(fusion_model_path, 'fusion_seismic_dip_best.pth')
        if os.path.exists(fusion_model_path_full):
            self.fusion_model.load_state_dict(torch.load(fusion_model_path_full))
            print("Loaded fusion model")
        else:
            print("Warning: No saved fusion model found")
        
        all_features = []
        
        with torch.no_grad():
            for classifier, (attr_name, dataloader) in zip(self.classifiers, attribute_test_loaders):
                classifier.eval()
                
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.device)
                    outputs = classifier(batch_x)
                    all_features.append(outputs)
        
        # Concatenate features
        total_features = torch.cat(all_features, dim=1)
        
        # Final prediction
        with torch.no_grad():
            self.fusion_model.eval()
            predictions = self.fusion_model(total_features)
            
        return predictions


def main():
    stime = time.ctime()
    
    # Create necessary directories
    os.makedirs(args.mm_ckpt_path, exist_ok=True)
    os.makedirs(args.fm_ckpt_path, exist_ok=True)
    os.makedirs(args.training_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./logs/training', exist_ok=True)
    os.makedirs('./outputs/figures', exist_ok=True)
    os.makedirs('./outputs/predictions', exist_ok=True)
    
    # Define attributes to use
    attribute_names = ['seismic', 'dip']
    
    # Initialize data factory
    data_factory = HorizonDataFactory(
        attr_dirs=args.attr_dirs, 
        kernel_size=(1, 288, 16), 
        stride=(1, 64, 32), 
        batch_size=args.batch_size
    )
    
    # Get dataloaders
    attribute_dataloaders = data_factory.get_dataloaders(attribute_names)
    
    # Extract train, val, test loaders
    attribute_keys = list(attribute_dataloaders.keys())
    
    train_values = [loaders["train"] for _, loaders in attribute_dataloaders.items()]
    attribute_train_loaders = list(zip(attribute_keys, train_values))
    
    val_values = [loaders["val"] for _, loaders in attribute_dataloaders.items()]
    attribute_val_loaders = list(zip(attribute_keys, val_values))
    
    test_values = [loaders["test"] for _, loaders in attribute_dataloaders.items()]
    attribute_test_loaders = list(zip(attribute_keys, test_values))
    
    # FIXED: Use correct dimensions that match the model architecture
    # These should match the channel dimensions after pre_dense layers
    embed_dims = [256, 512, 512, 256]  # Changed from [16, 8, 8, 8]
    num_heads = args.heads
    
    # Initialize Advanced Ensemble Learner
    ensemble_learner = AdvancedEnsembleLearner(
        dim=embed_dims,
        num_heads=num_heads,
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
        ensemble_learner.train_ensemble(
            attribute_train_loaders, 
            attribute_val_loaders, 
            epochs=args.num_epochs
        )
        
    if args.is_testing:
        print("============ Begin testing ============\n") 
        # Predict
        predictions = ensemble_learner.predict(
            attribute_test_loaders, 
            args.mm_ckpt_path, 
            args.fm_ckpt_path
        )
        # Save predictions
        np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions.cpu().numpy())
        print(f"Predictions saved to {os.path.join(args.output_dir, 'predictions.npy')}")
    
    etime = time.ctime()
    print(f"Start time: {stime}")
    print(f"End time: {etime}")


def parse_args():
    parser = argparse.ArgumentParser(description='Diformer ensemble trainer.')
    
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

    parser.add_argument('--is_testing', type=bool, default=False,
                        help='Script in testing mode')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device configuration')
    
    parser.add_argument('--heads', type=int, default=2,
                        help='Number of attention heads')
    
    parser.add_argument('--mm_ckpt_path', type=str, default='./checkpoints/meta_models', 
                        help='checkpoint saving/loading path of meta model')
    
    parser.add_argument('--fm_ckpt_path', type=str, default='./checkpoints/fusion_models', 
                        help='checkpoint saving/loading path of fusion model')

    parser.add_argument('--training_dir', type=str, default='./logs/training',
                        help='training log dir')
        
    parser.add_argument('--output_dir', type=str, default='./outputs/results',
                        help='output log dir')
    
    parser.add_argument('--attr_dirs', type=dict, default={
        "seismic": {
            "data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy", 
            "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"
        },
        "dip": {
            "data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_predict_MCDL_crossline.npy", 
            "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"
        },
    }, help='attr names and paths')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main()