# Ensemble Seismic Horizon Picker
# 
# Quick Start
# 1. Prepare your data:
#    - Place all attribute files (.npy) in appropriate directories
#    - Configure attribute paths in --attr_dirs parameter
#    - Requires seismic volume and corresponding label files
#
# 2. Training:
#    ```bash
#    python models/ensemseis.py --is_training True --num_epochs 100
#    ```
#
# 3. Testing:
#    ```bash
#    python models/ensemseis.py --is_testing True
#    ```
#
# Output
# - Meta-models: Individual models trained on each attribute
# - Fusion model: Combines features from all meta-models
# - Predictions: Combined horizon predictions with improved accuracy
# - Metrics: Accuracy and performance measurements
# All outputs saved to specified checkpoint and inference directories.

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

"""
Code for training ensemble model
"""

class AdvancedEnsembleLearner:
    """
    Advanced Ensemble Learning with Feature Fusion
    """
    def __init__(self, dim: List[Dict], num_heads: List[Dict], meta_model_path, fusion_model_path, num_classes: int, num_classifiers:int, height: int = 288, width: int = 1, patience: int=7):
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
                feature_projection_dim=288 # define projection dim regarding model_patch
            ).to(self.device) for _ in range(num_classifiers)
        ])
        
        total_feature_dim = num_classifiers * num_classes

        self.fusion_model = MemoryEfficientUNetFusion(
            total_feature_dim=total_feature_dim, 
            num_classes=num_classes,
            fusion_height=height,
            fusion_width=width
        )
        self.fusion_model = self.fusion_model.to(self.device)
        
        
    def train_meta_models(self, classifier, train_loader, optimizer, attr_name, epoch):
        classifier.train()
        total_loss = 0
        accuracy = []
        scaler = torch.cuda.amp.GradScaler()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]
        l_weight_tensor = torch.tensor(l_weight, requires_grad=False).to(self.device)
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                criterion = nn.CrossEntropyLoss(weight=l_weight_tensor)
                batch_y = torch.squeeze(batch_y.long())
                outputs = classifier(batch_x) 
                loss = criterion(outputs, batch_y) 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            with torch.no_grad():
                # _, predicted = torch.max(outputs[6], 1) # for dod model
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == batch_y).sum().item()
                accuracy.append(correct / batch_y.numel())
        print(f"Meta-Model {attr_name} - Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader)} Training Acc: {np.mean(accuracy)*100:.2f}%")


    def val_meta_models(self, classifier, val_loader, attr_name, epoch, early_stopping, save_path, scheduler):
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]
        l_weight_tensor = torch.tensor(l_weight, requires_grad=False).to(self.device)
        weighted_criterion = nn.CrossEntropyLoss(weight=l_weight_tensor)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_y = torch.squeeze(batch_y.long())
                # outputs, _ = classifier(batch_x) # for dod model
                # loss = weighted_criterion(outputs[6], batch_y.long()) # for dod model
                outputs = classifier(batch_x) 
                loss = weighted_criterion(outputs, batch_y.long()) 
                val_loss += loss.item()
                # _, predicted = outputs[6].max(1) # for dod model
                _, predicted = outputs.max(1)
                # total += batch_y.size(0)
                total += batch_y.numel()
                correct += predicted.eq(batch_y).sum().item()
        print(f"Meta-Model {attr_name} - Epoch {epoch+1}: Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {100. * correct / total:.2f}%")
        scheduler.step(val_loss)
        early_stopping(-val_loss, classifier, save_path, attr_name, stage_name='meta')
        
        
    def train_fusion_model(self, attribute_dataloaders, optimizer, criterion):
        self.fusion_model.train()
        scaler = torch.cuda.amp.GradScaler()
        
        train_loaders = [loader for _, loader in attribute_dataloaders]
        loader_iters = [iter(loader) for loader in train_loaders]
        min_batches = min(len(loader) for loader in train_loaders)
        
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        batch_count = 0
        
        for batch_idx in range(min_batches):
            batch_features = []
            
            try:
                # Process first batch
                first_batch = next(loader_iters[0])
                batch_size = first_batch[0].size(0)  
                first_labels = first_batch[1]
                x = first_batch[0].to(self.device)
                features = self.classifiers[0](x)
                batch_features.append(features)
                
                # Process remaining batches
                for classifier, loader_iter in zip(self.classifiers[1:], loader_iters[1:]):
                    try:
                        batch_x, _ = next(loader_iter)
                        
                        if batch_x.size(0) != batch_size:
                            if batch_x.size(0) > batch_size:
                                batch_x = batch_x[:batch_size]
                            else:
                                padding_needed = batch_size - batch_x.size(0)
                                padding_shape = list(batch_x.size())
                                padding_shape[0] = padding_needed
                                padding = torch.zeros(padding_shape, device=batch_x.device)
                                batch_x = torch.cat([batch_x, padding], dim=0)
                        
                        batch_x = batch_x.to(self.device)
                        features = classifier(batch_x)
                        batch_features.append(features)
                        
                    except StopIteration:
                        break
                    
                total_features = torch.cat(batch_features, dim=1)
                
                optimizer.zero_grad()
                labels = first_labels.to(self.device)
                labels = torch.squeeze(labels)
                
                fusion_outputs = self.fusion_model(total_features)
                loss = criterion(fusion_outputs, labels.long())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Calculate accuracy
                with torch.no_grad():
                    _, predicted = fusion_outputs.max(1)
                    correct = predicted.eq(labels).sum().item()
                    total_correct += correct
                    total_pixels += labels.numel()  # Count all pixels for segmentation
                
                total_loss += loss.item()
                batch_count += 1
                
            except StopIteration:
                break
            
        # Calculate average loss and accuracy
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = 100. * total_correct / total_pixels if total_pixels > 0 else 0
        
        # print(f"Fusion Model Training - Loss: {avg_loss:.4f}, Pixel-wise Accuracy: {accuracy:.2f}%")
        results = {"loss": avg_loss, "accuracy": accuracy}
        
        return results
    
    def validate_fusion_model(self, validation_dataloaders, criterion, fm_path, early_stopping, fusion_scheduler, epoch=None, total_epochs=None):
        self.fusion_model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        attr_names = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            val_loaders = [loader for _, loader in validation_dataloaders]
            loader_iters = [iter(loader) for loader in val_loaders]
            min_batches = min(len(loader) for loader in val_loaders)
            
            for batch_idx in range(min_batches):
                batch_features = []
                
                try:
                    # Process first loader
                    first_batch = next(loader_iters[0])
                    first_labels = first_batch[1]
                    x = first_batch[0].to(self.device)
                    features = self.classifiers[0](x)
                    batch_features.append(features)
                    
                    # Process remaining loaders
                    for classifier, loader_iter in zip(self.classifiers[1:], loader_iters[1:]):
                        batch_x, _ = next(loader_iter)
                        batch_x = batch_x.to(self.device)
                        features = classifier(batch_x)
                        batch_features.append(features)
                    
                    # Process this batch
                    total_features = torch.cat(batch_features, dim=1)
                    labels = first_labels.to(self.device)
                    labels = torch.squeeze(labels)
                    
                    outputs = self.fusion_model(total_features)
                    loss = criterion(outputs, labels.long())
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_samples += labels.numel()
                    total_correct += predicted.eq(labels).sum().item()
                    
                    # Store predictions and labels if it's a save trigger
                    if self._should_save_results(epoch, total_epochs):
                        all_predictions.append(predicted.cpu())
                        all_labels.append(labels.cpu())
                    
                except StopIteration:
                    break
            
            # Calculate average metrics
            avg_loss = total_loss / min_batches
            accuracy = 100. * total_correct / total_samples
            
            # Update scheduler and early stopping
            fusion_scheduler.step(avg_loss)
            attr_names = [name for name, _ in validation_dataloaders]
            combined_attr_name = '_'.join(attr_names)
            early_stopping(-avg_loss, self.fusion_model, fm_path, combined_attr_name, stage_name='fusion')
            
            # Save validation results if triggered
            if self._should_save_results(epoch, total_epochs) and all_predictions:
                self._save_validation_results(
                    all_predictions=torch.cat(all_predictions, dim=0),
                    all_labels=torch.cat(all_labels, dim=0),
                    accuracy=accuracy,
                    loss=avg_loss,
                    epoch=epoch,
                    combined_attr_name=combined_attr_name,
                    save_dir=fm_path
                )
            
            return {"loss": avg_loss, "accuracy": accuracy}

    def _should_save_results(self, epoch, total_epochs):
        """Determine if we should save validation results based on triggers"""
        if epoch is None or total_epochs is None:
            return False
            
        triggers = [
            epoch == total_epochs - 1,  # Last epoch
            epoch % 10 == 0,  # Every 10 epochs
            epoch == 0,  # First epoch
        ]
        return any(triggers)

    def _save_validation_results(self, all_predictions, all_labels, accuracy, loss, epoch, combined_attr_name, save_dir):
        """Save validation results to files"""
        # Create validation results directory
        val_dir = os.path.join(save_dir, 'validation_results')
        os.makedirs(val_dir, exist_ok=True)
        
        # Save predictions and labels
        epoch_dir = os.path.join(val_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Save predictions
        pred_path = os.path.join(epoch_dir, f'{combined_attr_name}_predictions.npy')
        np.save(pred_path, all_predictions.numpy())
        
        # Save labels
        labels_path = os.path.join(epoch_dir, f'{combined_attr_name}_labels.npy')
        np.save(labels_path, all_labels.numpy())
        
        # Save metrics
        metrics_path = os.path.join(epoch_dir, f'{combined_attr_name}_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f'Epoch: {epoch}\n')
            f.write(f'Validation Loss: {loss:.4f}\n')
            f.write(f'Validation Accuracy: {accuracy:.2f}%\n')
        
        print(f"Saved validation results for epoch {epoch} to {epoch_dir}")
        
    
    def train_ensemble(self, 
                       attribute_dataloaders: List, 
                       validation_dataloaders: List, 
                       epochs: int = 2, 
                       meta_learning_rate: float = 1e-3,
                       fusion_learning_rate :float = 1e-2                       
                       ):
        """
        Train meta-models and fusion model
        """
        
        # Individual optimizers for classifiers and fusion model
        classifier_optimizers = [
            optim.AdamW(classifier.parameters(), lr=meta_learning_rate) 
            for classifier in self.classifiers
        ]
        fusion_optimizer = optim.AdamW(self.fusion_model.parameters(), lr=fusion_learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        classifier_schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=5
            ) for optimizer in classifier_optimizers
        ]

        # Scheduler for fusion model
        fusion_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            fusion_optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
        
        # Init early stopping
        mm_path = os.path.join(self.mmp)
        fm_path = os.path.join(self.fmp)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        
        for epoch in range(epochs):
            # Stage 1: Individual meta-models for each attribute 
            for idx, (classifier, ((attr_name, train_loader), (attr_name, val_loader)), optimizer, scheduler) in enumerate(zip(self.classifiers, zip(attribute_dataloaders, validation_dataloaders), classifier_optimizers, classifier_schedulers)):
                self.train_meta_models(classifier, train_loader, optimizer, attr_name, epoch)
                self.val_meta_models(classifier, train_loader, attr_name, epoch, early_stopping, mm_path, scheduler)
            
            print('\n')     
            # Stage 2: Fusion model training and validation
            results = self.train_fusion_model(attribute_dataloaders, fusion_optimizer, criterion)
            
            fusion_val_metrics = self.validate_fusion_model(
                validation_dataloaders, 
                criterion, 
                fm_path, 
                early_stopping, 
                fusion_scheduler,
                epoch=epoch,
                total_epochs=epochs
            )
            print(f"Fusion Model Epoch {epoch+1}:Train Loss: {results['loss']:.4f} Train Acc: {results['accuracy']:.4f}  Val Loss: {fusion_val_metrics['loss']:.4f}, Val Acc: {fusion_val_metrics['accuracy']:.4f}%")
            print('\n')
            print('\n')    
    
    
        
    def predict(self, dim, num_heads, attribute_test_loaders):
        """
        Make predictions using feature fusion and compute accuracy
        """
        self.dim = dim
        self.num_heads = num_heads
        self.fusion_model.eval()
        meta_models = {}
        attr_names = []
        
        def load_checkpoint(model, path, optimizer=None, scheduler=None):
            try:
                checkpoint = torch.load(path, map_location=self.device)
                
                # Handle both old and new checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # Explicitly set to eval mode regardless of saved state
                    model.eval()
                    
                    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        
                    print(f"Loaded checkpoint with validation loss: {checkpoint.get('val_loss', 'N/A')}")
                else:
                    # Handle old format where checkpoint is just the state dict
                    model.load_state_dict(checkpoint)
                    model.eval()  # Explicitly set to eval mode
                    print("Loaded model state dict (old format)")
                
                # Double check training mode
                assert not model.training, "Model should be in eval mode after loading"
                print(f"Model parameters loaded: {sum(p.numel() for p in model.parameters())}")
                
                return model
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                raise
        
        def validate_model_states():
            print("\nValidating model states:")
            for attr_name, model in meta_models.items():
                if model.training:
                    print(f"Warning: {attr_name} model was in training mode! Setting to eval mode...")
                    model.eval()
                print(f"{attr_name} model is now in eval mode: {'✓' if not model.training else '✗'}")
            
            if self.fusion_model.training:
                print("Warning: Fusion model was in training mode! Setting to eval mode...")
                self.fusion_model.eval()
            print(f"Fusion model is in eval mode: {'✓' if not self.fusion_model.training else '✗'}")
            
        # 1. Load meta models
        for attr_name, _ in attribute_test_loaders:
            model_path = os.path.join(self.mmp, f"meta_{attr_name}_checkpoint.pth")
            attr_names.append(attr_name)
            print(f"\nLoading meta model: meta_{attr_name}_checkpoint.pth")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Meta model not found for {attr_name}: {model_path}")
            
            meta_model = Diformer(
                dim=self.dim,  
                num_heads=self.num_heads,
                feature_projection_dim=288
            ).to(self.device)
            
            meta_model = load_checkpoint(meta_model, model_path)
            meta_model.eval()  # Explicit eval mode
            meta_models[attr_name] = meta_model

        # 2. Load fusion model
        combined_attr_name = '_'.join(attr_names)
        fusion_model_path = os.path.join(self.fmp, f"fusion_{combined_attr_name}_checkpoint.pth")

        if not os.path.exists(fusion_model_path):
            raise FileNotFoundError(f"Fusion model not found: {fusion_model_path}")

        print(f"\nLoading fusion model: fusion_{combined_attr_name}_checkpoint.pth")
        # After loading fusion model
        self.fusion_model = load_checkpoint(self.fusion_model, fusion_model_path)
        self.fusion_model.eval()  # Explicit eval mode
        
        validate_model_states()
        
        # 3. Process data and make predictions
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            total_correct = 0
            total_pixels = 0
            
            test_loaders = [loader for _, loader in attribute_test_loaders]
            loader_iters = [iter(loader) for loader in test_loaders]
            
            while True:
                try:
                    batch_features = []
                    
                    # Get first batch to get labels
                    first_batch = next(loader_iters[0])
                    batch_labels = first_batch[1].to(self.device)
                    batch_x = first_batch[0].to(self.device)
                    features = meta_models[attr_names[0]](batch_x)
                    batch_features.append(features)
                    
                    # Process remaining attributes
                    for classifier, loader_iter, attr_name in zip(list(meta_models.values())[1:], loader_iters[1:], attr_names[1:]):
                        batch_x, _ = next(loader_iter)
                        batch_x = batch_x.to(self.device)
                        features = classifier(batch_x)
                        batch_features.append(features)
                    
                    # Make predictions
                    total_features = torch.cat(batch_features, dim=1)
                    predictions = self.fusion_model(total_features)
                    _, predicted_classes = torch.max(predictions, dim=1)
                    
                    # Calculate accuracy
                    batch_labels = torch.squeeze(batch_labels)
                    correct = predicted_classes.eq(batch_labels).sum().item()
                    total_correct += correct
                    total_pixels += batch_labels.numel()  # Count all pixels
                    
                    # Store predictions and labels
                    all_predictions.append(predicted_classes.cpu())
                    all_labels.append(batch_labels.cpu())
                    
                except StopIteration:
                    break

            # Compute overall accuracy
            accuracy = 100. * total_correct / total_pixels if total_pixels > 0 else 0
            print(f"Test Pixel-wise Accuracy: {accuracy:.2f}%")
            
            # Combine all predictions
            if all_predictions:
                final_predictions = torch.cat(all_predictions, dim=0)  # No second max needed
                all_labels = torch.cat(all_labels, dim=0)
                
                results = {
                    'predictions': final_predictions,
                    'labels': all_labels,
                    'accuracy': accuracy,
                    'combined_attr_name': combined_attr_name
                }
                
                return results
            else:
                return None
        

def main():
    stime = time.ctime()
    # attribute_names = ['seismic', 'freq', 'dip', 'phase','rms', 'complex', 'coherence','average_zero']
    attribute_names = ['seismic', 'phase']
    
    data_factory = HorizonDataFactory(attr_dirs=args.attr_dirs, kernel_size=(1, 288, 32), stride=(1, 32, 32), batch_size=args.batch_size) # the resulting 
    
    attribute_dataloaders = data_factory.get_dataloaders(attribute_names)
    
    attribute_keys = list(attribute_dataloaders.keys())  # Extract keys from the dictionary
    
    train_values = [loaders["train"] for _, loaders in attribute_dataloaders.items()]  # Extract corresponding "train" loaders
    attribute_train_loaders = list(zip(attribute_keys, train_values))
    
    val_values = [loaders["val"] for _, loaders in attribute_dataloaders.items()]  # Extract corresponding "val" loaders
    attribute_val_loaders = list(zip(attribute_keys, val_values))
    
    test_values = [loaders["test"] for _, loaders in attribute_dataloaders.items()]  # Extract corresponding "test" loaders
    attribute_test_loaders = list(zip(attribute_keys, test_values))
    
    embed_dims = [72, 36, 36, 36]
    
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

        results = ensemble_learner.predict(embed_dims, heads, attribute_test_loaders)
        
        if results is not None:
            predictions = results['predictions'].cpu().numpy()
            labels = results['labels'].cpu().numpy()
            accuracy = results['accuracy']
            combined_attr_name = results['combined_attr_name']
            
            # Create inference directory
            os.makedirs(args.inference_dir, exist_ok=True)
            
            # Save predictions
            pred_save_path = os.path.join(args.inference_dir, f'{combined_attr_name}_predictions.npy')
            np.save(pred_save_path, predictions)
            print(f'Saved predictions to {pred_save_path}')
            
            # Save ground truth labels
            label_save_path = os.path.join(args.inference_dir, f'{combined_attr_name}_labels.npy')
            np.save(label_save_path, labels)
            print(f'Saved ground truth labels to {label_save_path}')
            
            # Save accuracy to a text file
            metrics_save_path = os.path.join(args.inference_dir, f'{combined_attr_name}_metrics.txt')
            with open(metrics_save_path, 'w') as f:
                f.write(f'Test Accuracy: {accuracy:.2f}%\n')
            print(f'Saved metrics to {metrics_save_path}')
        else:
            print("No predictions were generated.")
        return  
    
    etime = time.ctime()
    print(stime, etime)
       

def parse_args():
    parser = argparse.ArgumentParser(description='Diformer_trainer.')
    
    parser.add_argument('--device', type=str, default='cuda:1',
                    help='device configuration')
    
    parser.add_argument('--is_training', type=bool, default=False, # False
                        help='Script in training mode')
    
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='Overall training epochs')
    
    parser.add_argument('--batch_size', type=int, default=36, 
                        help='Training batch size')
    
    parser.add_argument('--height', type=int, default=288, 
                        help='data height size')
    
    parser.add_argument('--width', type=int, default=32, 
                        help='data width size')

    parser.add_argument('--mm_ckpt_path', type=str, default='./process/meta_model_ckpt', 
                        help='checkpoint saving/loading path of meta model')
    
    parser.add_argument('--fm_ckpt_path', type=str, default='./process/fusion_model_ckpt', 
                        help='checkpoint saving/loading path of fusion model')

    parser.add_argument('--is_testing', type=bool, default=True,
                        help='Script in testing mode')
    
    # parser.add_argument('--device', type=str, default='cuda:1',
    #                     help='device configuration')
    
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
    
    parser.add_argument('--inference_dir', type=str,  default='./process/inference',
                        help='model inference prediction dir')
    
    parser.add_argument('--attr_dirs', type=dict,  default = {
        "seismic": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy", 
                    "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "freq": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy", 
        #          "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "dip": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_predict_MCDL_crossline.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        "phase": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_phase.npy", 
                  "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "rms": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_RMSAmp.npy", 
        #           "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "complex": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_complex_trace.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "coherence": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_coherence.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        # "average_zero": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_Average_zero_crossing.npy", 
        #         "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        
        
    }, help='attr names and paths')
    
    args = parser.parse_args()
        
    return args

if __name__ == '__main__':
    args = parse_args()
    main()
    