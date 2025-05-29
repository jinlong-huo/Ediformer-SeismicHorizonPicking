import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import yaml

from diformer_patch_attn import Diformer
# from models.DOD_ensemble import DexiNed
from memfusion import MemoryEfficientUNetFusion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.datafactory import HorizonDataFactory
from utils.tools import EarlyStopping

"""
we put everthing on cuda so  make sure the data and model are calculated in same stage
"""


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
                feature_projection_dim=288 # define projection dim regarding model_patch
            ).cuda() for _ in range(num_classifiers)
        ])
        
        total_feature_dim = num_classifiers * 7

        self.fusion_model = MemoryEfficientUNetFusion(
            total_feature_dim=total_feature_dim, 
            num_classes=num_classes,
            fusion_height=height,
            fusion_width=width
        )
        self.fusion_model = self.fusion_model.cuda()
        
    def train_meta_models(self, classifier, train_loader, optimizer, attr_name, epoch):
        classifier.train()
        total_loss = 0
        accuracy = []
        scaler = torch.cuda.amp.GradScaler()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        l_weight = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 7.8]
        l_weight_tensor = torch.tensor(l_weight, requires_grad=False).cuda()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
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
        l_weight_tensor = torch.tensor(l_weight, requires_grad=False).cuda()
        weighted_criterion = nn.CrossEntropyLoss(weight=l_weight_tensor)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
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
        batch_count = 0
        
        
        for batch_idx in range(min_batches):
            batch_features = []
            
            try:
                
                first_batch = next(loader_iters[0])
                batch_size = first_batch[0].size(0)  
                first_labels = first_batch[1]
                x = first_batch[0].cuda()
                features = self.classifiers[0](x)
                batch_features.append(features)
                
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
                        
                        batch_x = batch_x.cuda()
                        features = classifier(batch_x)
                        batch_features.append(features)
                        
                    except StopIteration:
                        break
                    
                total_features = torch.cat(batch_features, dim=1)
                
                optimizer.zero_grad()
                labels = first_labels.cuda()
                labels = torch.squeeze(labels)
                
                fusion_outputs = self.fusion_model(total_features)
                loss = criterion(fusion_outputs, labels.long())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                batch_count += 1
                
            except StopIteration:
                break
            
        return loss


    def validate_fusion_model(self, validation_dataloaders, criterion, fm_path, early_stopping, fusion_scheduler):
        
        self.fusion_model.eval()
        val_features = []
        attr_names = []
        
        with torch.no_grad():
            for classifier, (attr_name, val_loader) in zip(self.classifiers, validation_dataloaders):
                attr_names.append(attr_name)
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    
                    features = classifier(batch_x)
                    val_features.append(features)
                    
        total_val_features = torch.cat(val_features, dim=1)
        final_val_labels = torch.squeeze(batch_y)
        val_outputs = self.fusion_model(total_val_features)
        
        val_loss = criterion(val_outputs, final_val_labels.long())
        _, predicted = val_outputs.max(1)
        total = final_val_labels.numel()
        correct = predicted.eq(final_val_labels).sum().item()
        
        fusion_scheduler.step(val_loss)
        # Combine all attribute names with underscore
        combined_attr_name = '_'.join(attr_names)
        
        early_stopping(-val_loss, self.fusion_model, fm_path, combined_attr_name, stage_name='fusion')
      
        return {"loss": val_loss.item(), "accuracy": 100. * correct / total}

    
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
                # output explosion not caused by dataset
                self.val_meta_models(classifier, train_loader,  attr_name,  epoch, early_stopping, mm_path, scheduler)
                
            print('\n')     
            # Stage 2: Fusion model
            # Stage 2.1: Train funsion model
            # self.fusion_model.train()
            fusion_loss = self.train_fusion_model(attribute_dataloaders, fusion_optimizer, criterion)
            
            fusion_val_metrics = self.validate_fusion_model(validation_dataloaders, criterion, fm_path, early_stopping, fusion_scheduler)
            
            print(f"Fusion Model Epoch {epoch+1}: Loss: {fusion_loss.item()} Val Loss: {fusion_val_metrics['loss']:.4f}, Val Acc: {fusion_val_metrics['accuracy']:.4f}%")
            print('\n')
            print('\n')
    
    
    def predict(self, dim, num_heads, attribute_test_loaders):
        """
        Make predictions using feature fusion
        """
        # Keep models in evaluation mode
        self.dim = dim
        self.num_heads = num_heads
        
        self.fusion_model.eval()
        meta_models = {}
        attr_names = []

        # 1. Load all meta models first
        for attr_name, _ in attribute_test_loaders:
            model_path = os.path.join(self.mmp, f"meta_{attr_name}_checkpoint.pth")
            attr_names.append(attr_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Meta model not found for {attr_name}: {model_path}")
            
            # Create a new model instance for each attribute
            meta_model = Diformer(
                dim=self.dim,  
                num_heads=self.num_heads,
                feature_projection_dim=288
            ).cuda()
            
            state_dict = torch.load(model_path)
            meta_model.load_state_dict(state_dict)
            meta_model.eval()  # Set to evaluation mode
            meta_models[attr_name] = meta_model

        # 2. Load fusion model
        combined_attr_name = '_'.join(attr_names)
        fusion_model_path = os.path.join(self.fmp, f"fusion_{combined_attr_name}_checkpoint.pth")
        
        if not os.path.exists(fusion_model_path):
            raise FileNotFoundError(f"Fusion model not found: {fusion_model_path}")
        
        fusion_state_dict = torch.load(fusion_model_path)
        self.fusion_model.load_state_dict(fusion_state_dict)
        self.fusion_model.eval()

        # 3. Process data and make predictions
        with torch.no_grad():
            all_predictions = []
            
            # Assume all dataloaders have the same length
            test_loaders = [loader for _, loader in attribute_test_loaders]
            loader_iters = [iter(loader) for loader in test_loaders]
            
            while True:
                try:
                    batch_features = []
                    
                    # Process each attribute's data
                    for classifier, loader_iter in zip(meta_models.values(), loader_iters):
                        batch_x, _ = next(loader_iter)
                        batch_x = batch_x.cuda()
                        features = classifier(batch_x)
                        batch_features.append(features)
                    
                    # Combine features and make prediction
                    total_features = torch.cat(batch_features, dim=1)
                    predictions = self.fusion_model(total_features)
                    all_predictions.append(predictions.cpu())
                    
                except StopIteration:
                    break

            # Combine all predictions
            if all_predictions:
                cat_predictions = torch.cat(all_predictions, dim=0)
                final_predictions,_ = torch.max(cat_predictions, dim=1)
                
                return final_predictions, combined_attr_name
            else:
                return None       
            
    # def predict(self, attribute_test_loaders):
    #     """
    #     Make predictions using feature fusion
    #     """
    #     # Init early stopping
    #     mm_path = os.path.join(self.mmp)
    #     fm_path = os.path.join(self.fmp)
    #     meta_model = Diformer()
    #     fusion_model =  UNetFusionModel()
    #     # Extract features from meta-models
    #     all_features = []
    #     meta_models = {}
    #     attr_names = []
    #     for attr_name, _ in attribute_test_loaders:
    #         model_path = os.path.join(mm_path, f"meta_{attr_name}_checkpoint.pth")
    #         attr_names.append(attr_name)
    #         if not os.path.exists(model_path):
    #             raise FileNotFoundError(f"Meta model not found for {attr_name}: {model_path}")
    #         state_dict = torch.load(model_path)
    #         meta_model.load_state_dict(state_dict)
    #         # meta_models[attr_name] = torch.load(model_path)
    #         meta_models[attr_name] = meta_model
            
    #     combined_attr_name = '_'.join(attr_names)
    #     fusion_model_path = os.path.join(fm_path, f"fusion_{combined_attr_name}_checkpoint.pth")
    #     # Load fusion model
    #     if not os.path.exists(fusion_model_path):
    #         raise FileNotFoundError(f"Fusion model not found: {fusion_model_path}")
    #     # self.fusion_model = torch.load(fusion_model_path)
    #     fusion_state_dict = torch.load(fusion_model_path)
    #     fusion_model.load_state_dict(fusion_state_dict)
    #     # self.fusion_model.eval()
        
    #     for (attr_name, dataloader) in attribute_test_loaders:
    #         classifier_features = []
    #         classifier = meta_models[attr_name]
    #         # classifier.eval()
            
    #         with torch.no_grad():
    #             for batch_x, _ in dataloader:
    #                 batch_x = batch_x.cuda()
    #                 features = classifier(batch_x, extract_features=True)
    #                 classifier_features.append(features.cpu())  # Move to CPU to save GPU memory
    #     # Concatenate all features and move to GPU for final prediction
    #     try:
    #         total_features = torch.cat(all_features, dim=1).to(self.device)
            
    #         # Final prediction
    #         with torch.no_grad():
    #             predictions = fusion_model(total_features)
    #             return predictions.cpu()  # Return predictions on CPU    
    #     except RuntimeError as e:
    #         print(f"Error during feature fusion: {e}")
    #         print(f"Feature shapes: {[f.shape for f in all_features]}")
    #         raise   
        
            # Concatenate features for this attribute
        # attr_features = torch.cat(classifier_features, dim=0)
        # all_features.append(attr_features)
        # print(f"Processed features for {attr_name}")
        # for classifier, dataloader in zip(self.classifiers, attribute_test_loaders):
        #     classifier_features = []
            
        #     with torch.no_grad():
        #         for batch_x, _ in dataloader:
        #             features = classifier(batch_x, extract_features=True)
        #             classifier_features.append(features)
            
        #     all_features.append(torch.cat(classifier_features, dim=0))
            
        # # Concatenate features
        # total_features = torch.cat(all_features, dim=1)
        
        # # Final prediction
        # with torch.no_grad():
        #     self.fusion_model.eval()
        #     return self.fusion_model(total_features)


# class Config:
#     @classmethod
#     def load_config(cls, config_path: str = None) -> Dict:
#         """Load configuration from YAML file."""
#         if config_path and Path(config_path).exists():
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
                
#                 return config
#         raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
        
# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Horizon Picking Training Pipeline')
    
#     # Mode arguments
#     parser.add_argument('--mode', type=str, choices=['train', 'test'], 
#                        default='train', help='Operation mode')
#     parser.add_argument('--config', type=str, help='Path to config file')
#     parser.add_argument('--device', type=str, default='cuda:1',
#                        help='Device for computation')
    
#     return parser.parse_args()


# def setup_model(config: Dict, num_attributes: int) -> AdvancedEnsembleLearner:
#     """Initialize model with configuration."""
#     return AdvancedEnsembleLearner(
#         dim=config['training']['embed_dims'],
#         num_heads=config['training']['num_heads'],
#         num_classes=config['training']['num_classes'],
#         num_classifiers=num_attributes,
#         height=config['data']['height'],
#         width=config['data']['width'],
#         meta_model_path=config['paths']['meta_model_checkpoint'],
#         fusion_model_path=config['paths']['fusion_model_checkpoint']
#     )


# def train(model: AdvancedEnsembleLearner, 
#          train_loaders: List[Tuple], 
#          val_loaders: List[Tuple],
#          config: Dict) -> None:
#     """Training process."""
#     print("Starting training...")
#     model.train_ensemble(
#         train_loaders, 
#         val_loaders, 
#         epochs=config['training']['num_epochs']
#     )


# def test(model: AdvancedEnsembleLearner, 
#         test_loaders: List[Tuple],
#         output_path: str,
#         config: Dict) -> None:
#     """Testing process."""
#     print("Starting inference...")
#     predictions = model.predict(test_loaders,
#                                 meta_models_dir=config['paths']['meta_model_checkpoint'],
#                                 fusion_model_path=config['paths']['fusion_model_checkpoint'])
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#     np.save(output_path, predictions)


# def main():
#     """Main execution function."""
#     start_time = time.time()
    
#     # Parse arguments and load config
#     args = parse_args()
#     config = Config.load_config(args.config)
    
#     # Initialize data factory
#     data_factory = HorizonDataFactory(
#         attr_dirs=config['data_paths'],
#         kernel_size=config['data']['kernel_size'],
#         stride=config['data']['stride'],
#         batch_size=config['training']['batch_size']
#     )
    
#     # Prepare dataloaders
#     dataloaders = data_factory.get_dataloaders(config['attributes'])
#     train_loaders = [(k, v['train']) for k, v in dataloaders.items()]
#     val_loaders = [(k, v['val']) for k, v in dataloaders.items()]
#     test_loaders = [(k, v['test']) for k, v in dataloaders.items()]
    
#     # Setup model
#     model = setup_model(config, len(dataloaders))
    
#     # Execute based on mode
#     if args.mode == 'train': 
#         train(model, train_loaders, val_loaders, config)
#     else:
#         output_path = Path(config['paths']['output_dir']) / 'predictions.npy'
#         test(model, test_loaders, str(output_path), config)
    
#     elapsed_time = time.time() - start_time
#     print(f"Execution completed in {elapsed_time:.2f} seconds")


# if __name__ == '__main__':
#     main()

def main():
    stime = time.ctime()
    # freq phase seismic dip amp
    # attribute_names = ['freq', 'phase', 'seismic', 'dip', 'amp', 'complex', 'coherence','average_zero','azimuth']
    # attribute_names = [ 'amp', 'complex', 'coherence','average_zero','azimuth']
    attribute_names = ['seismic', 'freq', 'dip', 'phase']
    
    
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
        
        # return 
    
    if args.is_testing:
        print("============ Begin testing ============\n") 

        predictions, combined_attr_name  = ensemble_learner.predict(embed_dims, heads, attribute_test_loaders)
        predictions = predictions.cpu().numpy()
        
        # Create inference directory
        os.makedirs(args.inference_dir, exist_ok=True)  # Now creates ./process/inference directory
        save_path = os.path.join(args.inference_dir, f'{combined_attr_name}_predictions.npy')  # Creates ./process/inference/predictions.npy
        np.save(save_path, predictions)
        print('saved predictions to', save_path)
        
        return  
    
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
    
    parser.add_argument('--width', type=int, default=32, 
                        help='data width size')

    parser.add_argument('--mm_ckpt_path', type=str, default='/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/process/output/meta_model_ckpt', 
                        help='checkpoint saving/loading path of meta model')
    
    parser.add_argument('--fm_ckpt_path', type=str, default='/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/process/output/fusion_model_ckpt', 
                        help='checkpoint saving/loading path of fusion model')

    parser.add_argument('--is_testing', type=bool, default=True,
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
    
    parser.add_argument('--inference_dir', type=str,  default='./process/inference',
                        help='model inference prediction dir')
    
    parser.add_argument('--attr_dirs', type=dict,  default = {
        "seismic": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy", 
                    "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        "freq": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy", 
                 "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
        "dip": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_predict_MCDL_crossline.npy", 
                "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
        
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
    
    