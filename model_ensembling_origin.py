import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict

class SeismicAttributeClassifier(nn.Module):
    """
    Individual Classifier for a specific seismic attribute
    Now includes a feature extraction method
    """
    def __init__(self, input_dim: int, feature_dim: int = 128, hidden_dims: List[int] = [256, 128], num_classes: int = 2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Separate feature extraction and classification layers
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_projection = nn.Linear(prev_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, extract_features: bool = False):
        features = self.feature_extractor(x)
        projected_features = self.feature_projection(features)
        
        if extract_features:
            return projected_features
        
        return self.classifier(projected_features)

class FeatureFusionModel(nn.Module):
    """
    Final model to fuse features from meta-models
    """
    def __init__(self, total_feature_dim: int, num_classes: int = 2):
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
    def __init__(self, attribute_configs: List[Dict], num_classes: int = 2):
        self.num_classes = num_classes
        
        # Initialize classifiers for each attribute
        self.classifiers = nn.ModuleList([
            SeismicAttributeClassifier(
                input_dim=config['input_dim'], 
                feature_dim=config.get('feature_dim', 128),
                num_classes=num_classes
            ) for config in attribute_configs
        ])
        
        # Initialize fusion model
        total_feature_dim = sum(
            classifier.feature_projection.out_features 
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
            # Stage 1: Train individual meta-models
            for idx, (classifier, dataloader, optimizer) in enumerate(
                zip(self.classifiers, attribute_dataloaders, classifier_optimizers)
            ):
                classifier.train()
                total_loss = 0
                
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Compute loss for individual classifier
                    outputs = classifier(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Meta-Model {idx} Epoch {epoch}, Loss: {total_loss/len(dataloader)}")
            
            # Stage 2: Train fusion model
            self.fusion_model.train()
            all_features = []
            all_labels = []
            
            # Extract features from meta-models
            for classifier, dataloader in zip(self.classifiers, attribute_dataloaders):
                classifier_features = []
                classifier_labels = []
                
                for batch_x, batch_y in dataloader:
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
    attribute_configs = [
        {'input_dim': 100, 'feature_dim': 128},  # Attribute 1
        {'input_dim': 150, 'feature_dim': 128},  # Attribute 2
        {'input_dim': 200, 'feature_dim': 128}   # Attribute 3
    ]
    
    # Initialize Advanced Ensemble Learner
    ensemble_learner = AdvancedEnsembleLearner(
        attribute_configs=attribute_configs, 
        num_classes=2
    )
    
    # Simulated dataloaders for each attribute
    attribute_dataloaders = [...]
    
    # Train ensemble
    ensemble_learner.train_ensemble(attribute_dataloaders)
    
    # Predict
    test_dataloaders = [...]
    predictions = ensemble_learner.predict(test_dataloaders)

if __name__ == "__main__":
    main()