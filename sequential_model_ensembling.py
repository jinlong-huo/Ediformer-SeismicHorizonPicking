import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class IndividualFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class FeatureFusionModel(nn.Module):
    def __init__(self, total_feature_dim: int, num_classes: int):
        super().__init__()
        self.fusion_network = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, fused_features):
        return self.fusion_network(fused_features)

class SequentialEnsembleTrainer:
    def __init__(self, attribute_configs: List[dict], num_classes: int):
        self.feature_extractors = nn.ModuleList([
            IndividualFeatureExtractor(
                input_dim=config['input_dim'], 
                feature_dim=config['feature_dim']
            ) for config in attribute_configs
        ])
        
        self.fusion_model = None
        self.num_classes = num_classes
    
    def train_individual_extractors(self, dataloaders: List, epochs: int = 50):
        """
        Train each feature extractor separately
        
        Args:
            dataloaders: List of dataloaders, one for each attribute set
            epochs: Number of training epochs
        """
        for idx, (extractor, dataloader) in enumerate(zip(self.feature_extractors, dataloaders)):
            print(f"Training Feature Extractor {idx}")
            optimizer = optim.AdamW(extractor.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                extractor.train()
                total_loss = 0
                
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = extractor(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader)}")
    
    def extract_features(self, dataloaders: List):
        """
        Extract features using pre-trained individual models
        
        Args:
            dataloaders: List of dataloaders for feature extraction
        
        Returns:
            List of extracted feature tensors
        """
        features = []
        for extractor, dataloader in zip(self.feature_extractors, dataloaders):
            extractor.eval()
            batch_features = []
            
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    batch_features.append(extractor(batch_x))
            
            features.append(torch.cat(batch_features, dim=0))
        
        return features
    
    def train_fusion_model(self, extracted_features, labels, epochs: int = 30):
        """
        Train fusion model using extracted features
        
        Args:
            extracted_features: List of feature tensors
            labels: Ground truth labels
            epochs: Number of training epochs
        """
        # Concatenate features
        total_features = torch.cat(extracted_features, dim=1)
        
        # Create fusion model
        self.fusion_model = FeatureFusionModel(
            total_feature_dim=total_features.size(1), 
            num_classes=self.num_classes
        )
        
        optimizer = optim.AdamW(self.fusion_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.fusion_model.train()
            optimizer.zero_grad()
            
            outputs = self.fusion_model(total_features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            print(f"Fusion Model Epoch {epoch}, Loss: {loss.item()}")
    
    def predict(self, test_dataloaders):
        """
        Predict using the entire ensemble pipeline
        
        Args:
            test_dataloaders: List of test dataloaders
        
        Returns:
            Predictions
        """
        test_features = self.extract_features(test_dataloaders)
        total_test_features = torch.cat(test_features, dim=1)
        
        with torch.no_grad():
            self.fusion_model.eval()
            return self.fusion_model(total_test_features)

# Example usage would involve preparing your specific dataloaders
# and configuring the ensemble trainer accordingly