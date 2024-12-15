import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class FeatureFusionModel(nn.Module):
    """Advanced Feature Fusion Model with U-Net-like Architecture"""
    def __init__(self, total_feature_dim: int, num_classes: int, fusion_height: int, fusion_width: int):
        super().__init__()
        
        # Initial feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Reshape and expand features
        self.reshape_layer = nn.Linear(512, 512 * fusion_height * fusion_width)
        
        # Encoder-Decoder Blocks
        self.encoder_blocks = nn.ModuleList([
            DoubleConv(1, 64),   # Initial block
            DoubleConv(64, 128), # Deeper block
            DoubleConv(128, 256) # Deepest block
        ])
        
        # Decoder blocks with upsampling
        self.decoder_blocks = nn.ModuleList([
            DoubleConv(256, 128),
            DoubleConv(128, 64)
        ])
        
        # Final classification layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Store parameters
        self.num_classes = num_classes
        self.height = fusion_height
        self.width = fusion_width

    def forward(self, fused_features):
        # Project input features
        x = self.feature_projection(fused_features)
        
        # Reshape to 2D feature map
        x = self.reshape_layer(x)
        x = x.view(-1, 1, self.height, self.width)
        
        # Encoder path
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = F.max_pool2d(x, 2)
        
        # Decoder path with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat([x, encoder_outputs[-(i+2)]], dim=1)
            x = block(x)
        
        # Final classification layer
        x = self.final_conv(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Example instantiation
    total_feature_dim = 1024
    num_classes = 10
    fusion_height = 32
    fusion_width = 32
    
    model = FeatureFusionModel(total_feature_dim, num_classes, fusion_height, fusion_width)
    
    # Create a sample input
    sample_input = torch.randn(4, total_feature_dim)
    
    # Forward pass
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")