import torch
import torch.nn as nn

class UNetFusionModel(nn.Module):
    """ Fusion model inspired by U-Net architecture """
    def __init__(self, total_feature_dim: int, num_classes: int, fusion_height: int, fusion_width: int):
        super(UNetFusionModel, self).__init__()
        self.num_classes = num_classes
        self.height = fusion_height
        self.width = fusion_width

        # Initial convolution to adjust the number of input channels
        self.initial_projection = nn.Sequential(
            nn.Conv2d(total_feature_dim, 64, kernel_size=1),  # 1x1 conv to adjust channel size
            nn.ReLU()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Convolve to get 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample again
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample again
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)  # Final output layer
        )

    def forward(self, fused_features):
        # Step 1: Project 2D features to an adjusted channel space (using 1x1 convolution)
        x = self.initial_projection(fused_features)
        
        # Step 2: Pass through encoder
        enc_out = self.encoder(x)
        
        # Step 3: Bottleneck
        bottleneck_out = self.bottleneck(enc_out)
        
        # Step 4: Pass through decoder
        dec_out = self.decoder(bottleneck_out)
        
        return dec_out

# Example usage
if __name__ == "__main__":
    # Example instantiation
    total_feature_dim = 1024
    num_classes = 7
    fusion_height = 16
    fusion_width = 288
    
    model = UNetFusionModel(total_feature_dim, num_classes, fusion_height, fusion_width)
    
    # Create a sample input
    sample_input = torch.randn(4, total_feature_dim)
    
    # Forward pass
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")