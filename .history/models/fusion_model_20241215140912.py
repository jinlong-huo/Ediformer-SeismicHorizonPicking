import torch
import torch.nn as nn

class FeatureFusionModel(nn.Module):
    """
    Final model to fuse features from meta-models with a U-Net like structure.
    """
    def __init__(self, total_feature_dim: int, num_classes: int, fusion_height: int, fusion_width: int):
        super().__init__()

        # Downsampling path (encoder)
        self.conv1 = nn.Conv2d(total_feature_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        # Bottleneck (middle layer)
        self.bottleneck = nn.Sequential(
            nn.Linear(512 * fusion_height // 8 * fusion_width // 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512)
        )

        # Upsampling path (decoder)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(64 + 64, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, fused_features):
        # Downsampling path
        x1 = self.conv1(fused_features)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Flatten for the bottleneck layer
        x = torch.flatten(x4, 1)
        x = self.bottleneck(x)

        # Upsampling path with skip connections
        x = x.view(-1, 512, fusion_height // 8, fusion_width // 8)
        x = self.upconv1(x) + x3
        x = self.upconv2(x) + x2
        x = self.upconv3(x) + x1
        x = self.upconv4(x)

        return x