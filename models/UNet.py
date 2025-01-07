import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.up(x)

class UNetClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        """
        UNet-style neural network classifier adapted for 1D feature input.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Base number of hidden units
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Encoder path
        self.down1 = DownBlock(input_dim, hidden_dim)
        self.down2 = DownBlock(hidden_dim, hidden_dim * 2)
        self.down3 = DownBlock(hidden_dim * 2, hidden_dim * 4)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Decoder path
        self.up1 = UpBlock(hidden_dim * 12, hidden_dim * 4)  # 8 + 4 = 12
        self.up2 = UpBlock(hidden_dim * 6, hidden_dim * 2)   # 4 + 2 = 6
        self.up3 = UpBlock(hidden_dim * 3, hidden_dim)       # 2 + 1 = 3
        
        # Output layer
        self.final = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        
        # Bridge
        bridge = self.bridge(d3)
        
        # Decoder with skip connections
        u1 = self.up1(bridge, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        
        # Output
        return self.final(u3)

# # Example usage:
# if __name__ == "__main__":
#     # Test the model
#     input_dim = 8  # number of features
#     hidden_dim = 32
#     num_classes = 7
#     batch_size = 16
    
#     model = UNetClassifier(input_dim, hidden_dim, num_classes)
#     x = torch.randn(batch_size, input_dim)
#     output = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")  # Should be [batch_size, num_classes]