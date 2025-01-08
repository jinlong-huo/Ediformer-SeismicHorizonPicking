import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightChannelAttention(nn.Module):
    """Memory-efficient channel attention using grouped convolutions"""
    def __init__(self, in_channels, reduction_ratio=16, groups=8):
        super(LightweightChannelAttention, self).__init__()
        self.groups = groups
        group_channels = in_channels // groups
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv = nn.Conv1d(groups, groups, kernel_size=1, groups=groups)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        # Group-wise pooling
        pooled = self.avg_pool(x).view(b, self.groups, -1)
        # Efficient group convolution
        attention = self.group_conv(pooled)
        attention = self.sigmoid(attention).view(b, c, 1, 1)
        return attention

class EfficientSpatialAttention(nn.Module):
    """Memory-efficient spatial attention using separable convolutions"""
    def __init__(self, kernel_size=7):
        super(EfficientSpatialAttention, self).__init__()
        self.depth_conv = nn.Conv2d(2, 2, kernel_size, padding=kernel_size//2, groups=2)
        self.point_conv = nn.Conv2d(2, 1, 1)
        
    def forward(self, x):
        # Efficient pooling
        avg_pool = F.adaptive_avg_pool2d(x, (x.size(2), x.size(3)))
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pool.mean(dim=1, keepdim=True), max_pool], dim=1)
        
        # Separable convolution for attention
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return torch.sigmoid(x)

class SparseCrossAttention(nn.Module):
    """Memory-efficient cross attention using sparse attention patterns"""
    def __init__(self, channels, block_size=8, sparse_factor=4):
        super(SparseCrossAttention, self).__init__()
        self.block_size = block_size
        self.sparse_factor = sparse_factor
        self.channels = channels
        
        # Keep the same reduced dimension for Q, K, V
        self.scale_factor = channels ** -0.5
        self.qkv_conv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.output_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, skip):
        B, C, H, W = x.size()
        N = H * W
        
        # Generate Q from x and K,V from skip
        q = self.qkv_conv(x).chunk(3, dim=1)[0]
        k, v = self.qkv_conv(skip).chunk(3, dim=1)[1:]
        
        # Reshape to sequence form
        q = q.view(B, C, N).transpose(-2, -1)  # B, N, C
        k = k.view(B, C, N)  # B, C, N
        v = v.view(B, C, N)  # B, C, N
        
        # Compute attention scores
        attn = torch.bmm(q, k) * self.scale_factor  # B, N, N
        
        # Sparsify attention
        if self.sparse_factor > 1:
            topk = N // self.sparse_factor
            top_values, _ = torch.topk(attn, topk, dim=-1)
            threshold = top_values[..., -1:]
            attn = attn * (attn >= threshold)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn, v.transpose(-2, -1))  # B, N, C
        out = out.transpose(-2, -1).view(B, C, H, W)  # B, C, H, W
        out = self.output_conv(out)
        
        return self.gamma * out + x
    
    def _reshape_to_blocks(self, x):
        B, C, H, W = x.size()
        blocks_h = H // self.block_size
        blocks_w = W // self.block_size
        # Reshape into blocks for sparse attention
        x = x.view(B, C, blocks_h, self.block_size, blocks_w, self.block_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * blocks_h * blocks_w, C, -1)
        return x
    
    def _reshape_from_blocks(self, x, B, C, H, W):
        blocks_h = H // self.block_size
        blocks_w = W // self.block_size
        x = x.view(B, blocks_h, blocks_w, C, self.block_size, self.block_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x
    
    def _sparsify_attention(self, attention):
        # Keep only top-k values per row
        k = attention.size(-1) // self.sparse_factor
        topk_values, _ = torch.topk(attention, k, dim=-1)
        threshold = topk_values[..., -1:]
        return attention * (attention >= threshold)

class EfficientAttentiveConvBlock(nn.Module):
    """Memory-efficient attentive convolution block"""
    def __init__(self, in_channels, out_channels):
        super(EfficientAttentiveConvBlock, self).__init__()
        # Use depthwise separable convolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.channel_attention = LightweightChannelAttention(out_channels)
        self.spatial_attention = EfficientSpatialAttention()
        
    def forward(self, x):
        x = self.conv(x)
        # Sequential attention application
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class MemoryEfficientUNetFusion(nn.Module):
    """Memory-efficient UNet with sparse attention mechanisms"""
    def __init__(self, total_feature_dim: int, num_classes: int, fusion_height: int, fusion_width: int):
        super(MemoryEfficientUNetFusion, self).__init__()
        self.num_classes = num_classes
        self.height = fusion_height
        self.width = fusion_width
        
        # Initial projection with efficient attention
        self.initial_projection = EfficientAttentiveConvBlock(total_feature_dim, 64)
        
        # Encoder with efficient attention blocks
        self.enc1 = EfficientAttentiveConvBlock(64, 128)
        self.enc2 = EfficientAttentiveConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = EfficientAttentiveConvBlock(256, 512)
        
        # Sparse cross-attention for skip connections
        self.cross_att1 = SparseCrossAttention(256)
        self.cross_att2 = SparseCrossAttention(128)
        
        # Decoder path
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = EfficientAttentiveConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = EfficientAttentiveConvBlock(256, 128)
        
        # Efficient final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
        
    def forward(self, x):
        # Initial projection
        x = self.initial_projection(x)
        
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder path with sparse cross-attention
        d1 = self.up1(b)
        e2 = self.cross_att1(e2, d1)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        e1 = self.cross_att2(e1, d2)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        # Final convolution
        out = self.final_conv(d2)
        
        return out

if __name__ == "__main__":
    # Test the memory-efficient model
    total_feature_dim = 14
    num_classes = 7
    fusion_height = 16
    fusion_width = 288
    
    model = MemoryEfficientUNetFusion(
        total_feature_dim=total_feature_dim,
        num_classes=num_classes,
        fusion_height=fusion_height,
        fusion_width=fusion_width
    )
    
    # Test input
    x = torch.randn(4, 14, 16, 288)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")