import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
# import tensorflow as tf
import json
import numpy as np


# 权重初始化
def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0) 
        # torch.nn.init.kaiming_normal(m.weight)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0) 
        # torch.nn.init.kaiming_normal(m.weight)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  #


class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=1,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=1,
                               stride=1, padding=1)
        self.relu = nn.ReLU()   

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=(1,3), stride=1*1, padding=(0,2), bias=True)),     
        self.add_module('norm1', nn.BatchNorm2d(out_features)),                                          
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=(1,3), stride=1*1, bias=True)),                    
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))  
        
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):  
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]    
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, (1, kernel_size), stride=(2, 2), padding=(0, pad))) #gaile kernal_size
            in_features = out_features
            
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, # out_features --7
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module): 
    def       __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,  
                               (1, 3), padding=(0, 1), stride=stride) 
        
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, (1,3), padding=(0, 1)) # 
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.dropout = nn.Dropout(p=0.5) 
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)

        return x



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, kernel_size=16,  stride=16, padding=0, in_chans=3, embed_dim=768):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = x.permute(0,2,3,1)
        return x

class FirstPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, kernel_size=3,  stride=2, padding=1, in_chans=3, embed_dim=768):
        super().__init__()
        
        self.proj1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm1 = nn.BatchNorm2d(embed_dim // 2)
        self.gelu1 = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding )
        self.norm2 = nn.BatchNorm2d(embed_dim)
        
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.proj2(x)
        x = self.norm2(x)    
        x = x.permute(0,2,3,1)
        return x
    
class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
        **kwargs, ):
        super().__init__()
        
        self.cnn_in = cnn_in = dim // 2
        self.pool_in = pool_in = dim // 2
        
        self.cnn_dim = cnn_dim = cnn_in * 2
        self.pool_dim = pool_dim = pool_in * 2

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()
       
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2d(pool_in, pool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

    def forward(self, x):
        # B, C H, W
        
        cx = x[:,:self.cnn_in,:,:].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)
        
        px = x[:,self.cnn_in:,:,:].contiguous()
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)
        
        hx = torch.cat((cx, px), dim=1)
        return hx

class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
        **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()
        

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        xa = self.pool(x)
        xa = xa.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N**0.5), int(N**0.5))#.permute(0, 3, 1, 2)
        
        xa = self.uppool(xa)
        return xa

class Mixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2, 
        **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        
        self.low_dim = low_dim = attention_head * head_dim
        self.high_dim = high_dim = dim - low_dim
        
        
        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size,)

        self.conv_fuse = nn.Conv2d(low_dim+high_dim*2, low_dim+high_dim*2, kernel_size=3, stride=1, padding=1, bias=False, groups=low_dim+high_dim*2)
        self.proj = nn.Conv2d(low_dim+high_dim*2, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        
        hx = x[:,:self.high_dim,:,:].contiguous()
        hx = self.high_mixer(hx)
        
        lx = x[:,self.high_dim:,:,:].contiguous()
        lx = self.low_mixer(lx)
            
        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class DexiNed(nn.Module):
    """ Definition of the DXtrem network. """

    # def __init__(self):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., drop_path=0.,
                 attention_head=1, pool_size=1, norm_layer=nn.LayerNorm,
                 attn=Mixer):
        
        super(DexiNed, self).__init__()                           
        self.block_1 = DoubleConvBlock(1, 32, 64, stride=2,)  # Change into 1 Channel
        self.block_2 = DoubleConvBlock(64, 128, stride=1, use_act=False)
        # self.dblock_3 = _DenseBlock(2, 128, 256) # [128,256,100,100]
        # self.dblock_4 = _DenseBlock(3, 256, 512)
        # self.dblock_5 = _DenseBlock(3, 512, 512)
        # self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)  # stride-->(1, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        # right skip connections, figure in Journal paper
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)

        # USNet
        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 7, stride=1, use_bs=False)  # hed fusion method
        
        self.norm1 = norm_layer(dim[0])
        self.norm2 = norm_layer(dim[1])
        self.norm3 = norm_layer(dim[2])
        self.norm4 = norm_layer(dim[3])
        
        self.attn = attn(dim[0], num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, attention_head=attention_head, pool_size=pool_size,)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.apply(weight_init)
        # self.linear = torch.nn.Linear(288, 7)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor
        # tensor[..., :height, :width]
        return new_tensor

    def forward(self, x): 
        assert x.ndim == 4, x.shape
       
        block_1 = self.block_1(x)
        
        block_1_side = self.side_1(block_1)
       
        # Block 2
        block_2 = self.block_2(block_1)
        
        block_2_down = self.maxpool(block_2)
       
        block_2_add = block_2_down + block_1_side  
        block_2_side = self.side_2(block_2_add)
        
        # ============= ================
        # -------update with mixer------
        # ============= ================
        
        # # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        # block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])    # original dense block
        # block_2_add shape [2, 128, 1, 72] here we can choose use scale layer or not
        # block_3_pre_dense [2, 256, 1, 72]
        block_3 = block_2_add + self.drop_path(self.attn(self.norm1(block_3_pre_dense)))
        block_3_down = self.maxpool(block_3)         
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)
      
        # Block 4
        block_4_pre_dense = self.pre_dense_4(block_3_down)
        # block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4 = block_3_add + self.drop_path(self.attn(self.norm2(block_4_pre_dense)))
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(
            block_4_down) #block_5_pre_dense_512 +block_4_down
        # block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5 = block_4_add + self.drop_path(self.attn(self.norm3(block_5_pre_dense)))
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        # block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])
        block_6 = block_5_add + self.drop_path(self.attn(self.norm(block_6_pre_dense)))


        # upsampling blocks

        out_1 = self.up_block_1(block_1)
        

        out_2 = self.up_block_2(block_2)
        

        out_3 = self.up_block_3(block_3)
        

        out_4 = self.up_block_4(block_4)
        

        out_5 = self.up_block_5(block_5)
        

        out_6 = self.up_block_6(block_6)       
        

        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs      
        block_cat = torch.cat(results, dim=1)  # 

        block_cat = self.block_cat(block_cat)  # Bx1xHxW# 

        # block_cat = self.conv3(block_cat)      # Bx1xHxW 
        # return results
        results.append(block_cat)
        # print('**********',block_cat.shape)
        # another linear layer for test
        # results[6] = results[6].view(-1, 288)
        # results[6] = F.relu(self.linear(results[6]))
        return results


if __name__ == '__main__':
    batch_size = 2                                                                  
    img_height = 1    # 1   352
    img_width = 288   # 288 352
    
    # print
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    input = torch.rand(batch_size, 1, img_height, img_width).to(device)             
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    embed_dims = [72, 72, 72, 72] # embed dim
    heads = 2
    model = DexiNed(dim=embed_dims, num_heads=heads).to(device)
    output = model(input)
    # print(output[0].shape)
    print(f"output shapes: {[t.shape for t in output]}")

