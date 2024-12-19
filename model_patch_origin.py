import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
# import tensorflow as tf
import json
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=1, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze操作的定义
        # print(channel, channel // reduction)
        self.fc = nn.Sequential(  # Excitation操作的定义
            nn.Linear(channel, channel // reduction, bias=False),  # 压缩
            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),  # 恢复
            nn.Sigmoid()  # 定义归一化操作
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 得到H和W的维度，在这两个维度上进行全局池化
        y = self.avg_pool(x).view(b, c)  # Squeeze操作的实现
        y = self.fc(y).view(b, c, 1, 1)  # Excitation操作的实现
        # 将y扩展到x相同大小的维度后进行赋权
        return x * y.expand_as(x)

# 权重初始化
def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0) # 这里改过
        # torch.nn.init.kaiming_normal(m.weight)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0) # 这里改过
        # torch.nn.init.kaiming_normal(m.weight)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  #


# CoFusion也不需要
class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=1,
                               stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1,
                               stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=1,
                               stride=1, padding=0)
        self.relu = nn.ReLU()   # ReLU的激活层

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        # print(attn.shape,'1')
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        # print(attn.shape,'2')
        attn = F.softmax(self.conv3(attn), dim=1)
        # print(x.shape,attn.shape,'3')
        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=(3,3), stride=1*1, padding=(2), bias=True)),     # 学长说把这里的kernal_size进行修改 我改成了（1，3）
        self.add_module('norm1', nn.BatchNorm2d(out_features)),                                           # 从3改为1 padding改变数据的大小 从原始的2 改为 1(误)
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=(3,3), stride=1*1, bias=True)),                    # 同理这里的kernal_size也改成了（1，3）而不是（1*3）
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()  #x2 和 new_features 是同维的
        # if new_features.shape[-1]!=x2.shape[-1]:
        #     new_features =F.interpolate(new_features,size=(x2.shape[2],x2.shape[-1]), mode='bicubic',
        #                                 align_corners=False)
        # print(x1.shape,x2.shape,new_features.shape)
        # print(new_features.shape, x1.shape, x2.shape) # x2的shape和new_features的shape的第二个参数也就是H不一样 现在看来是block_2的H不对一会改一下
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):  # num_layers 分别是2 3 3 .......
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
        all_pads=[0,0,1,3,7]    # pad选取
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1

            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, (kernel_size), stride=(2, 2), padding=(pad))) #gaile kernal_size
            in_features = out_features
            # print(layers)
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
        # print(out_features)
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):  # kernalsize 改了 padding部分改了
    def       __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,  # in_features 3 mid_features  32    64  128
                               (3), padding=(1), stride=stride) # 这里的（1，3）就是kernal_size # channels 3-1  kernal_size 改成了1
        # print(in_features, mid_features)
        self.bn1 = nn.BatchNorm2d(mid_features)

        self.conv2 = nn.Conv2d(mid_features, out_features, (3), padding=(1)) # 这里的（1，3）就是kernal_size
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # self.dropout = nn.Dropout(p=0.5)  # dropout训练
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)

        return x


class DexiNed(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(DexiNed, self).__init__()
        self.block_1 = DoubleConvBlock(1, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock(64, 128, stride=1, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)  # 这里的stride改为了(1, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
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

        # self.conv3 = nn.Conv2d(in_channels=7, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True) 这里先不改
        self.block_cat = CoFusion(6, 6)# cats fusion method
        self.block_cat = SingleConvBlock(6, 7, stride=1, use_bs=False)  # hed fusion method
        self.block_cat2 = SingleConvBlock(18, 6, stride=1, use_bs=False)
    

        self.apply(weight_init)

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

    def forward(self, x): # 前馈网络就对应网络图
        # print(x.shape)
        assert x.ndim == 4, x.shape
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side  # block_2_down的维度和block_2_down是一样的
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])    # 问题出现在了这里 dblock这里有点问题
        block_3_down = self.maxpool(block_3) # [128,256,50,50]
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)


        # Block 4
        block_4_pre_dense = self.pre_dense_4(block_3_down)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)


        # Block 5
        block_5_pre_dense = self.pre_dense_5(block_4_down) #block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks

        out_1 = self.up_block_1(block_1)
        # print(block_1.shape, out_1.shape)

        out_2 = self.up_block_2(block_2)
        # print(block_2.shape, out_2.shape)

        out_3 = self.up_block_3(block_3)
        # print(block_3.shape, out_3.shape)

        out_4 = self.up_block_4(block_4)
        # print(block_4.shape, out_4.shape)

        out_5 = self.up_block_5(block_5)
        # print(block_5.shape, out_5.shape)

        out_6 = self.up_block_6(block_6)       # 在这里所有的通道数都是1 而 高度和宽度都是设置值
        # print(block_6.shape, out_6.shape)

        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs       # 拼接通道所以要保证其他相同 dim = 1 ，但是在这里H不统一
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW 按列拼接所以行数要相同  问题out_*的shape H部分不一致
        # block_cat = self.cbam_block(block_cat)
        block_cat = self.block_cat(block_cat)  # Bx1xHxW # 1*1卷积 通道数改变

        results.append(block_cat)
        
        return results


class SimpleConvBlock(nn.Module):
    def __init__(self):
        super(SimpleConvBlock, self).__init__()
        # self.block_cat = SingleConvBlock(6, 18, stride=1, use_bs=False)  # hed fusion method
        self.cbam_block = cbam_block(3)
        self.block_cat1 = SingleConvBlock(3, 6, stride=1, use_bs=False)

    def forward(self, x):
        results = self.cbam_block(x)
        x = self.block_cat1(results)
        return x


if __name__ == '__main__':  # patch的效果应该是 input 20 1 1 288--> 20 7 1 288 变成 20 1 5 288-->20 7 5 288
                            # patch--5不容易改 input 20 1 1 288--> 20 7 1 288 变成 20 1 16 288-->20 7 16 288（暂时妥协）
    batch_size = 20                                                                  # batch_size = 8
    img_height = 128    # 1   352
    img_width = 128   # 288 352
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    input = torch.rand(batch_size, 1, 16, 288).to(device)             #这里是batch_size的设置 C H W 1 1 288
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")

    model = DexiNed().to(device)
    output = model(input)
    print(f"output shapes: {[t.shape for t in output]}")


# import torch.nn.functional as F

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
#         self.fc1 = nn.Linear(64*7*7,1024)
#         self.fc2 = nn.Linear(1024,512)
#         self.fc3 = nn.Linear(512,10)
#
#
#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#
#         x = x.view(-1, 64 * 7* 7)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x
#
#
#
# batch_size = 20  # batch_size = 8
# img_height = 16  # 1   352
# img_width = 288  # 288 352
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # device = "cpu"
# input = torch.rand(batch_size, 1, 16, img_width)
#
# net = CNN()
# output_cnn = net(input)
# print(output_cnn.shape)
