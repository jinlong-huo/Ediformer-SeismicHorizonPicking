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
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=1,
                               stride=1, padding=1)
        self.relu = nn.ReLU()   # ReLU的激活层

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)

# _DenseLayer里需要修改
class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=(1,3), stride=1*1, padding=(0,2), bias=True)),     # 学长说把这里的kernal_size进行修改 我改成了（1，3）
        self.add_module('norm1', nn.BatchNorm2d(out_features)),                                           # 从3改为1 padding改变数据的大小 从原始的2 改为 1(误)
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=(1,3), stride=1*1, bias=True)),                    # 同理这里的kernal_size也改成了（1，3）而不是（1*3）
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
                out_features, out_features, (1, kernel_size), stride=(2, 2), padding=(0, pad))) #gaile kernal_size
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
                               (1, 3), padding=(0, 1), stride=stride) # 这里的（1，3）就是kernal_size # channels 3-1  kernal_size 改成了1
        # print(in_features, mid_features)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, (1,3), padding=(0, 1)) # 这里的（1，3）就是kernal_size
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)

        return x


class DexiNed(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(DexiNed, self).__init__()                           # 权重部分也要改(待定)
        self.block_1 = DoubleConvBlock(1, 32, 64, stride=2,)  # 3是通道数 32 和 64 分别是输入特征和输出特征 其实应该要把通道数改为1的因为我们是单通道
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
        # self.conv3 = nn.Conv2d(in_channels=7, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True) 这里先不改
        # self.block_cat = CoFusion(6,6)# cats fusion method

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

    def forward(self, x): # 前馈网络就对应网络图
        assert x.ndim == 4, x.shape
        #  print(x.ndim, x.shape) x 的维度是4 shape是【8，3，1，288】 也就是需要把这里改成我们的一维数据
        # Block 1
        # print(x.shape)
        block_1 = self.block_1(x)
        # print(block_1.shape)
        # print("block_1.shape:", block_1.shape)            # 正常情况下应该是 8，144 通过上采样之后变成16，288 所以这里的H出现了问题
        block_1_side = self.side_1(block_1)
        # print(x.shape, block_1.shape) # 这里一开始block_1_side的size和 block_2down的size不同，原因在于side_1 的 单卷积块里面的stride是2，改成（1，2）
        # print(block_1_side.shape)  # 查看block1的大小 通道数变成了64 高度宽度降低一半

        # Block 2
        block_2 = self.block_2(block_1)
        # print(block_2.shape)
        # print("block_2.shape:", block_2.shape)
        block_2_down = self.maxpool(block_2)
        # print(block_2_down.shape, block_1_side.shape)
        block_2_add = block_2_down + block_1_side  # block_2_down的维度和block_2_down是一样的
        block_2_side = self.side_2(block_2_add)
        # print("*"*88)
        # print("block_2_side.shape:               ", block_2_side.shape)  # 查看block_2_side的大小

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        # print("block_2_add.shape:                ", block_2_add.shape)
        # print("block_3_pre_dense.shape:          ", block_3_pre_dense.shape)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])    # 问题出现在了这里 dblock这里有点问题
        # print("block_3.shape:                ", block_3.shape)
        block_3_down = self.maxpool(block_3) # [128,256,50,50]
        # print("block_3_down.shape:                ", block_3_down.shape)   # 查看block_3的大小
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)
        # print(block_3.shape)  # 查看block2的大小

        # Block 4
        block_4_pre_dense = self.pre_dense_4(block_3_down)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(
            block_4_down) #block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # print(block_1.shape)
        # print(block_2.shape)
        # print(block_3.shape)
        # print(block_4.shape)
        # print(block_5.shape)
        # print(block_6.shape)


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

        block_cat = self.block_cat(block_cat)  # Bx1xHxW # 1*1卷积 通道数改变

        # block_cat = self.conv3(block_cat)      # Bx1xHxW # 这里先不改变网络模式
        # return results
        results.append(block_cat)
        # print('**********',block_cat.shape)
        # 下面是自己加的一个线性层
        # results[6] = results[6].view(-1, 288)
        # results[6] = F.relu(self.linear(results[6]))
        return results


if __name__ == '__main__':
    batch_size = 20                                                                  # batch_size = 8
    img_height = 1    # 1   352
    img_width = 288   # 288 352
    # print
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    input = torch.rand(batch_size, 1, img_height, img_width).to(device)             #这里是batch_size的设置 C H W 1 1 288
    # target = torch.rand(batch_size, 1, img_height, img_width).to(device)
    print(f"input shape: {input.shape}")
    model = DexiNed().to(device)
    output = model(input)
    # print(output[0].shape)
    print(f"output shapes: {[t.shape for t in output]}")

