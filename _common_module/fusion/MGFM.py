"""
Paper: https://www.sciencedirect.com/science/article/pii/S1569843224005971#sec3
Code: https://github.com/yeyuanxin110/YESeg-OPT-SAR/blob/main/code/MGFNet/scripts/models/MGFNet.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return y


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MGFM(nn.Module):
    def __init__(self, in_channels):
        super(MGFM, self).__init__()
        self.in_channel = in_channels
        self.eca_x = eca_block(channel=self.in_channel)
        self.eca_y = eca_block(channel=self.in_channel)
        self.mlp_x = Mlp(in_features=self.in_channel * 2, out_features=self.in_channel)
        self.mlp_y = Mlp(in_features=self.in_channel * 2, out_features=self.in_channel)
        self.sigmoid = nn.Sigmoid()

        self.mlp = Mlp(in_features=in_channels, out_features=in_channels)

    def forward(self, opt, sar):

        # Fusion-Stage-1 ECA Channel Attention
        w_opt = self.eca_x(opt)
        w_sar = self.eca_y(sar)
        N, C, H, W = w_opt.shape

        w_opt = torch.flatten(w_opt, 1)
        w_sar = torch.flatten(w_sar, 1)

        w = torch.concat([w_opt, w_sar], 1)

        # Fusion-Stage-2 MLP
        w1 = self.mlp_x(w)
        w1 = self.sigmoid(w1.reshape([N, self.in_channel, H, W]))

        w2 = self.mlp_y(w)
        w2 = self.sigmoid(w2.reshape([N, self.in_channel, H, W]))

        # Gating-Stage
        out1 = opt * w1
        out2 = sar * w2
        f = torch.cat((out1, out2), 1)

        return f


if __name__ == "__main__":
    # 创建模型参数

    # 创建输入数据
    inputs1 = torch.randn(2, 2048, 18, 9)
    inputs2 = torch.randn(2, 2048, 18, 9)

    # 初始化模型
    model = MGFM(in_channels=2048)

    # 前向传播
    outputs = model(inputs1, inputs2)

    # # 打印输出形状
    print(outputs.shape)
