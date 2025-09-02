"""

https://arxiv.org/pdf/2009.14082

https://0809zheng.github.io/2020/12/01/aff.html

"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = x + y
        xg = self.global_att(xy)
        wei = self.sigmoid(xg)
        xo = x * wei + y * (1 - wei)
        return xo


if __name__ == "__main__":
    # 创建模型参数

    # 创建输入数据
    inputs1 = torch.randn(2, 2048, 18, 9)
    inputs2 = torch.randn(2, 2048, 18, 9)

    # 初始化模型
    model = AFF(channels=2048)

    # 前向传播
    outputs = model(inputs1, inputs2)

    # # 打印输出形状
    print(outputs.shape)
