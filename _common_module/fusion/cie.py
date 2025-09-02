import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class CIE(nn.Module):
    def __init__(self, in_channel):
        super(CIE, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, f1, f2):

        f = f1.mul(f2)
        f = self.relu(self.bn(self.conv(f)))
        f = self.maxpool(self.upsample(f))

        f1 = f + f1
        f2 = f + f2
        f1 = self.maxpool(self.upsample(f1))
        f2 = self.maxpool(self.upsample(f2))

        f = f1 + f2

        return f


if __name__ == "__main__":
    # 创建模型参数

    # 创建输入数据
    inputs1 = torch.randn(2, 2048, 18, 9)
    inputs2 = torch.randn(2, 2048, 18, 9)

    # 初始化模型
    model = CIE(in_channel=2048)

    # 前向传播
    outputs = model(inputs1, inputs2)

    # # 打印输出形状
    print(outputs.shape)
