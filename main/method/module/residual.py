import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    """
    残差模块类，用于实现残差连接。
    """

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


if __name__ == "__main__":
    block = nn.Sequential(
        nn.Conv2d(2048, 2048, 1, 1, 0),
        nn.GELU(),
        nn.BatchNorm2d(2048),
    )
    model = Residual(block)
    x = torch.rand(2, 2048, 16, 8)
    y = model(x)
    print(model)
    print(y.shape)
