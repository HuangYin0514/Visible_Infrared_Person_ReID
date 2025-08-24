import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Patch_Embedding(nn.Module):

    def __init__(self, in_cdim=3, out_cdim=768):
        super().__init__()
        self.proj = nn.Linear(in_cdim, out_cdim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        return x


if __name__ == "__main__":

    # 创建输入数据
    inputs = torch.randn(2, 2048, 18, 9)
    print("input.shape", inputs.shape)

    # 初始化模型
    model = Patch_Embedding(in_cdim=2048, out_cdim=768)

    # 前向传播
    outputs = model(inputs)

    # # 打印输出形状
    print(outputs.shape)
