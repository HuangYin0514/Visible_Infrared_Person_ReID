import torch
import torch.nn as nn


# 你的模块
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  # [B, H/patch, W/patch, C]
        if self.norm is not None:
            x = self.norm(x)
        return x


# ===== 测试 =====
if __name__ == "__main__":
    # 模拟一个 batch=2 的输入图像，大小 32x32, 通道=3
    x = torch.randn(2, 3, 16, 8)

    # 实例化 patch embedding
    patch_embed = PatchEmbed2D(patch_size=4, in_chans=3, embed_dim=96)

    # 前向传播
    out = patch_embed(x)

    print("输入大小:", x.shape)  # torch.Size([2, 3, 32, 32])
    print("输出大小:", out.shape)  # torch.Size([2, 8, 8, 96])
