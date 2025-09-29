import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat

# ===== 测试 =====
if __name__ == "__main__":
    # --- Method 1 ---
    # x1 = torch.ones([12, 2048, 6])
    # x2 = torch.ones([12, 2048, 6]) * 2

    # x_cat = torch.stack([x1, x2], dim=2).view([12, 2048 * 2, 6])

    # print(x_cat)

    # x_cat = x_cat.view([12, 2048, 2, 6])
    # x1_hat = x_cat[:, :, 0, :]
    # x2_hat = x_cat[:, :, 1, :]

    # print(x1_hat)
    # print(x2_hat)

    # --- Method 2 ---
    x1 = torch.ones([12, 2048, 6])
    x2 = torch.ones([12, 2048, 6]) * 5

    x_cat = torch.stack([x1, x2], dim=2)
    x_cat = rearrange(x_cat, "B C D N -> B (C D) N")  # [B, 2C, H, W]

    print(x_cat)
    x1_hat, x2_hat = x_cat[:, 0::2, :], x_cat[:, 1::2, :]

    print(x1_hat)
    print(x2_hat)
