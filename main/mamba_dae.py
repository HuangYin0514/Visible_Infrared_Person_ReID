import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class VisionMambaModule(nn.Module):
    def __init__(self, in_cdim=2048, hidden_cdim=768):
        super(VisionMambaModule, self).__init__()

        self.pe = Patch_Embedding(in_cdim=in_cdim, out_cdim=hidden_cdim, size=(3, 3))
        self.mamba = Mamba(in_cdim=hidden_cdim, out_cdim=in_cdim)
        self.ie = Inverse_Patch_Embedding(size=(3, 3))

    def forward(self, x):
        B, C, H, W = x.shape
        token_x = self.pe(x)  # [bs, H*W, hidden_cdim]
        output = self.mamba(token_x)  # [bs, H*W, in_cdim]
        output = self.ie(output, H, W)  # [bs, in_cdim, H, W]
        return output


class Inverse_Patch_Embedding(nn.Module):

    def __init__(self, size=(3, 3)):
        super(Inverse_Patch_Embedding, self).__init__()

        self.size = size

    def forward(self, x, h, w):
        out = rearrange(x, "b (h w) c -> b c h w", h=self.size[0], w=self.size[1])
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


class Patch_Embedding(nn.Module):

    def __init__(self, in_cdim=3, out_cdim=768, size=(3, 3)):
        super(Patch_Embedding, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(size)
        self.proj = nn.Linear(in_cdim, out_cdim)

    def forward(self, x):
        x = self.pool(x)  # [B, C, H, W] -> [B, C, size[0], size[1]]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        return x


class Mamba(nn.Module):
    def __init__(self, in_cdim=768, out_cdim=2048, expand_ratio=2):
        super(Mamba, self).__init__()

        self.inner_dim = in_cdim * expand_ratio
        kernel_size = 3

        self.layer_norm = nn.LayerNorm(in_cdim)

        self.in_proj = nn.Linear(in_cdim, self.inner_dim * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            bias=True,
            kernel_size=kernel_size,
            groups=self.inner_dim,
            padding=(kernel_size - 1) // 2,
        )

        self.ssm = SSM(
            inner_dim=self.inner_dim,
            state_dim=16,
            dt_rank=math.ceil(in_cdim / 16),
        )

        self.out_proj = nn.Linear(self.inner_dim, out_cdim, bias=False)

    def forward(self, x):
        B, L, D = x.shape

        # Step 0: Layer Normalization
        x = self.layer_norm(x)  # [B, L, D]

        # Step 1: Project x to xz
        xz = self.in_proj(x)  # [B, L, D] -> [B, L, 2*D]
        (x, z) = xz.split(split_size=[self.inner_dim, self.inner_dim], dim=-1)  # [B, L, D] and [B, L, D]

        # Step 2: Conv1d
        x = rearrange(x, "B L D -> B D L")
        x = self.conv1d(x)
        x = rearrange(x, "B D L -> B L D")
        x = F.silu(x)  # [B, L, D]

        # Step 3: SSM
        y = self.ssm(x)  # [B, L, D]

        # Step 4: residual connection
        y = y * F.silu(z)

        # Step 5: Project y to out
        out = self.out_proj(y)  # [B, L, D] -> [B, L, out_cdim]
        return out


class SSM(nn.Module):

    def __init__(self, inner_dim=768 * 2, state_dim=16, expand_ratio=2, dt_rank=None):
        super(SSM, self).__init__()

        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        # A
        A = repeat(torch.arange(1, self.state_dim + 1), "state_dim -> d state_dim", d=self.inner_dim)  # Shape (inner_dim, state_dim); Ex. [[1, 2, ... , 16], ... ]
        self.A_log = nn.Parameter(torch.log(A))

        # x is projected to delta_ori, B, C
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + self.state_dim * 2, bias=False)

        # delta_ori is projected to delta
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # D
        self.D = nn.Parameter(torch.ones(self.inner_dim))  # Ex. [[1, 1, ... , 1], ... ]

    def forward(self, x):
        B, L, D = x.shape

        # Step 0: Get A and D
        A_parameter = -torch.exp(self.A_log.float())  # Shape (D, state_dim); Ex. [-[1, 2, ... , 16], ... ]
        D_parameter = self.D.float()

        # Step 1: Project x to delta_B_C
        delta_B_C = self.x_proj(x)  # [B, L, D + state_dim * 2]
        (delta, B_parameter, C_parameter) = delta_B_C.split(
            split_size=[self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )  # delta: (B, L, dt_rank). B, C: (B, L, state_dim)
        delta_parameter = F.softplus(self.dt_proj(delta))  # (B, L, D)

        y = self.selective_scan(x, delta_parameter, A_parameter, B_parameter, C_parameter, D_parameter)  # [B, L, D]

        return y

    def selective_scan(self, u, delta_parameter, A_parameter, B_parameter, C_parameter, D_parameter):
        B, L, D = u.shape

        # Step 1: Discretize continuous parameters (A, B)
        delta_A = torch.exp(einsum(delta_parameter, A_parameter, "B L D, D state_dim -> B L D state_dim"))
        delta_B_u = einsum(delta_parameter, B_parameter, u, "B L D, B L state_dim, B L D -> B L D state_dim")

        x = torch.zeros((B, D, self.state_dim), device=delta_A.device)
        ys = []
        for i in range(L):
            x = delta_A[:, i] * x + delta_B_u[:, i]
            y = einsum(x, C_parameter[:, i, :], "B D state_dim, B state_dim -> B D")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # [B, L, D]

        y = y + u * D_parameter  # [B, L, D]

        return y


################################################

if __name__ == "__main__":

    # 创建输入数据
    inputs = torch.randn(2, 2048, 18, 9)
    print("input.shape", inputs.shape)

    # # 初始化模型
    model = VisionMambaModule(in_cdim=2048, hidden_cdim=96)

    # # 前向传播
    outputs = model(inputs)

    # # # 打印输出形状
    print(outputs.shape)
