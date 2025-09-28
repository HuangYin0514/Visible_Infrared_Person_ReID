import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


#############################################################
class Mamba(nn.Module):
    def __init__(self, in_cdim=2048, d_model=96):
        super(Mamba, self).__init__()

        # Mamba
        self.norm_1 = nn.LayerNorm(in_cdim)
        self.ss1d = SS1D(in_cdim=in_cdim, d_model=d_model)

        # FFN
        self.norm_2 = nn.LayerNorm(in_cdim)
        self.ffn = nn.Sequential(
            nn.Linear(in_cdim, in_cdim, bias=False),
            nn.LayerNorm(in_cdim),
            nn.ReLU(),
        )

    def forward(self, feat_patch):
        B, L, C = feat_patch.shape
        feat_patch = self.ss1d(self.norm_1(feat_patch)) + feat_patch  # [B, n_patch, C]
        # feat_patch = self.ffn(self.norm_2(feat_patch)) + feat_patch  # [B, n_patch, C]
        return feat_patch


class SS1D(nn.Module):
    def __init__(self, in_cdim=2048, d_model=96):
        super(SS1D, self).__init__()

        ssm_ratio = 1.0
        d_inner = int(ssm_ratio * d_model)
        kernel_size = 3

        self.in_proj = nn.Linear(in_cdim, d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=kernel_size,
            groups=d_inner,
            padding=(kernel_size - 1) // 2,
        )
        self.ssm = SSM(d_model=d_inner)
        self.out_proj = nn.Linear(d_inner, in_cdim, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        x = self.in_proj(x)  # [B, L, D]
        x = rearrange(x, "B L D -> B D L")
        x = self.conv1d(x)
        x = rearrange(x, "B D L -> B L D")
        x = F.silu(x)  # [B, L, D]
        y = self.ssm(x)  # [B, L, D]
        out = self.out_proj(y)  # [B, L, in_cdim]
        return out


class SSM(nn.Module):

    def __init__(
        self,
        d_model=96,
        d_state=16,
    ):
        super(SSM, self).__init__()

        ssm_ratio = 1.0
        self.d_inner = int(ssm_ratio * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        # A
        A = repeat(torch.arange(1, self.d_state + 1), "d_state -> d_inner d_state", d_inner=self.d_inner)  # Shape (d_inner, state_dim); Ex. [[1, 2, ... , 16], ... ]
        self.A_log = nn.Parameter(torch.log(A))

        # x is projected to delta_ori, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # delta_ori is projected to delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # D
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Ex. [[1, 1, ... , 1], ... ]

    def forward(self, x):
        B, L, D = x.shape

        # Step 0: Get A and D
        A_parameter = -torch.exp(self.A_log.float())  # Shape (D, d_state); Ex. [-[1, 2, ... , 16], ... ]
        D_parameter = self.D.float()

        # Step 1: Project x to delta_B_C
        delta_B_C = self.x_proj(x)  # [B, L, D + d_state * 2]
        (delta, B_parameter, C_parameter) = delta_B_C.split(
            split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1
        )  # delta: (B, L, dt_rank). B, C: (B, L, d_state)
        delta_parameter = F.softplus(self.dt_proj(delta)) * 0.001  # (B, L, D)

        y = self.selective_scan(x, delta_parameter, A_parameter, B_parameter, C_parameter, D_parameter)  # [B, L, D]

        return y

    def selective_scan(self, u, delta_parameter, A_parameter, B_parameter, C_parameter, D_parameter):
        B, L, D = u.shape

        # Step 1: Discretize continuous parameters (A, B)
        delta_A = torch.exp(einsum(delta_parameter, A_parameter, "B L D, D state_dim -> B L D state_dim"))
        delta_B_u = einsum(delta_parameter, B_parameter, u, "B L D, B L state_dim, B L D -> B L D state_dim")

        x = torch.zeros((B, D, self.d_state), device=delta_A.device)
        ys = []
        for i in range(L):
            x = delta_A[:, i] * x + delta_B_u[:, i]
            y = einsum(x, C_parameter[:, i, :], "B D state_dim, B state_dim -> B D")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # [B, L, D]

        y = y + u * D_parameter  # [B, L, D]

        return y


if __name__ == "__main__":

    inp_1 = torch.randn(2, 6, 2048)
    print("input.shape", inp_1.shape)
    model = Mamba(in_cdim=2048, d_model=96)
    outputs = model(inp_1)
    print(outputs.shape)
