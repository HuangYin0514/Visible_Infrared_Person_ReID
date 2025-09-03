import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class CrossModalMambaModule(nn.Module):
    def __init__(self, in_cdim=2048, hidden_cdim=768):
        super(CrossModalMambaModule, self).__init__()

        d_inner = hidden_cdim * 2
        d_proj = d_inner * 2

        self.pool_size = (3, 3)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

        self.in_proj = nn.Conv2d(in_cdim, d_proj, 1, 1)
        self.ssm = drqssm(d_model=hidden_cdim * 2)
        self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)
        self.act = nn.SiLU()

    def forward(self, vis_feat, inf_feat):
        B, C, H, W = vis_feat.shape
        vis_feat = self.pool(vis_feat)  # [B, C, H, W] -> [B, C, size[0], size[1]]
        inf_feat = self.pool(inf_feat)

        # Vis branch
        vis_xz = self.in_proj(vis_feat)  # [B, C, H, W] -> [B, hidden_cdim*2*2, H, W]
        vis_x, vis_z = vis_xz.chunk(2, dim=1)  # [B, hidden_cdim*2, H, W],[B, hidden_cdim*2, H, W]

        # Inf branch
        inf_xz = self.in_proj(inf_feat)
        inf_x, inf_z = inf_xz.chunk(2, dim=1)

        # Cross modal branch
        modal_x = torch.stack([vis_x, inf_x], dim=2)  # [B, hidden_cdim*2*2, 2, H, W]
        modal_x = rearrange(modal_x, "B hidden s2 H W -> B (hidden s2) H W")

        B, C_token, H_token, W_token = modal_x.shape
        ssm_out = self.ssm(modal_x.flatten(2))  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=H_token, w=W_token)

        vis_ssm_out = ssm_out[:, 0::2, :, :]  # [B, hidden_cdim*2, H, W]
        inf_ssm_out = ssm_out[:, 1::2, :, :]  # [B, hidden_cdim*2, H, W]

        # Merge
        vis_out = vis_ssm_out * self.act(vis_z)
        inf_out = inf_ssm_out * self.act(inf_z)

        vis_out = self.out_proj(vis_out)
        inf_out = self.out_proj(inf_out)

        vis_feat_hat = F.interpolate(vis_out, size=(H, W), mode="bilinear", align_corners=False)
        inf_feat_hat = F.interpolate(inf_out, size=(H, W), mode="bilinear", align_corners=False)

        return vis_feat_hat, inf_feat_hat


class CrossModalMambaModule_20250903(nn.Module):
    def __init__(self, in_cdim=2048, hidden_cdim=768):
        super(CrossModalMambaModule, self).__init__()

        d_inner = hidden_cdim * 2
        d_proj = d_inner * 2

        self.pool_size = (3, 3)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

        self.in_proj = nn.Conv2d(in_cdim, d_proj, 1, 1)
        self.ssm = drqssm(d_model=hidden_cdim)
        self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)
        self.act = nn.SiLU()

    def forward(self, vis_feat):
        B, C, H, W = vis_feat.shape
        vis_feat = self.pool(vis_feat)  # [B, C, H, W] -> [B, C, size[0], size[1]]

        xz = self.in_proj(vis_feat)
        x, z = xz.chunk(2, dim=1)
        b3, c3, h3, w3 = x.shape
        ssm_out = self.ssm(x.flatten(2))  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=h3, w=w3)
        out = ssm_out * self.act(z)
        out = self.out_proj(out)

        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


def cross_selective_scan(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    B, D, L = x.shape
    D, N = A_logs.shape
    D, R = dt_projs_weight.shape

    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)

    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=-1)
    dts = dt_projs_weight @ dts.t()
    dts = F.softplus(dts)  # ADD for mine
    dts = rearrange(dts, "d (b l) -> b l d", l=L)
    Bs = rearrange(Bs, "(b l) dstate -> b dstate l", l=L).contiguous()
    Cs = rearrange(Cs, "(b l) dstate -> b dstate l", l=L).contiguous()
    As = -torch.exp(A_logs.to(torch.float))  # (D, d_state)
    Ds = Ds.to(torch.float)  # (D)

    # Step 1: Discretize continuous parameters (A, B)
    dt_A = torch.exp(einsum(dts, As, "B L D, D dstate -> B L D dstate"))
    dt_B_x = einsum(dts, Bs, x, "B L D, B dstate L, B D L -> B L D dstate")

    # Step 2: Perform selective scan
    u = torch.zeros((B, D, N), device=dt_A.device)
    ys = []
    for i in range(L):
        u = dt_A[:, i] * u + dt_B_x[:, i]
        y = einsum(u, Cs[:, :, i], "B D dstate, B dstate -> B D")
        ys.append(y)
    y = torch.stack(ys, dim=-1)  # [B, D, L]
    y = y + einsum(x, Ds, "B D L, D -> B D L")  # [B, D, L]

    # Normalize output
    y = rearrange(y, "b d l -> b l d")
    y = out_norm(y)
    return y.to(x.dtype)


class drqssm(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        d_conv=4,
        bias=False,
        device=None,
        dtype=None,
        ssm_ratio=2.0,
        dt_rank="auto",
        dropout=0.0,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        conv_bias=True,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        # print('newmamba')
        factory_kwargs = {"device": device, "dtype": dtype}
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        # in proj =======================================

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # x proj ============================
        self.x_proj = nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)

        # dt proj ============================
        self.dt_projs = self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)

        # softmax | sigmoid | dwconv | norm ===========================

        self.out_norm = nn.LayerNorm(d_inner)
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, merge=True)  # (K * D)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        # inv_dt = dt + torch.log(-torch.expm1(-dt))
        # with torch.no_grad():
        #     dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan):
        return cross_selective_scan(
            x,
            self.x_proj.weight,
            self.dt_projs.weight,
            self.A_logs,
            self.Ds,
            out_norm=self.out_norm,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        y = self.forward_core(x)
        return y


if __name__ == "__main__":

    # 创建输入数据
    inp_1 = torch.randn(2, 2048, 18, 9)
    inp_2 = torch.randn(2, 2048, 18, 9)
    print("input.shape", inp_1.shape)

    # # 初始化模型
    model = CrossModalMambaModule(in_cdim=2048, hidden_cdim=96)

    # # 前向传播
    outputs = model(inp_1, inp_2)

    # # # 打印输出形状
    for output in outputs:
        print(output.shape)
    # print(outputs.shape)
