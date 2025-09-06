import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class CrossModalMamba(nn.Module):
    def __init__(self, in_cdim=2048, hidden_cdim=96):
        super(CrossModalMamba, self).__init__()
        # hidden_cdim 实际是ssm中运行维度的一半，在代码中未直接调用

        ssm_ratio = 2
        d_inner = hidden_cdim * ssm_ratio  # 实际ssm过程中的维度
        d_proj = d_inner * 2  # 投影x和z

        cross_modal_ratio = 2
        part_num = 6

        self.pool_size = (part_num, 1)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

        # Mamba block
        self.in_proj = nn.Conv2d(in_cdim * cross_modal_ratio, d_proj, 1, 1)
        self.ssm = drqssm(d_model=hidden_cdim)
        self.act = nn.SiLU()
        self.out_proj = nn.Conv2d(d_inner, in_cdim * cross_modal_ratio, 1, 1)

        self.cat_proj = nn.Conv2d(hidden_cdim * part_num, in_cdim, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_path = DropPath(0.5)

    def mamba_core(self, token):
        B, CR, H_token, W_token = token.shape  # [B, in_cdim*cross_modal_ratio, H_token, W_token]
        mamba_xz = self.in_proj(token)  # [B, d_inner*2, H_token, W_token]
        mamba_x, mamba_z = mamba_xz.chunk(2, dim=1)
        ssm_out = self.ssm(mamba_x.flatten(2))  # [B, d_inner, H_token, W_token]
        ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=H_token, w=W_token)
        # ssm_out = ssm_out * self.act(mamba_z)  # [B,  d_inner, H_token, W_token]
        # ssm_out = self.out_proj(ssm_out)  # [B,  in_cdim*cross_modal_ratio, H_token, W_token]
        # ssm_out = self.drop_path(ssm_out) + token
        return ssm_out  # [B, in_cdim*cross_modal_ratio, H_token, W_token]

    def forward(self, vis_feat, inf_feat):
        B, C, H, W = vis_feat.shape
        vis_feat = self.pool(vis_feat)  # [B, in_cdim, H_new, W_new]
        inf_feat = self.pool(inf_feat)

        modal_x = torch.stack([vis_feat, inf_feat], dim=2)  # [B, in_cdim, 2, H, W]
        modal_x = rearrange(modal_x, "B hidden s2 H W -> B (hidden s2) H W")  # # [B, in_cdim*2, H, W]

        ssm_out = self.mamba_core(modal_x)  # [B, in_cdim*cross_modal_ratio, H_token, W_token]
        vis_out = ssm_out[:, 0::2]  # [B, in_cdim, H_new, W_new]
        inf_out = ssm_out[:, 1::2]

        vis_cat_out = vis_out.reshape(B, -1, 1, 1)  # [B, in_cdim*part_num, 1, 1]
        inf_cat_out = inf_out.reshape(B, -1, 1, 1)

        vis_att = self.sigmoid(self.cat_proj(vis_cat_out))
        inf_att = self.sigmoid(self.cat_proj(inf_cat_out))
        return vis_att, inf_att


####################################################################################
# class CrossModalMamba(nn.Module):
#     def __init__(self, in_cdim=2048, hidden_cdim=96):
#         super(CrossModalMamba, self).__init__()

#         d_inner = hidden_cdim * 2
#         d_proj = d_inner * 2

#         self.pool_size = (6, 1)
#         self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

#         # Mamba block
#         self.in_proj = nn.Conv2d(in_cdim * 2, d_proj, 1, 1)
#         self.ssm = drqssm(d_model=hidden_cdim)
#         self.act = nn.SiLU()
#         # self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)

#         self.out_pool = nn.AdaptiveAvgPool2d(1)
#         self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, vis_feat, inf_feat):
#         B, C, H, W = vis_feat.shape
#         vis_feat = self.pool(vis_feat)  # [B, C, H, W] -> [B, C, size[0], size[1]]
#         inf_feat = self.pool(inf_feat)

#         modal_x = torch.stack([vis_feat, inf_feat], dim=2)  # [B, C, 2, H, W]
#         modal_x = rearrange(modal_x, "B hidden s2 H W -> B (hidden s2) H W")  # # [B, C*2, H, W]

#         # mamba block
#         mamba_xz = self.in_proj(modal_x)  # [B, C, H, W] -> [B, hidden_cdim*2*2, H, W]
#         mamba_x, mamba_z = mamba_xz.chunk(2, dim=1)  # [B, hidden_cdim*2, H, W],[B, hidden_cdim*2, H, W]
#         B, C_token, H_token, W_token = modal_x.shape
#         ssm_out = self.ssm(mamba_x.flatten(2))  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
#         ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=H_token, w=W_token)
#         ssm_out = ssm_out * self.act(mamba_z)

#         M_con = self.out_proj(self.out_pool(ssm_out))
#         # M_con = rearrange(M_con, "b (c d)-> b c d", c=C)
#         # M_con = M_con.view(B, C, C)
#         M_con = self.sigmoid(M_con)
#         return M_con


####################################################################################
# class MAMBA(nn.Module):
#     def __init__(self, in_cdim=2048, hidden_cdim=768):
#         super(MAMBA, self).__init__()

#         d_inner = hidden_cdim * 2
#         d_proj = d_inner * 2

#         self.pool_size = (6, 1)
#         self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

#         self.in_proj = nn.Conv2d(in_cdim, d_proj, 1, 1)
#         self.ssm = drqssm(d_model=hidden_cdim)
#         self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)
#         self.act = nn.SiLU()

#     def forward(self, feat):
#         B, C, H, W = feat.shape
#         skip = feat
#         feat = self.pool(feat)  # [B, C, H, W] -> [B, C, size[0], size[1]]

#         xz = self.in_proj(feat)
#         x, z = xz.chunk(2, dim=1)
#         b, c, h, w = x.shape
#         ssm_out = self.ssm(x.flatten(2))  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
#         ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=h, w=w)
#         out = ssm_out * self.act(z)
#         out = self.out_proj(out)
#         out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#         return out


####################################################################################


# class MAMBA(nn.Module):
#     def __init__(self, in_cdim=2048, hidden_cdim=768):
#         super(MAMBA, self).__init__()

#         d_inner = hidden_cdim * 2
#         d_proj = d_inner * 2

#         self.pool_size = (6, 1)
#         self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

#         self.in_proj = nn.Conv2d(in_cdim, d_proj, 1, 1)
#         self.ssm = drqssm(d_model=hidden_cdim)
#         self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)
#         self.act = nn.SiLU()

#     def forward(self, feat):
#         B, C, H, W = feat.shape
#         feat = self.pool(feat)  # [B, C, H, W] -> [B, C, size[0], size[1]]

#         xz = self.in_proj(feat)
#         x, z = xz.chunk(2, dim=1)
#         b, c, h, w = x.shape
#         ssm_out = self.ssm(x.flatten(2))  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
#         ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=h, w=w)
#         out = self.out_proj(ssm_out)
#         out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
#         return out


#############################################################
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """https://github.com/hu-xh/CPNet/blob/main/models/CPNet.py#L231
    DropPath (Stochastic Depth) 实现：
    - 类似 Dropout，但不是随机丢弃单个神经元，而是随机丢弃整个残差分支。
    - 训练时：每个样本的残差分支要么全部保留，要么整体置零。
    - 推理时：不做丢弃，保持完整。

    self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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

    # # 创建输入数据
    # inp_1 = torch.randn(2, 2048, 18, 9)
    # print("input.shape", inp_1.shape)

    # model = MAMBA(in_cdim=2048, hidden_cdim=256)
    # outputs = model(inp_1)
    # print(outputs.shape)

    # # # 创建输入数据
    inp_1 = torch.randn(2, 2048, 18, 9)
    inp_2 = torch.randn(2, 2048, 18, 9)
    print("input.shape", inp_1.shape)

    # # 初始化模型
    model = CrossModalMamba(in_cdim=2048, hidden_cdim=96)

    # # 前向传播
    outputs = model(inp_1, inp_2)

    # # # 打印输出形状
    for output in outputs:
        print(output.shape)
