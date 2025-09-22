import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


class MAMBA(nn.Module):
    def __init__(self, in_cdim=2048, hidden_cdim=96):
        super(MAMBA, self).__init__()

        d_inner = hidden_cdim * 2

        self.norm = LayerNorm(in_cdim, "with_bias")
        self.in_proj = nn.Conv2d(in_cdim, d_inner * 2, 1, 1)
        self.ssm = SSM(d_model=hidden_cdim)
        self.out_proj = nn.Conv2d(d_inner, in_cdim, 1, 1)
        self.act = nn.SiLU()

        self.drop_path = DropPath(0.5)

        self.mamba_out = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape

        feat_map = self.norm(feat_map)
        xz = self.in_proj(feat_map)
        x, z = xz.chunk(2, dim=1)
        b, c, h, w = x.shape
        ssm_forward_out = self.ssm(x.flatten(2))  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        ssm_backward_out = self.ssm(x.flatten(2).flip([-1]))
        ssm_out = ssm_forward_out + ssm_backward_out.flip([1])
        ssm_out = rearrange(ssm_out, "b (h w) c -> b c h w", h=h, w=w)
        ssm_out = ssm_out * self.act(z)  # [B, C, H, W]
        out = self.out_proj(ssm_out)
        out = self.mamba_out(self.drop_path(out) + feat_map)
        return out


#############################################################
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


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


#############################################################
class SSM(nn.Module):
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
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        conv_bias=True,
        dt_init_floor=1e-4,
    ):
        super().__init__()
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

    def cross_selective_scan(
        self,
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
    ):
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
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor):
        y = self.cross_selective_scan(
            x,
            self.x_proj.weight,
            self.dt_projs.weight,
            self.A_logs,
            self.Ds,
        )
        return y


if __name__ == "__main__":
    inp_1 = torch.randn(2, 96 * 2, 6)
    print("input.shape", inp_1.shape)
    model = SSM(d_model=96)
    outputs = model(inp_1)
    print(outputs.shape)

    inp_1 = torch.randn(2, 2048, 3, 2)
    print("input.shape", inp_1.shape)
    model = MAMBA(in_cdim=2048, hidden_cdim=96)
    outputs = model(inp_1)
    print(outputs.shape)
