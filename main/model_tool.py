import torch
import torch.nn as nn
from torch.nn import functional as F


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
class LayerNorm(nn.Module):
    """https://github.com/hu-xh/CPNet/blob/main/models/CPNet.py#L231
    channels_last：常用于 Transformer 或 MLP，输入形状 [B, H, W, C]
    channels_first：常用于 CNN，输入形状 [B, C, H, W]
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


#############################################################
class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        #
        p_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        loss = self.kl(p_s, p_t)  # * (self.T ** 2)
        return loss


#############################################################
class CIE(nn.Module):
    """
    https://github.com/RanwanWu/HINet/blob/main/Models/HINet.py
    Cross-modal hierarchical interaction network for RGB-D salient object detection
    """

    def __init__(self, in_channel):
        super(CIE, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, f1, f2):

        f = f1 + f2

        # middle
        f = self.relu(self.bn(self.conv(f)))
        f = self.maxpool(self.upsample(f))

        # up
        f1 = f + f1
        f1 = self.maxpool(self.upsample(f1))

        # down
        f2 = f + f2
        f2 = self.maxpool(self.upsample(f2))

        f = f1 + f2

        return f, f1, f2


#############################################################
class Residual(nn.Module):
    """
    残差模块类，用于实现残差连接。
    """

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Conv1d_BN_Relu(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(Conv1d_BN_Relu, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 1, 1, 0),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cbr(x)


class Fusion_By_C1C3BR(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(Conv1d_BN_Relu, self).__init__()
        self.fl = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 1, 1, 0, bias=False),
            nn.Conv2d(input_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.fl(x)


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


#############################################################
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
