import torch
import torch.nn as nn
from torch.nn import functional as F

from .. import module


class Featuremap_Fusion(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(Featuremap_Fusion, self).__init__()

        # self.fusion_layer = module.Residual(
        #     nn.Sequential(
        #         nn.Conv2d(input_dim, out_dim, 1, 1, 0),
        #         nn.ReLU(),
        #         nn.BatchNorm2d(out_dim),
        #         nn.Conv2d(out_dim, out_dim, 1, 1, 0),
        #     )
        # )
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(input_dim * 2, out_dim, 1, 1, 0),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.residual_layer = module.Residual(
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            )
        )
        self.fusion_layer.apply(module.weights_init_kaiming)

    def forward(self, features_1, features_2):
        fused = torch.cat([features_1, features_2], dim=1)
        fused = self.fusion_layer(fused)
        fused = self.residual_layer(fused)
        return fused


class View_Fusion(nn.Module):

    def __init__(self, view_num):
        super(View_Fusion, self).__init__()
        self.view_num = view_num

    def forward(self, features, pids):
        B, C = features.shape  # batch size, channels
        chunk_size = B // self.view_num

        # View-level fusion: reshape to [chunk_size, view_num, C]
        fused = features.view(chunk_size, self.view_num, C)
        fused = fused.mean(dim=1)  # [chunk_size, C]

        # PID-level fusion: assume same identity across views
        fused_pids = pids.view(chunk_size, self.view_num)[:, 0]

        return fused, fused_pids


class Feature_Quantification(nn.Module):

    def __init__(self, view_num):
        super(Feature_Quantification, self).__init__()
        self.view_num = view_num

    def forward(self, features, cls_scores, pids):
        size = features.size(0)
        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(size), pids]
        weights = torch.softmax(probs.view(-1, self.view_num), dim=1).view(-1).clone().detach()
        quantified_features = weights.unsqueeze(1) * features  #  注意：调整weight的维度
        return quantified_features


class ContrastLoss(nn.Module):
    def __init__(self, view_num):
        super(ContrastLoss, self).__init__()
        self.view_num = view_num

    def forward(self, features_1, features_2):
        # new_features_22 = torch.zeros(features_1.size()).to(features_1.device)
        # for i in range(int(features_1.size(0) / 4)):
        #     new_features_22[i * 4 : i * 4 + 4] = features_2[i]
        new_features_2 = torch.repeat_interleave(features_2, self.view_num, dim=0)  # [batch_size, c] -> [batch_size * view_num, c]
        input1_normed = F.normalize(features_1, p=2, dim=1)
        input2_normed = F.normalize(new_features_2, p=2, dim=1)
        loss = 0.05 * torch.norm((input1_normed - input2_normed), p=2)
        return loss


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, features_1_logits, features_2_logits):
        p_s = F.log_softmax(features_1_logits / self.T, dim=1)
        p_t = F.softmax(features_2_logits / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / features_1_logits.shape[0]
        return loss


class MVDistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, view_num, mkl_weight, T=4):
        super(MVDistillKL, self).__init__()
        self.T = T
        self.view_num = view_num
        self.mkl_weight = mkl_weight

        self.distillKL = DistillKL(T)

    def forward(self, features_1_logits, features_2_logits, features_1, features_2):

        loss = 0.0
        new_features_2_logits = torch.repeat_interleave(features_2_logits, self.view_num, dim=0)  # [batch_size, c] -> [batch_size * view_num]
        kl_loss = self.mkl_weight * (self.distillKL(features_1_logits, new_features_2_logits) + self.distillKL(new_features_2_logits, features_1_logits))
        loss += kl_loss
        return loss
