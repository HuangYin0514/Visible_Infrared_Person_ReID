import copy

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .mamba import MAMBA
from .model_aux_tool import Gate_Fusion


class Interaction(nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()

        self.mamba = MAMBA(in_cdim=2048, hidden_cdim=256)

        self.pooling = nn.AdaptiveAvgPool2d((6, 3))
        # self.un_pooling = nn.Upsample(scale_factor=3, mode="bilinear", align_corners=True)
        # F.interpolate(f_i, size=size_in, mode="nearest")

        self.local_rgb_conv = nn.Sequential(
            nn.Conv2d(2048 * 2, 2048, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.local_inf_conv = copy.deepcopy(self.local_rgb_conv)

        self.vis_add_inf = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.inf_add_vis = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_map):
        B, C, H, W = feat_map.shape

        vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)

        cat_vis_inf_feat_map = torch.cat([vis_feat_map, inf_feat_map], dim=1)
        vis_local_feat_map = self.local_rgb_conv(cat_vis_inf_feat_map) * vis_feat_map
        inf_local_feat_map = self.local_inf_conv(cat_vis_inf_feat_map) * inf_feat_map

        # Split Horizon Strategy & Patch mixed reordering
        vis_part_feat_map = self.pooling(vis_feat_map)  # [B, C, 6, 3]
        vis_part_rearrange_feat_map = rearrange(vis_part_feat_map, "b c h w -> b c (h w)")
        inf_part_feat_map = self.pooling(inf_feat_map)
        inf_part_rearrange_feat_map = rearrange(inf_part_feat_map, "b c h w -> b c (h w)")

        mixed_feat = []
        for i in range(6 * 3):
            mixed_feat.append(vis_part_rearrange_feat_map[:, :, i])
            mixed_feat.append(inf_part_rearrange_feat_map[:, :, i])
        mixed_feat = torch.stack(mixed_feat, dim=2).unsqueeze(3)  # [B//2, C, self.part_num * 2, 1]

        # Mamba
        mamba_feat = self.mamba(mixed_feat)  # [B//2, C, num_patch, 1]
        vis_mamba_feat = mamba_feat[:, :, 0::2]  # [B//2, C, num_patch, 1]
        vis_mamba_feat = rearrange(vis_mamba_feat.squeeze(), "b c (h w)-> b c h w", h=6, w=3)  # [B//2, C, 6, 3]
        vis_mamba_feat = F.interpolate(vis_mamba_feat, size=(H, W), mode="nearest")  # [B//2, C, H, W]
        inf_mamba_feat = mamba_feat[:, :, 1::2]
        inf_mamba_feat = rearrange(inf_mamba_feat.squeeze(), "b c (h w)-> b c h w", h=6, w=3)
        inf_mamba_feat = F.interpolate(inf_mamba_feat, size=(H, W), mode="nearest")

        # Fusion
        vis_feat_map = self.vis_add_inf(inf_mamba_feat + inf_local_feat_map) + vis_feat_map
        inf_feat_map = self.inf_add_vis(vis_mamba_feat + vis_local_feat_map) + inf_feat_map
        feat_map = torch.cat([vis_feat_map, inf_feat_map], dim=0)
        return feat_map


# class Interaction(nn.Module):

#     def __init__(self):
#         super(Interaction, self).__init__()

#         self.vis_add_inf = nn.Sequential(
#             nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2048),
#             nn.ReLU(inplace=True),
#         )
#         self.inf_add_vis = nn.Sequential(
#             nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2048),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, feat_map):
#         vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)
#         vis_feat_map = self.vis_add_inf(inf_feat_map) + vis_feat_map
#         inf_feat_map = self.inf_add_vis(vis_feat_map) + inf_feat_map
#         feat_map = torch.cat([vis_feat_map, inf_feat_map], dim=0)
#         return feat_map


class Calibration(nn.Module):

    def __init__(self):
        super(Calibration, self).__init__()

        c_dim = 2048
        self.vis_gate_calibration = Gate_Fusion(c_dim)
        self.inf_gate_calibration = Gate_Fusion(c_dim)

    def forward(self, feat_map, res_feat_map):
        vis_feat, inf_feat = torch.chunk(feat_map, 2, dim=0)
        res_vis_feat, res_inf_feat = torch.chunk(res_feat_map, 2, dim=0)
        vis_feat = self.vis_gate_calibration(vis_feat, res_vis_feat)
        inf_feat = self.inf_gate_calibration(inf_feat, res_inf_feat)

        calibration_feat_map = torch.cat([vis_feat, inf_feat], dim=0)
        return calibration_feat_map


class Propagation(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T=4):
        super(Propagation, self).__init__()
        self.T = T
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        p_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        loss = self.kl(p_s, p_t)  # * (self.T ** 2)
        return loss
