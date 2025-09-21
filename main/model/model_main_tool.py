import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mamba import MAMBA
from .model_aux_tool import Gate_Fusion


class Interaction(nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()

        self.part_num = 6

        self.vis_part_pool = nn.ModuleList()
        self.vis_part_att = nn.ModuleList()
        for i in range(self.part_num):
            self.vis_part_pool.append(nn.AdaptiveMaxPool2d((1, 1)))
            self.vis_part_att.append(
                nn.Sequential(
                    nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
                    nn.Sigmoid(),
                )
            )

        self.inf_part_pool = copy.deepcopy(self.vis_part_pool)
        self.inf_part_att = copy.deepcopy(self.vis_part_att)

        self.mamba = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        # self.mamba = MAMBA(in_cdim=2048, hidden_cdim=96)

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

        # Split Horizon Strategy & Patch mixed reordering
        vis_part_feat_map = torch.chunk(vis_feat_map, self.part_num, dim=2)  # list(self.part_num)
        inf_part_feat_map = torch.chunk(inf_feat_map, self.part_num, dim=2)

        mixed_feat = []
        for i in range(self.part_num):
            mixed_feat.append(self.vis_part_pool[i](vis_part_feat_map[i]).squeeze(-1))
            mixed_feat.append(self.inf_part_pool[i](inf_part_feat_map[i]).squeeze(-1))
        mixed_feat = torch.stack(mixed_feat, dim=2)  # [B//2, C, self.part_num * 2, 1]

        # Mamba
        mamba_feat = self.mamba(mixed_feat)  # [B//2, C, self.part_num * 2, 1]
        vis_mamba_feat = mamba_feat[:, :, 0::2]  # [B//2, C, self.part_num, 1]
        inf_mamba_feat = mamba_feat[:, :, 1::2]

        # Weighted Fusion
        vis_weighted_feat_map = []  # [B//2, C, H, W]
        inf_weighted_feat_map = []
        for i in range(self.part_num):
            vis_weight_i = self.vis_part_att[i](vis_mamba_feat[:, :, i].unsqueeze(-1))
            inf_weight_i = self.inf_part_att[i](inf_mamba_feat[:, :, i].unsqueeze(-1))
            vis_weighted_feat_map.append(vis_weight_i * vis_part_feat_map[i])
            inf_weighted_feat_map.append(inf_weight_i * inf_part_feat_map[i])
        vis_weighted_feat_map = torch.cat(vis_weighted_feat_map, dim=2)
        inf_weighted_feat_map = torch.cat(inf_weighted_feat_map, dim=2)

        # Fusion
        vis_feat_map = self.vis_add_inf(inf_weighted_feat_map) + vis_feat_map
        inf_feat_map = self.inf_add_vis(vis_weighted_feat_map) + inf_feat_map
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
