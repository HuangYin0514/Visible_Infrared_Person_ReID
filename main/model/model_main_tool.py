import torch
import torch.nn as nn
from torch.nn import functional as F

from .mamba import MAMBA
from .model_aux_tool import Gate_Fusion


class Interaction(nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()

        self.part_num = 6

        self.pool = nn.AdaptiveAvgPool2d((self.part_num, 1))

        self.mamba = MAMBA(in_cdim=2048, hidden_cdim=96)

        self.mamba_att = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        # self.sigmoid = nn.Sigmoid()

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

        # Split Horizon Strategy
        vis_part_feat, inf_part_feat = self.pool(vis_feat_map), self.pool(inf_feat_map)

        # Patch mixed reordering
        fused_feat = torch.ones([int(B // 2), C, self.part_num * 2, 1]).to(feat_map.device)
        for i in range(self.part_num):
            fused_feat[:, :, 2 * i] = vis_part_feat[:, :, i, :]
            fused_feat[:, :, 2 * i + 1] = inf_part_feat[:, :, i, :]

        # Mamba
        fused_feat_weight = self.mamba_att(fused_feat)

        # Enhance
        vis_feat_weight = fused_feat_weight[:, :, 0::2]
        inf_feat_weight = fused_feat_weight[:, :, 1::2]
        vis_feat_weight = torch.repeat_interleave(vis_feat_weight, repeats=3, dim=2)
        inf_feat_weight = torch.repeat_interleave(inf_feat_weight, repeats=3, dim=2)
        complementary_vis_feat_map = vis_feat_weight * vis_feat_map
        complementary_inf_feat_map = inf_feat_weight * inf_feat_map

        # Fusion
        vis_feat_map = self.vis_add_inf(complementary_inf_feat_map + inf_feat_map) + vis_feat_map
        inf_feat_map = self.inf_add_vis(complementary_vis_feat_map + vis_feat_map) + inf_feat_map
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
