import torch
import torch.nn as nn
from torch.nn import functional as F


class Interaction(nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()

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
        vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)
        vis_feat_map = self.vis_add_inf(inf_feat_map) + vis_feat_map
        inf_feat_map = self.inf_add_vis(vis_feat_map) + inf_feat_map
        feat_map = torch.cat([vis_feat_map, inf_feat_map], dim=0)
        return feat_map


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
