import torch
import torch.nn as nn
from model_aux_tool import *


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
