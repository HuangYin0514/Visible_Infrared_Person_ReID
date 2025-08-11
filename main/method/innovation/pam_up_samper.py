import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import module


class PamUpSamper(nn.Module):
    def __init__(self, in_dim, output_dim, bias=False, scale=2):
        super(PamUpSamper, self).__init__()

        self.scale = scale
        self.out_dim = output_dim

        self.upsapmle = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_dim, output_dim, 1, bias=bias)
        self.conv.apply(module.weights_init_kaiming)

    def forward(self, x):
        x = self.upsapmle(x)
        x = self.conv(x)
        return F.relu(x, inplace=True)
