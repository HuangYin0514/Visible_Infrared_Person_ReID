import torch
import torch.nn as nn
from torch.nn import functional as F


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
