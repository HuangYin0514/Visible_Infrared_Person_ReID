import torch
import torch.nn as nn
from torch.nn import functional as F


def modal_Quantification(vis_scores, inf_scores, pids):
    # cls_scores (32, n_class)
    # pids (32, )
    B, n_class = vis_scores.shape

    vis_probs = torch.log_softmax(vis_scores, dim=1)
    inf_probs = torch.log_softmax(inf_scores, dim=1)

    vis_probs = vis_probs[torch.arange(B), pids]  # 取pids对应概率
    inf_probs = inf_probs[torch.arange(B), pids]

    modal_weights = torch.stack([vis_probs, inf_probs], dim=1)  # 计算权重
    modal_weights = F.softmax(modal_weights, dim=1).clone().detach()
    vis_weights = modal_weights[:, 0].unsqueeze(1)
    inf_weights = modal_weights[:, 1].unsqueeze(1)

    return vis_weights, inf_weights
