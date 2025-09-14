import torch
import torch.nn as nn
from torch.nn import functional as F


def modal_Quantification(vis_modal_feat, inf_modal_feat, vis_scores, inf_scores, pids):
    # 根据分类情况对特征进行加权
    # Features (32, 2048)
    # cls_scores (32, n_class)
    # pids (32)
    B, C = vis_modal_feat

    vis_probs = torch.log_softmax(vis_scores, dim=1)
    inf_probs = torch.log_softmax(inf_scores, dim=1)

    vis_probs = vis_probs[torch.arange(B), pids]
    inf_probs = inf_probs[torch.arange(B), pids]

    modal_weights = torch.stack([vis_probs, inf_probs], dim=1)
    modal_weights = F.softmax(modal_weights, dim=1).clone().detach()
    vis_weights = modal_weights[:, 0].unsqueeze(1)
    inf_weights = modal_weights[:, 1].unsqueeze(1)

    quantified_features = vis_weights * vis_modal_feat + inf_weights * inf_modal_feat  # 注意调整维度
    return quantified_features
