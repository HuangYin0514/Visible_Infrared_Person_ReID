import torch

# def feat_map_integrating(feat_map, labels):
#     # feat_map: [B, C, H, W]
#     # labels: [B, H, W]
#     B, C, H, W = feat_map.size()
#     chunk_size = int(B // 4)  # 两个模态，每个模态取两张图片。 64 -> 16

#     vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)  # 切分模态
#     vis_labels, inf_labels = torch.chunk(labels, 2, dim=0)

#     integrating_feat_map = torch.zeros([chunk_size, C, H, W]).to(feat_map.device)  # 融合
#     integrating_labels = torch.zeros([chunk_size]).to(labels.device).long()
#     chunk_vis_feat_map = torch.chunk(vis_feat_map, chunk_size, dim=0)
#     chunk_inf_feat_map = torch.chunk(inf_feat_map, chunk_size, dim=0)
#     chunk_vis_labels = torch.chunk(vis_labels, chunk_size, dim=0)
#     for i in range(chunk_size):
#         integrating_feat_map[i, :, :, :] = chunk_vis_feat_map[i][0] + chunk_vis_feat_map[i][1] + chunk_inf_feat_map[i][0] + chunk_inf_feat_map[i][1]
#         integrating_labels[i] = chunk_vis_labels[i][0]
#     return integrating_feat_map, integrating_labels


def feat_map_integrating(feat_map, labels):
    # feat_map: [B, C, H, W]
    # labels: [B, H, W]
    B, C, H, W = feat_map.size()

    vis_feat_map, inf_feat_map = torch.chunk(feat_map, 2, dim=0)  # 切分模态
    vis_labels, inf_labels = torch.chunk(labels, 2, dim=0)

    integrating_feat_map = ((vis_feat_map + inf_feat_map) / 2).clone().detach()
    integrating_labels = vis_labels
    return integrating_feat_map, integrating_labels
