import torch
import torch.nn.functional as F
from method import innovation, loss_function, module
from tools import MultiItemAverageMeter
from tqdm import tqdm


def train(base, loaders, config):
    base.set_train()
    meter = MultiItemAverageMeter()
    for epoch, data in enumerate(tqdm(loaders.train_loader)):
        imgs, pids, cids = data
        imgs, pids, cids = imgs.to(base.device), pids.to(base.device).long(), cids.to(base.device).long()
        if config.MODEL.MODULE == "Baseline":
            total_loss = 0.0

            # R: Resnet
            resnet_featuremaps, copy_resnet_featuremaps, resnet_internal_featuremaps = base.model(imgs)

            # ------------- Hard content branch -----------------------
            # Global
            global_features = base.model.module.global_pooling(resnet_featuremaps).squeeze()
            global_bn_features, global_cls_score = base.model.module.global_classifier(global_features)
            global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(global_cls_score, pids)
            total_loss += global_pid_loss

            # Part
            PART_NUM = config.MODEL.PART_NUM
            hard_part_chunk_features = torch.chunk(resnet_featuremaps, PART_NUM, dim=2)
            hard_part_pid_loss = 0.0
            for i in range(PART_NUM):
                hard_part_chunk_feature_item = hard_part_chunk_features[i]
                hard_part_pooling_features = base.model.module.hard_part_pooling[i](hard_part_chunk_feature_item).squeeze()
                hard_part_bn_features, hard_part_cls_score = base.model.module.hard_part_classifier[i](hard_part_pooling_features)
                hard_part_pid_loss += (1 / PART_NUM) * loss_function.CrossEntropyLabelSmooth().forward(hard_part_cls_score, pids)
            total_loss += hard_part_pid_loss

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "global_pid_loss": global_pid_loss.data,
                    "hard_part_pid_loss": hard_part_pid_loss.data,
                }
            )

        if config.MODEL.MODULE == "Lucky":
            total_loss = 0.0

            # R: Resnet
            resnet_featuremaps, copy_resnet_featuremaps, resnet_internal_featuremaps = base.model(imgs)

            # ------------- Hard content branch -----------------------
            # Global
            global_features = base.model.module.global_pooling(resnet_featuremaps).squeeze()
            global_bn_features, global_cls_score = base.model.module.global_classifier(global_features)
            global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(global_cls_score, pids)
            total_loss += global_pid_loss

            # Part
            PART_NUM = config.MODEL.PART_NUM
            hard_part_chunk_features = torch.chunk(resnet_featuremaps, PART_NUM, dim=2)
            hard_part_pid_loss = 0.0
            for i in range(PART_NUM):
                hard_part_chunk_feature_item = hard_part_chunk_features[i]
                hard_part_pooling_features = base.model.module.hard_part_pooling[i](hard_part_chunk_feature_item).squeeze()
                hard_part_bn_features, hard_part_cls_score = base.model.module.hard_part_classifier[i](hard_part_pooling_features)
                hard_part_pid_loss += (1 / PART_NUM) * loss_function.CrossEntropyLabelSmooth().forward(hard_part_cls_score, pids)
            total_loss += hard_part_pid_loss

            # ------------- Soft content branch -----------------------
            # Soft global
            soft_global_pooling_features = base.model.module.soft_global_pooling(copy_resnet_featuremaps).squeeze()
            soft_global_bn_features, soft_global_cls_score = base.model.module.soft_global_classifier(soft_global_pooling_features)
            soft_global_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_global_cls_score, pids)
            total_loss += soft_global_pid_loss

            # Soft attention
            soft_attention_featuremaps = base.model.module.soft_attention(resnet_internal_featuremaps + [copy_resnet_featuremaps])
            soft_attention_features = base.model.module.soft_attention_pooling(soft_attention_featuremaps).squeeze()
            soft_attention_bn_features, soft_attention_cls_score = base.model.module.soft_attention_classifier(soft_attention_features)
            soft_attention_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(soft_attention_cls_score, pids)
            total_loss += soft_attention_pid_loss

            # ------------- Multiview content branch  -----------------------
            # Position
            multiview_hard_CAM_featuremaps = base.model.module.multiview_hard_CAM(resnet_featuremaps, pids, base.model.module.global_classifier)
            multiview_soft_CAM_featuremaps = base.model.module.multiview_soft_CAM(copy_resnet_featuremaps, pids, base.model.module.soft_global_classifier)

            # Featuremaps fusion
            multiview_featuremaps = base.model.module.multiview_featuremap_fusion(multiview_hard_CAM_featuremaps, multiview_soft_CAM_featuremaps)

            # Quantification
            multiview_features = base.model.module.multiview_pooling(multiview_featuremaps).squeeze()
            _, multiview_hard_cls_score = base.model.module.global_classifier(multiview_features)
            _, multiview_soft_cls_score = base.model.module.soft_global_classifier(multiview_features)
            # multiview_quantification_cls_score = (multiview_hard_cls_score + multiview_soft_cls_score) / 2
            multiview_quantification_cls_score = multiview_hard_cls_score + multiview_soft_cls_score
            multiview_quantification_cls_score = F.softmax(multiview_quantification_cls_score, dim=1)
            multiview_features = base.model.module.multiview_feature_quantification(multiview_features, multiview_quantification_cls_score, pids)

            # View fusion
            multiview_features, multiview_pids = base.model.module.multiview_view_fusion(multiview_features, pids)

            # Classification
            multiview_bn_features, multiview_cls_score = base.model.module.multiview_classifier(multiview_features)
            multiview_pid_loss = loss_function.CrossEntropyLabelSmooth().forward(multiview_cls_score, multiview_pids)
            total_loss += multiview_pid_loss

            # ------------- ContrastLoss  -----------------------
            # contrast_loss = base.model.module.contrast_loss(global_bn_features, multiview_bn_features)
            contrast_loss = base.model.module.contrast_kl_loss(global_cls_score, multiview_cls_score, global_bn_features, multiview_bn_features)
            contrast_loss_2 = base.model.module.contrast_kl_loss(soft_global_cls_score, multiview_cls_score, soft_global_bn_features, multiview_bn_features)
            total_loss += contrast_loss + contrast_loss_2

            base.model_optimizer.zero_grad()
            total_loss.backward()
            base.model_optimizer.step()

            meter.update(
                {
                    "global_pid_loss": global_pid_loss.data,
                    "hard_part_pid_loss": hard_part_pid_loss.data,
                    "soft_global_pid_loss": soft_global_pid_loss.data,
                    "soft_attention_pid_loss": soft_attention_pid_loss.data,
                    "multiview_pid_loss": multiview_pid_loss.data,
                    "contrast_loss": contrast_loss.data,
                }
            )

    return meter
