import torch
import torch.nn as nn

from . import innovation, module
from .backbone import Backbone


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        BACKBONE_FEATURES_DIM = config.MODEL.BACKBONE_FEATURES_DIM
        VIEW_NUM = config.MODEL.VIEW_NUM
        PID_NUM = config.DATASET.PID_NUM
        PART_NUM = config.MODEL.PART_NUM
        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE
        MKL_WEIGHT = config.MODEL.MKL_WEIGHT

        # ------------- Backbone -----------------------
        self.backbone = Backbone(BACKBONE_TYPE)

        # ------------- Hard content branch -----------------------
        # Global
        self.global_pooling = module.GeneralizedMeanPoolingP()
        self.global_classifier = module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM)

        # Part
        self.hard_part_pooling = nn.ModuleList()
        self.hard_part_classifier = nn.ModuleList()
        for i in range(PART_NUM):
            self.hard_part_pooling.append(module.GeneralizedMeanPoolingP())
            self.hard_part_classifier.append(module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM))

        # ------------- soft content branch -----------------------
        # Soft global
        self.soft_global_pooling = module.GeneralizedMeanPoolingP()
        self.soft_global_classifier = module.Classifier(BACKBONE_FEATURES_DIM, config.DATASET.PID_NUM)

        self.soft_attention = innovation.attention_module.Feature_Pyramid_Network(in_cdim_list=[256, 512, 1024, 2048])
        self.soft_attention_pooling = module.GeneralizedMeanPoolingP()
        self.soft_attention_classifier = module.Classifier(2048, config.DATASET.PID_NUM)

        # ------------- Multiview content branch -----------------------
        # TODO: 1.innovation.multi_view.Featuremap_Fusion 修改为高级融合方式/concat融合/11conv
        # TODO: 2.multiview_pooling 池化部署在不同位置
        # TODO: 3.quantification权重之前加入Softmax
        # Postion
        self.multiview_hard_CAM = innovation.cam.CAM()
        self.multiview_soft_CAM = innovation.cam.CAM()

        # Featuremaps fusion
        self.multiview_featuremap_fusion = innovation.multi_view.Featuremap_Fusion(BACKBONE_FEATURES_DIM, BACKBONE_FEATURES_DIM)

        # Feature quantification
        self.multiview_feature_quantification = innovation.multi_view.Feature_Quantification(VIEW_NUM)

        # View fusion
        self.multiview_view_fusion = innovation.multi_view.View_Fusion(VIEW_NUM)

        # Classification
        self.multiview_pooling = module.GeneralizedMeanPoolingP()
        self.multiview_classifier = module.Classifier(BACKBONE_FEATURES_DIM, PID_NUM)

        # ------------- Contrast  Module -----------------------
        self.contrast_kl_loss = innovation.multi_view.MVDistillKL(VIEW_NUM, MKL_WEIGHT)

    def heatmap(self, x):
        resnet_feature_maps, copy_resnet_feature_maps, resnet_internal_feature_maps = self.backbone(x)
        return (resnet_feature_maps + copy_resnet_feature_maps) / 2

    def forward(self, x):
        if self.training:
            resnet_feature_maps, copy_resnet_feature_maps, resnet_internal_feature_maps = self.backbone(x)
            return resnet_feature_maps, copy_resnet_feature_maps, resnet_internal_feature_maps
        else:
            eval_features = []
            resnet_feature_maps, copy_resnet_feature_maps, resnet_internal_feature_maps = self.backbone(x)

            # Hard global
            global_features = self.global_pooling(resnet_feature_maps).squeeze()
            global_bn_features, global_cls_score = self.global_classifier(global_features)
            eval_features.append(global_bn_features)

            # Soft global
            soft_global_pooling_features = self.soft_global_pooling(copy_resnet_feature_maps).squeeze()
            soft_global_bn_features, soft_global_cls_score = self.soft_global_classifier(soft_global_pooling_features)
            eval_features.append(soft_global_bn_features)

            eval_features = torch.cat(eval_features, dim=1)
            return eval_features
