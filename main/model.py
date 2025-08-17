import copy

import torch
import torch.nn as nn
from gem_pool import GeneralizedMeanPoolingP
from model_tool import *
from resnet import resnet50
from resnet_ibn_a import resnet50_ibn_a


class ReIDNet(nn.Module):

    def __init__(self, config, n_class):
        super(ReIDNet, self).__init__()
        self.config = config

        BACKBONE_FEATURES_DIM = config.MODEL.BACKBONE_FEATURES_DIM
        BACKBONE_TYPE = config.MODEL.BACKBONE_TYPE

        # ------------- Backbone -----------------------
        self.backbone = Backbone(BACKBONE_TYPE)

        self.backbone_pooling = GeneralizedMeanPoolingP()
        self.backbone_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)

    def forward(self, x_vis, x_inf, modal):
        shared_feature_map, specific_feature_map = self.backbone(x_vis, x_inf, modal)

        if self.training:
            return shared_feature_map, specific_feature_map
        else:
            eval_features = []

            # Backbone
            backbone_features = self.backbone_pooling(shared_feature_map).squeeze()
            backbone_bn_features, backbone_cls_score = self.backbone_classifier(backbone_features)
            eval_features.append(backbone_bn_features)

            eval_features = torch.cat(eval_features, dim=1)
            return eval_features


class Classifier(nn.Module):
    """
    BN -> Classifier
    """

    def __init__(self, c_dim, pid_num):
        super(Classifier, self).__init__()
        self.pid_num = pid_num

        self.bottleneck = nn.BatchNorm1d(c_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(c_dim, self.pid_num, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features):
        bn_features = self.bottleneck(features.squeeze())
        cls_score = self.classifier(bn_features)
        return bn_features, cls_score


class Backbone(nn.Module):
    def __init__(self, BACKBONE_TYPE):
        super(Backbone, self).__init__()
        # resnet = torchvision.models.resnet50(pretrained=True)
        resnet = None
        if BACKBONE_TYPE == "resnet50":
            resnet = resnet50(pretrained=True)
        elif BACKBONE_TYPE == "resnet50_ibn_a":
            resnet = resnet50_ibn_a(pretrained=True)

        # Modifiy backbone
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        # Backbone structure
        self.low_shared_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        self.shared_layer = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
        )

        self.specific_layer = copy.deepcopy(self.shared_layer)

    def forward(self, x_vis, x_inf, modal):
        if modal == "all":
            x = torch.cat([x_vis, x_inf], dim=0)
        elif modal == "vis":
            x = x_vis
        elif modal == "inf":
            x = x_inf
        outs = self.low_shared_layer(x)
        shared_feature_map = self.shared_layer(outs)
        specific_feature_map = self.specific_layer(outs)
        return shared_feature_map, specific_feature_map
