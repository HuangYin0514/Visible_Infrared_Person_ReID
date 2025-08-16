import copy

import torch
import torch.nn as nn
from gem_pool import GeneralizedMeanPoolingP
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

        # ------------- Modal fusion -----------------------
        self.modal_fusion_layer = Modal_Fusion(BACKBONE_FEATURES_DIM, BACKBONE_FEATURES_DIM)
        self.modal_fusion_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)

    def forward(self, x_vis, x_inf, modal):
        resnet_feature_map = self.backbone(x_vis, x_inf, modal)

        if self.training:
            return resnet_feature_map
        else:
            eval_features = []

            # Backbone
            backbone_features = self.backbone_pooling(resnet_feature_map).squeeze()
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
        self.vis_pre_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        self.inf_pre_layer = copy.deepcopy(self.vis_pre_layer)

        self.shared_layer = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x_vis, x_inf, modal):
        if modal == "all":
            x_vis = self.vis_pre_layer(x_vis)
            x_inf = self.inf_pre_layer(x_inf)
            x = torch.cat([x_vis, x_inf], dim=0)
        elif modal == "vis":
            x_vis = self.vis_pre_layer(x_vis)
            x = x_vis
        elif modal == "inf":
            x_inf = self.inf_pre_layer(x_inf)
            x = x_inf

        l4_out = self.shared_layer(x)

        return l4_out


class Modal_Fusion(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(Modal_Fusion, self).__init__()

        self.cbr = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 1, 1, 0),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        self.fusion_layer = Residual(
            nn.Sequential(
                self.cbr,
            )
        )
        self.fusion_layer.apply(weights_init_kaiming)

    def forward(self, features_1, features_2):
        feature = (features_1 + features_2).unsqueeze(-1)  # (bs, 1, 2048)
        fused = self.fusion_layer(feature).squeeze(-1)  # (bs, 2048)
        return fused


class Residual(nn.Module):
    """
    残差模块类，用于实现残差连接。
    """

    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("InstanceNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
