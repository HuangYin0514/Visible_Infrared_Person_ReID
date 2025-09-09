import copy

import torch
import torch.nn as nn
from gem_pool import GeneralizedMeanPoolingP
from mamba import CrossModalMamba
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

        # ------------- Modal interaction -----------------------
        self.modal_interaction = Modal_Interaction(BACKBONE_FEATURES_DIM)

        # ------------- Modal calibration -----------------------
        self.modal_calibration = Modal_Calibration(BACKBONE_FEATURES_DIM)

        # ------------- modal propagation -----------------------
        self.modal_propagation_pooling = GeneralizedMeanPoolingP()
        self.modal_propagation_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)
        self.modal_propagation = DistillKL(T=4)

        # ------------- weights init -----------------------
        self.modal_interaction.apply(weights_init_kaiming)
        self.modal_calibration.apply(weights_init_kaiming)

    def forward(self, x_vis, x_inf, modal):
        backbone_feature_map = self.backbone(x_vis, x_inf, modal)

        if self.training:
            return backbone_feature_map
        else:
            eval_features = []

            # Backbone
            backbone_features = self.backbone_pooling(backbone_feature_map).squeeze()
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
        self.vis_specific_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.inf_specific_layer = copy.deepcopy(self.vis_specific_layer)

        self.layer1 = resnet.layer1  # 3 blocks
        self.layer2 = resnet.layer2  # 4 blocks
        self.layer3 = resnet.layer3  # 6 blocks
        self.layer4 = resnet.layer4  # 3 blocks

        self.NL_2 = nn.ModuleList([Non_local(512) for i in range(2)])
        self.NL_3 = nn.ModuleList([Non_local(1024) for i in range(3)])

    def _NL_forward_layer(self, x, layer, NL_modules):
        num_blocks = len(layer)
        nl_start_idx = num_blocks - len(NL_modules)  # 从倒数层开始插入
        nl_counter = 0
        for i, block in enumerate(layer):
            x = block(x)
            if i >= nl_start_idx:
                x = NL_modules[nl_counter](x)
                nl_counter += 1
        return x

    def forward(self, x_vis, x_inf, modal):
        if modal == "all":
            x_vis = self.vis_specific_layer(x_vis)
            x_inf = self.inf_specific_layer(x_inf)
            x = torch.cat([x_vis, x_inf], dim=0)
        elif modal == "vis":
            x_vis = self.vis_specific_layer(x_vis)
            x = x_vis
        elif modal == "inf":
            x_inf = self.inf_specific_layer(x_inf)
            x = x_inf

        out = self.layer1(x)
        out = self._NL_forward_layer(out, self.layer2, self.NL_2)
        out = self._NL_forward_layer(out, self.layer3, self.NL_3)
        out = self.layer4(out)

        return out


class Modal_Interaction(nn.Module):
    def __init__(self, c_dim):
        super(Modal_Interaction, self).__init__()
        self.MAMBA = CrossModalMamba(in_cdim=c_dim, hidden_cdim=256)
        # self.vis_weight = nn.Parameter(torch.tensor(0.001))
        # self.inf_weight = nn.Parameter(torch.tensor(0.001))
        self.M_con = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_dim),
            # nn.ReLU(),
        )
        self.vis_sp = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_dim),
            # nn.ReLU(),
        )
        self.inf_sp = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_dim),
            # nn.ReLU(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, vis_feat, inf_feat):
        # vis_mamba_feat, inf_mamba_feat = self.MAMBA(vis_feat, inf_feat)
        M_con = self.M_con(vis_feat + inf_feat)
        M_vis_sp = self.vis_sp(vis_feat)
        M_inf_sp = self.inf_sp(inf_feat)

        vis_feat = vis_feat * (1 - self.sigmoid(M_con * M_vis_sp))
        inf_feat = inf_feat * (1 - self.sigmoid(M_con * M_inf_sp))
        return vis_feat, inf_feat


class Modal_Calibration(nn.Module):
    def __init__(self, c_dim):
        super(Modal_Calibration, self).__init__()
        self.c_dim = c_dim

        self.vis_gate_calibration = Gate_Fusion(c_dim)
        self.inf_gate_calibration = Gate_Fusion(c_dim)

    def forward(self, vis_feat, res_vis_feat, inf_feat, res_inf_feat):
        vis_feat = self.vis_gate_calibration(vis_feat, res_vis_feat)
        inf_feat = self.inf_gate_calibration(inf_feat, res_inf_feat)
        return vis_feat, inf_feat


class Gate_Fusion(nn.Module):
    """

    https://arxiv.org/pdf/2009.14082

    https://0809zheng.github.io/2020/12/01/aff.html

    基于情感表征校准的图文情感分析模型

    """

    def __init__(self, c_dim):
        super(Gate_Fusion, self).__init__()
        self.c_dim = c_dim

        r = 4
        inter_c_dim = int(c_dim // r)

        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_dim, inter_c_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_c_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_c_dim, c_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c_dim),
        )
        self.sigmoid = nn.Sigmoid()

        self.value_stable = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_1, feat_2):
        # feat_1 -> H
        # feat_2 -> E
        # F = g * H + (1-g) * E

        gate = self.sigmoid(self.att(feat_1))
        feat = feat_1 * gate + feat_2 * (1 - gate)
        feat = self.value_stable(feat)

        return feat
