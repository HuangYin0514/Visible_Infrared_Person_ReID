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

        # ------------- Specific -----------------------
        self.specific_pooling = GeneralizedMeanPoolingP()
        self.specific_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)

        # ------------- Modal classification -----------------------
        self.dual_modal_classifier = Classifier(BACKBONE_FEATURES_DIM, 2)
        self.tri_modal_classifier = Classifier(BACKBONE_FEATURES_DIM, 3)

        # ------------- Modal Fusion -----------------------
        self.modal_fusion = Modal_Fusion(BACKBONE_FEATURES_DIM * 3, BACKBONE_FEATURES_DIM)
        self.modal_fusion_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)

    def forward(self, x_vis, x_inf, modal):
        backbone_feature_map, specific_feature_map = self.backbone(x_vis, x_inf, modal)

        if self.training:
            return backbone_feature_map, specific_feature_map
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

        self.backbone_encoder_module = Backbone_Encoder_Module(resnet, use_NL=False)
        self.backbone_decoupling_module = Backbone_Decoupling_Module(resnet, use_NL=False)

        self.shared_module = self.backbone_decoupling_module
        self.specific_module = copy.deepcopy(self.backbone_decoupling_module)

    def forward(self, x_vis, x_inf, modal):
        if modal == "all":
            x = torch.cat([x_vis, x_inf], dim=0)
        elif modal == "vis":
            x = x_vis
        elif modal == "inf":
            x = x_inf

        encoder_out = self.backbone_encoder_module(x)
        shared_out = self.shared_module(encoder_out)
        specific_out = self.specific_module(encoder_out)
        return shared_out, specific_out


class Backbone_Encoder_Module(nn.Module):
    """conv1 bn1 relu maxpool layer1 layer2"""

    def __init__(self, resnet, use_NL=False):
        super(Backbone_Encoder_Module, self).__init__()
        self.use_NL = use_NL

        self.pre_processing_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 3 blocks
        )
        self.layer2 = resnet.layer2  # 4 blocks
        if self.use_NL:
            self.NL_2 = nn.ModuleList([Non_local(512) for i in range(2)])

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

    def forward(self, x):
        out = self.pre_processing_layer(x)
        if self.use_NL:
            out = self._NL_forward_layer(out, self.layer2, self.NL_2)
        else:
            out = self.layer2(out)
        return out


class Backbone_Decoupling_Module(nn.Module):
    """layer3 layer4"""

    def __init__(self, resnet, use_NL=False):
        super(Backbone_Decoupling_Module, self).__init__()
        self.use_NL = use_NL

        self.layer3 = resnet.layer3  # 6 blocks
        self.layer4 = resnet.layer4  # 3 blocks

        if self.use_NL:
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

    def forward(self, x):
        if self.use_NL:
            out = self._NL_forward_layer(x, self.layer3, self.NL_3)
        else:
            out = self.layer3(x)
        out = self.layer4(out)
        return out


class Modal_Fusion(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(Modal_Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.fusion.apply(weights_init_kaiming)

    def forward(self, shared_vis_feat, shared_inf_feat, specific_vis_feat, specific_inf_feat):
        shared_feat = shared_vis_feat + shared_inf_feat
        feat = torch.cat([shared_feat, specific_vis_feat, specific_inf_feat], dim=1).unsqueeze(2)
        fused_feat = self.fusion(feat).squeeze(2) + shared_feat
        return fused_feat
