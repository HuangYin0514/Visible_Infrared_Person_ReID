import copy

import torch
import torch.nn as nn
from gem_pool import GeneralizedMeanPoolingP
from model_main_tool import Interaction
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

        # ------------- interaction -----------------------
        self.interaction = Interaction()
        self.interaction_pooling = GeneralizedMeanPoolingP()
        self.interaction_classifier = Classifier(BACKBONE_FEATURES_DIM, n_class)

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


#############################################################
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


#############################################################
class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


#############################################################
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
