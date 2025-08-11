import copy

import torch
import torch.nn as nn

from .module import resnet50, resnet50_ibn_a


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
        self.resnet_preprocessing_layer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        self.copyed_resnet_layer4 = copy.deepcopy(resnet.layer4)

    def forward(self, x):

        preprocessing_out = self.resnet_preprocessing_layer(x)

        l1_out = self.resnet_layer1(preprocessing_out)
        l2_out = self.resnet_layer2(l1_out)
        l3_out = self.resnet_layer3(l2_out)
        l4_out = self.resnet_layer4(l3_out)

        copy_l4_out = self.copyed_resnet_layer4(l3_out)

        internal_outs = [l1_out, l2_out, l3_out]

        return l4_out, copy_l4_out, internal_outs
