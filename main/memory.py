import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, nn


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_classes, temperature=0.05, momentum=0.01):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.momentum = momentum
        self.temperature = temperature

        # 初始化样本记忆
        self.features_memory = torch.randn(num_classes, num_features)

    def updateMemory(self, inputs, targets):
        with torch.no_grad():
            features = self.features_memory
            momentum = self.momentum
            for x, y in zip(inputs, targets):
                features[y] = momentum * features[y] + (1.0 - momentum) * x
                features[y] /= features[y].norm()
            self.features_memory = features
