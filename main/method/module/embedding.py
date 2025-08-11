import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .batch_norm import get_norm
from .weights_init import weights_init_classifier, weights_init_kaiming


class Embedding(nn.Module):
    def __init__(self, input_dim, out_dim, bias=False, bias_freeze=False):
        super(Embedding, self).__init__()

        self.embedding = nn.Conv2d(input_dim, out_dim, 1, 1, bias=bias)
        self.embedding.apply(weights_init_kaiming)

        self.bn = get_norm("BN", out_dim, bias_freeze=bias_freeze)
        self.bn.apply(weights_init_kaiming)

    def forward(self, features):
        f = self.embedding(features)
        f = self.bn(f)
        return F.relu(f, inplace=True)


class FCEmbedding(nn.Module):
    def __init__(self, input_dim, out_dim, bias=False):
        super(FCEmbedding, self).__init__()

        self.embedding = nn.Linear(input_dim, out_dim, bias=bias)
        self.embedding.apply(weights_init_kaiming)

        self.bn = nn.BatchNorm1d(out_dim)
        self.bn.apply(weights_init_kaiming)

    def forward(self, features):
        f = self.embedding(features)
        f = self.bn(f)
        return f
