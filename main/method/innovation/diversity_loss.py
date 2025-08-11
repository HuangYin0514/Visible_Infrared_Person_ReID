import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def diversity_loss(x, gamma=1.0):
    for i in range(len(x)):
        x[i] = F.normalize(x[i], dim=1)

    loss = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            loss = loss + torch.mean(torch.sum(x[i] * x[j], dim=1))

    loss = gamma * loss

    return loss
