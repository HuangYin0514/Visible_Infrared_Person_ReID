import torch.nn as nn


class Criterion:
    def __init__(self, config):
        self.name = "Criterion"
        self.load_criterion(config)

    def load_criterion(self, config):
        self.id = nn.CrossEntropyLoss()
