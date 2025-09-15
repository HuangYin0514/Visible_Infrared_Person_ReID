import torch.nn as nn
from cross_entropy_label_smooth import CrossEntropyLabelSmooth
from ori_triplet_loss import OriTripletLoss


class Criterion:
    def __init__(self, config):
        self.name = "Criterion"
        self.load_criterion(config)

    def load_criterion(self, config):
        self.id = nn.CrossEntropyLoss()
        # self.id_ls = CrossEntropyLabelSmooth()
        self.tri = OriTripletLoss(batch_size=config.DATALOADER.BATCHSIZE, margin=0.3)
