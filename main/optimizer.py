import torch.optim as optim


class Optimizer:
    def __init__(self, config, net):
        self.name = "optimizer"
        self.load_optimizer(config, net)

    def load_optimizer(self, config, net):
        ################################################################################
        # Ignored parameters
        ignored_params = []
        ignored_params += list(map(id, net.backbone_classifier.parameters()))

        ################################################################################
        # Base parameters
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        ################################################################################
        # Optimizer
        optimizer = optim.SGD(
            [
                {"params": base_params, "lr": 0.1 * config.OPTIMIZER.LEARNING_RATE},
                {"params": net.backbone_classifier.parameters(), "lr": config.OPTIMIZER.LEARNING_RATE},
            ],
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        )
        self.optimizer = optimizer
