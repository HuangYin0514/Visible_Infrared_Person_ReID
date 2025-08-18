import torch.optim as optim


class Scheduler:
    def __init__(self):
        self.name = "Scheduler"

    def adjust_learning_rate(self, config, optimizer, epoch):

        CONFIG_LR = config.OPTIMIZER.LEARNING_RATE

        if epoch < 10:
            lr = CONFIG_LR * (epoch + 1) / 10
        elif epoch >= 10 and epoch < 20:
            lr = CONFIG_LR
        elif epoch >= 20 and epoch < 50:
            lr = CONFIG_LR * 0.1
        elif epoch >= 50:
            lr = CONFIG_LR * 0.01

        optimizer.param_groups[0]["lr"] = 0.1 * lr
        for i in range(len(optimizer.param_groups) - 1):
            optimizer.param_groups[i + 1]["lr"] = lr

        return lr
