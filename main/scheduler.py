import torch.optim as optim


class Scheduler:
    def __init__(self):
        self.name = "Scheduler"

    def adjust_learning_rate(self, config, optimizer, epoch):

        if epoch < config.SCHEDULER.MILESTONES[0]:
            lr = config.OPTIMIZER.LEARNING_RATE * (epoch + 1) / 10

        elif epoch >= config.SCHEDULER.MILESTONES[0] and epoch < config.SCHEDULER.MILESTONES[1]:
            lr = config.OPTIMIZER.LEARNING_RATE * config.SCHEDULER.FACTORS[0]

        elif epoch >= config.SCHEDULER.MILESTONES[1] and epoch < config.SCHEDULER.MILESTONES[2]:
            lr = config.OPTIMIZER.LEARNING_RATE * config.SCHEDULER.FACTORS[1]

        elif epoch >= config.SCHEDULER.MILESTONES[2]:
            lr = config.OPTIMIZER.LEARNING_RATE * config.SCHEDULER.FACTORS[2]

        optimizer.param_groups[0]["lr"] = 0.1 * lr  # backbone 网络
        for i in range(len(optimizer.param_groups) - 1):
            optimizer.param_groups[i + 1]["lr"] = lr  # 其余网络参数

        return lr
