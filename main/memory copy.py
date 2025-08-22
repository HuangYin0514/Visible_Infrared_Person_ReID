import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, nn


class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, features):
        ctx.features = features
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        features = ctx.features

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(features)

        return grad_inputs, None


class MemoryBank(nn.Module):
    def __init__(self, num_features, num_classes, temperature=0.05, momentum=0.01):
        super(MemoryBank, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.momentum = momentum
        self.temperature = temperature

        # 初始化样本记忆
        self.features_memory = nn.Parameter(torch.randn(num_classes, num_features))

    def updateMemory(self, inputs, targets):
        with torch.no_grad():
            features = self.features_memory
            momentum = self.momentum
            for x, y in zip(inputs, targets):
                features[y] = momentum * features[y] + (1.0 - momentum) * x
                features[y] /= features[y].norm()
            self.features_memory = features

    def forward(self, inputs, labels, epoch=None):
        norm_inputs = F.normalize(inputs, dim=1)

        # alpha = self.alpha * epoch
        outputs = CM.apply(norm_inputs, self.features_memory)
        outputs /= self.temperature
        loss = F.cross_entropy(outputs, labels)
        return loss


if __name__ == "__main__":
    print("main")
    # 定义测试参数
    num_features = 2048  # 特征维度
    num_classes = 702  # 类别数
    batch_size = 64  # 批量大小

    # 初始化网络
    model = MemoryBank(num_features, num_classes)

    # 创建随机输入数据 (batch_size, num_features)
    features = torch.randn(batch_size, num_features, requires_grad=True)

    # 随机生成标签 (batch_size)
    targets = torch.randint(0, num_classes, (batch_size,))

    # 设定 epoch（影响 alpha 的计算）
    epoch = 10

    # 前向传播
    loss = model(features, targets, epoch=epoch)

    # 反向传播
    loss.backward()

    model.updateMemory(features, targets)

    # 打印损失
    print("Loss:", loss.item())

    # 打印部分梯度检查
    print("Input gradients:", features.grad)
