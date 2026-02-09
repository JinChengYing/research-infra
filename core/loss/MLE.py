import torch
from core.registry.loss import register_loss

@register_loss("mle")
class MLELoss:
    def __call__(self, x_pred, x_true):
        # 对每个样本取负 log likelihood
        return ((x_pred - x_true)**2).mean()  # 简单示意，正式可用 GMM log prob
