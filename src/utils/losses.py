import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, q: float):
        super().__init__()
        self.q = q

    def forward(self, preds, target):
        """
        preds: (batch, horizon)
        target: (batch, horizon)
        """
        errors = target - preds
        loss = torch.max(
            self.q * errors,
            (self.q - 1) * errors
        )
        return loss.mean()


import torch
import torch.nn as nn

class WeightedMSE(nn.Module):
    def __init__(self, alpha: float = 1.0, eps: float = 1e-6):
        """
        alpha: độ mạnh của weighting
        eps: tránh chia 0
        """
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, preds, target):
        """
        preds: (batch, horizon)
        target: (batch, horizon)
        """

        # Normalize magnitude inside batch
        mean_abs = target.abs().mean().detach() + self.eps
        weight = 1.0 + self.alpha * (target.abs() / mean_abs)

        loss = weight * (preds - target) ** 2
        return loss.mean()