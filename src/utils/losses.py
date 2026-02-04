"""Custom loss functions."""
import torch


def spike_aware_huber(yhat, y, delta=0.9, alpha=2.0):
    err = (yhat - y).abs()
    huber = torch.where(
        err <= delta,
        0.5 * err**2,
        delta * (err - 0.5 * delta)
    )
    # penalize under-prediction more
    weight = torch.where(err < 0, alpha, 1.0)
    return (weight * huber).mean()

