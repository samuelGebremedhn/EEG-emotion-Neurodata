import numpy as np
import torch


def gradient_saliency(model, X, target_class=None):
    """Return absolute input gradients for X=(batch, channels, time)."""
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(dim=1)
    score = logits.gather(1, target_class.view(-1, 1)).sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    return x.grad.detach().abs().numpy()


def channel_importance(saliency):
    return saliency.mean(axis=(0, 2))
