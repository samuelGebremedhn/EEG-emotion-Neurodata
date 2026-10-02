import torch
from torch import nn

class EEGNet(nn.Module):
    """Compact EEGNet-style network for X=(batch, channels, time)."""
    def __init__(self, channels=32, samples=512, n_classes=2):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), groups=16, padding=(0, 8), bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            n = self._forward_features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(n, n_classes)

    def _forward_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.depthwise(x)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.classifier(self._forward_features(x).flatten(1))
