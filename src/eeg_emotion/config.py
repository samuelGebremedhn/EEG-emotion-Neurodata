from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    sampling_rate: int = 128
    channels: int = 32
    low_hz: float = 4.0
    high_hz: float = 45.0
    window_seconds: float = 4.0
    window_overlap: float = 0.5
    baseline_seconds: float = 3.0
    label_threshold: float = 5.0
    bands: dict | None = None
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(**d)

    @property
    def window_samples(self):
        return int(self.window_seconds * self.sampling_rate)
