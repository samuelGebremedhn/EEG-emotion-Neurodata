import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_emotion.config import Config
from eeg_emotion.data import iter_deap, save_windows
from eeg_emotion.preprocessing import prepare_trial

p = argparse.ArgumentParser()
p.add_argument("--raw-dir", required=True)
p.add_argument("--output", required=True)
p.add_argument("--config", default="configs/default.yaml")
args = p.parse_args()
cfg = Config.from_yaml(ROOT / args.config)

Xs=[]; yv=[]; ya=[]; subs=[]; trials=[]
for subject, data, labels in iter_deap(args.raw_dir):
    for trial_id, (trial, lab) in enumerate(zip(data, labels)):
        # Common DEAP labels: valence and arousal are the first two values.
        epochs = prepare_trial(trial, cfg.sampling_rate, cfg.low_hz, cfg.high_hz, cfg.window_seconds, cfg.window_overlap, cfg.baseline_seconds)
        Xs.append(epochs)
        yv.extend([int(lab[0] >= cfg.label_threshold)] * len(epochs))
        ya.extend([int(lab[1] >= cfg.label_threshold)] * len(epochs))
        subs.extend([subject] * len(epochs))
        trials.extend([trial_id] * len(epochs))

X = np.concatenate(Xs, axis=0)
save_windows(args.output, X, yv, ya, subs, trials)
print(f"Saved {X.shape} to {args.output}")
