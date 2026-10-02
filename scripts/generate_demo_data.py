from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_emotion.data import save_windows

rng = np.random.default_rng(42)
n_subjects, trials_per_subject, channels, samples = 10, 12, 32, 512
X, yv, ya, subjects = [], [], [], []
t = np.arange(samples) / 128
for s in range(1, n_subjects + 1):
    for _ in range(trials_per_subject):
        v = int(rng.integers(0, 2)); a = int(rng.integers(0, 2))
        x = rng.normal(0, 1, (channels, samples)).astype(np.float32)
        # Add weak, class-linked synthetic alpha/beta components for a smoke test.
        f = 10 if v else 20
        amp = 0.7 + 0.2 * a
        x += amp * np.sin(2*np.pi*f*t)[None, :].astype(np.float32)
        X.append(x); yv.append(v); ya.append(a); subjects.append(s)

out = ROOT / "data/processed/demo_windows.npz"
save_windows(out, np.stack(X), yv, ya, subjects)
print(f"Wrote {out} with {len(X)} windows")
