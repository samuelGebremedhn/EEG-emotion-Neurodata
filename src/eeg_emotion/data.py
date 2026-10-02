from pathlib import Path
import pickle
import numpy as np


def load_deap_file(path):
    """Load one DEAP Python-format .dat participant file.

    Returns EEG shape (trials, channels, samples) and labels shape (trials, 4).
    DEAP labels are ordered valence, arousal, dominance, liking.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    data = np.asarray(obj["data"], dtype=np.float32)
    labels = np.asarray(obj["labels"], dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D EEG array, got {data.shape}")
    return data[:, :32, :], labels


def iter_deap(raw_dir):
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("s*.dat"))
    if not files:
        raise FileNotFoundError(f"No s*.dat files found in {raw_dir}")
    for path in files:
        subject = int(path.stem[1:])
        yield subject, *load_deap_file(path)


def save_windows(path, X, y_valence, y_arousal, subjects, trial_ids=None):
    payload = {
        "X": np.asarray(X, dtype=np.float32),
        "y_valence": np.asarray(y_valence, dtype=np.int64),
        "y_arousal": np.asarray(y_arousal, dtype=np.int64),
        "subjects": np.asarray(subjects, dtype=np.int64),
    }
    if trial_ids is not None:
        payload["trial_ids"] = np.asarray(trial_ids, dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_windows(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
