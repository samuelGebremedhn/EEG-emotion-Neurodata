import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(x, sfreq, low=4.0, high=45.0, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def baseline_correct(x, baseline_samples=128 * 3):
    baseline_samples = min(baseline_samples, x.shape[-1])
    return (x - x[..., :baseline_samples].mean(axis=-1, keepdims=True)).astype(np.float32)


def epoch_trial(trial, sfreq, window_seconds=4.0, overlap=0.5):
    n = int(window_seconds * sfreq)
    step = max(1, int(n * (1.0 - overlap)))
    epochs = []
    for start in range(0, trial.shape[-1] - n + 1, step):
        epochs.append(trial[..., start:start+n])
    return np.stack(epochs) if epochs else np.empty((0, trial.shape[0], n), dtype=np.float32)


def prepare_trial(trial, sfreq, low=4.0, high=45.0, window_seconds=4.0, overlap=0.5, baseline_seconds=3.0):
    # Remove the explicit pre-stimulus baseline before creating learning windows.
    trial = trial.astype(np.float32)
    baseline_n = int(baseline_seconds * sfreq)
    if trial.shape[-1] <= baseline_n:
        raise ValueError("Trial is shorter than the configured baseline interval")
    trial = trial[..., baseline_n:]
    filtered = bandpass_filter(trial, sfreq, low, high)
    return epoch_trial(filtered, sfreq, window_seconds, overlap)
