import numpy as np
from scipy.signal import welch


def _bandpower(psd, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(psd.shape[:-1], dtype=np.float32)
    return np.trapezoid(psd[..., mask], freqs[mask], axis=-1)


def bandpower_features(X, sfreq, bands):
    """Relative band-power features. X=(windows, channels, samples)."""
    freqs, psd = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(sfreq * 2)), axis=-1)
    total = np.trapezoid(psd, freqs, axis=-1) + 1e-12
    cols, feats = [], []
    for name, (lo, hi) in bands.items():
        bp = _bandpower(psd, freqs, lo, hi)
        rel = bp / total
        feats.append(rel)
        cols.extend([f"{name}_ch{c+1:02d}" for c in range(X.shape[1])])
    F = np.concatenate([f.reshape(f.shape[0], -1) for f in feats], axis=1)
    return F.astype(np.float32), cols


def spectral_entropy_features(X, sfreq):
    freqs, psd = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(sfreq * 2)), axis=-1)
    p = psd / (psd.sum(axis=-1, keepdims=True) + 1e-12)
    ent = -(p * np.log2(p + 1e-12)).sum(axis=-1)
    return ent.astype(np.float32)


def extract_features(X, sfreq, bands):
    bp, names = bandpower_features(X, sfreq, bands)
    ent = spectral_entropy_features(X, sfreq)
    enames = [f"spectral_entropy_ch{c+1:02d}" for c in range(X.shape[1])]
    return np.concatenate([bp, ent], axis=1), names + enames
