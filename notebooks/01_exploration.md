# Exploration checklist

Use this notebook outline to document the first inspection of the prepared DEAP windows.

1. Verify shapes and subject counts.
2. Plot one EEG window across channels.
3. Plot PSD for representative channels.
4. Inspect valence/arousal class balance.
5. Compare per-subject label distributions.
6. Confirm that subject IDs survive preprocessing.
7. Save a short data-quality note before training.

Example:

```python
import numpy as np
z = np.load('data/processed/deap_windows.npz')
print(z['X'].shape)
print(np.unique(z['subjects'], return_counts=True))
print(np.bincount(z['y_valence']))
```
