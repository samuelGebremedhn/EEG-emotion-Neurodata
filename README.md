# EEG Emotion-State Classification

A reproducible Neurodata portfolio project for EEG-based emotion-state classification using the **DEAP** dataset. The project implements leakage-aware preprocessing, interpretable frequency-band features, classical machine learning, a compact EEGNet-style deep model, subject-independent evaluation, calibration/confusion-matrix reporting, frequency/channel ablations, and a Streamlit demonstration.

> **Research/portfolio project — not a clinical diagnostic system.**

## Research question

Can EEG recordings support subject-independent classification of self-reported **valence** and **arousal** states? The project compares an interpretable band-power pipeline against an end-to-end compact neural network and explicitly contrasts random trial splits with subject-wise splits.

## Dataset

This repository is designed for **DEAP: A Dataset for Emotion Analysis using Physiological and Audiovisual Signals**. DEAP contains EEG and peripheral physiological recordings from 32 participants who watched 40 one-minute music-video excerpts and rated affective dimensions including valence and arousal.

DEAP access is controlled by its EULA. **Do not commit or redistribute DEAP files in this repository.** Obtain the dataset through the official DEAP process and place the authorized Python-format files under `data/raw/`.

Official dataset page: https://eecs.qmul.ac.uk/mmv/datasets/deap/
Official download page: https://eecs.qmul.ac.uk/mmv/datasets/deap/download.html
Official EULA: https://eecs.qmul.ac.uk/mmv/datasets/deap/doc/eula.pdf

## Project structure

```text
.
├── app/streamlit_app.py
├── configs/default.yaml
├── notebooks/01_exploration.md
├── reports/RESULTS_TEMPLATE.md
├── scripts/
│   ├── generate_demo_data.py
│   ├── prepare_deap.py
│   ├── train_classical.py
│   ├── train_eegnet.py
│   ├── evaluate.py
│   └── run_ablation.py
├── src/eeg_emotion/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── utils.py
├── tests/
├── requirements.txt
└── README.md
```

## Environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 1. Verify the project without DEAP

Generate a small synthetic EEG dataset and run the full classical pipeline:

```bash
python scripts/generate_demo_data.py
python scripts/train_classical.py --input data/processed/demo_windows.npz --output results/demo_classical.csv
python -m pytest -q
```

This smoke test is intentionally independent of DEAP so the GitHub repository can be tested without distributing restricted research data.

## 2. Prepare DEAP

After obtaining authorized access, place the DEAP Python-format participant files in `data/raw/`. The expected files are normally named like `s01.dat`, `s02.dat`, etc.

Then run:

```bash
python scripts/prepare_deap.py --raw-dir data/raw --output data/processed/deap_windows.npz
```

The preparation script:

1. reads the 32 EEG channels from DEAP,
2. applies a configurable band-pass filter,
3. optionally removes baseline samples,
4. segments trials into fixed windows,
5. computes binary valence/arousal labels from the configured threshold,
6. stores subject IDs so train/test separation can be performed by participant.

## 3. Classical baseline

```bash
python scripts/train_classical.py \
  --input data/processed/deap_windows.npz \
  --target valence \
  --split subject \
  --output results/classical_valence.csv
```

Supported baseline models include logistic regression, random forest, and gradient boosting. Features include relative band power in delta, theta, alpha, beta and gamma bands plus spectral entropy.

## 4. EEGNet-style deep model

```bash
python scripts/train_eegnet.py \
  --input data/processed/deap_windows.npz \
  --target valence \
  --epochs 30 \
  --output models/eegnet_valence.pt
```

The neural pipeline uses a compact EEGNet-style architecture implemented locally in PyTorch. No pretrained weights or external model repository are required.

## 5. Leakage comparison

To demonstrate why EEG leakage matters, compare:

```bash
python scripts/train_classical.py --input data/processed/deap_windows.npz --target valence --split random --output results/random_split.csv
python scripts/train_classical.py --input data/processed/deap_windows.npz --target valence --split subject --output results/subject_split.csv
```

The **subject-wise result is the primary portfolio result**. Random window/trial splits can place recordings from the same person in both train and test sets and therefore can overestimate generalization to unseen people.

## Metrics

The evaluation module reports:

- accuracy
- balanced accuracy
- macro-F1
- ROC-AUC when probabilities are available
- confusion matrix
- per-class precision/recall/F1
- Brier score and calibration curve where applicable

## Explainability

The classical pipeline exposes feature importance. The neural pipeline includes gradient-based saliency over channels/time. These are intended to show which signal regions or frequency-derived features contributed to predictions; they are **not evidence that a particular brain region causes an emotion**.

## Streamlit demo

After generating a compatible window file or CSV:

```bash
streamlit run app/streamlit_app.py
```

The demo accepts a prepared `.npz` window file, displays signal traces, and shows a compatible trained model prediction. It is an educational visualization and should not be used for clinical decisions.

## Reproducibility checklist

- Fixed random seeds where practical.
- Dataset paths are external and never committed.
- Subject IDs are preserved through preprocessing.
- Preprocessing configuration is stored in `configs/default.yaml`.
- Primary evaluation uses subject-wise splits.
- Results are exported as CSV.
- Synthetic smoke-test data is included through a generator rather than committed data.

## Limitations

DEAP is a relatively small, controlled research dataset. Self-reported labels are subjective; participant variability is substantial; the stimulus protocol does not represent all real-world emotional states; EEG artifacts and preprocessing choices can affect results; and model performance on DEAP must not be interpreted as clinical validity or universal emotion recognition.

## Citation

If you use DEAP, cite:

> S. Koelstra et al., “DEAP: A Database for Emotion Analysis using Physiological Signals,” IEEE Transactions on Affective Computing, vol. 3, no. 1, pp. 18–31, 2012.

See the official DEAP EULA for the dataset's required acknowledgement and usage conditions.
