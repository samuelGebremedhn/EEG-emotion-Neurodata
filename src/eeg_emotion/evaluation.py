import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_indices(y, subjects, mode="subject", test_size=0.2, seed=42):
    idx = np.arange(len(y))
    if mode == "subject":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(idx, y, groups=subjects))
        return tr, te
    if mode == "random":
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    raise ValueError("mode must be 'subject' or 'random'")


def metrics(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["brier_score"] = brier_score_loss(y_true, y_prob)
    return out


def classification_table(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def calibration_data(y_true, y_prob, bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted_probability": mean_pred, "fraction_positive": frac_pos})
