from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity), "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def optimize_threshold(labels: np.ndarray, scores: np.ndarray, min_sensitivity: float = 0.85) -> tuple[float, list[dict]]:
    candidates = np.unique(np.concatenate(([0.01], np.linspace(0.05, 0.95, 181), scores, [0.99])))
    rows = [calculate_metrics(labels, scores, float(value)) for value in candidates]
    eligible = [row for row in rows if row["recall_sensitivity"] >= min_sensitivity]
    pool = eligible or rows
    best = max(pool, key=lambda row: (row["f1"], row["specificity"], row["accuracy"]))
    return best["threshold"], rows


def reliability_points(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    return calibration_curve(labels, scores, n_bins=bins, strategy="uniform")

