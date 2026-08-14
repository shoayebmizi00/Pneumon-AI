from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch import nn

from training.dataset import make_loaders
from training.metrics import calculate_metrics, reliability_points
from training.models import create_model
from training.train import run_epoch
from training.utils import read_json, write_json


def save_plots(labels: np.ndarray, scores: np.ndarray, metrics: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"], ax=ax)
    ax.set(xlabel="Predicted class", ylabel="Actual class", title="Untouched test-set confusion matrix")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=200); plt.close(fig)
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "--", color="#64748b"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve"); ax.legend(); fig.tight_layout(); fig.savefig(output / "roc_curve.png", dpi=200); plt.close(fig)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(recall, precision, label=f"AP = {metrics['pr_auc']:.3f}"); ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve"); ax.legend(); fig.tight_layout(); fig.savefig(output / "precision_recall_curve.png", dpi=200); plt.close(fig)
    observed, predicted = reliability_points(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(predicted, observed, "o-", label="Model"); ax.plot([0, 1], [0, 1], "--", label="Ideal"); ax.set(xlabel="Mean prediction score", ylabel="Observed frequency", title="Reliability diagram"); ax.legend(); fig.tight_layout(); fig.savefig(output / "reliability_diagram.png", dpi=200); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time evaluation on the untouched patient-level test set.")
    parser.add_argument("--dataset", default="pneumonia_dataset/chest_xray")
    parser.add_argument("--manifest", default="results/split_manifest.csv")
    parser.add_argument("--model", default="models/best_pneumonia_model.pth")
    parser.add_argument("--metadata", default="models/model_metadata.json")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing final test report.")
    args = parser.parse_args()
    report_path = Path("results/test_metrics.json")
    if report_path.exists() and not args.force:
        raise SystemExit("Test metrics already exist. Refusing repeated test-set evaluation; pass --force only with a documented reason.")
    metadata = read_json(args.metadata)
    image_size = int(metadata["image_size"][0])
    loaders, counts = make_loaders(args.dataset, image_size, 24, 0, args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model = create_model(metadata["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts["NORMAL"] / counts["PNEUMONIA"]], device=device))
    _, labels, _, logits = run_epoch(model, loaders["test"], criterion, device)
    scores = 1 / (1 + np.exp(-(logits / float(metadata.get("temperature", 1.0)))))
    metrics = calculate_metrics(labels, scores, float(metadata["threshold"]))
    write_json(report_path, metrics)
    metadata["test_metrics"] = metrics
    write_json(args.metadata, metadata)
    save_plots(labels, scores, metrics, Path("results"))
    print(metrics)


if __name__ == "__main__":
    main()
