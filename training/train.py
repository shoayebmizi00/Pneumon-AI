from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from training.config import ExperimentConfig
from training.dataset import make_loaders
from training.metrics import calculate_metrics, optimize_threshold
from training.models import MODEL_NAMES, create_model, set_backbone_trainable
from training.utils import EarlyStopper, seed_everything, write_json


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    losses, labels, scores, logits_out = [], [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets, _paths in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images).reshape(-1)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.item() * len(targets))
            labels.extend(targets.detach().cpu().numpy())
            logits_out.extend(logits.detach().cpu().numpy())
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy())
    return sum(losses) / len(loader.dataset), np.asarray(labels), np.asarray(scores), np.asarray(logits_out)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    value = torch.nn.Parameter(torch.ones(1))
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.LBFGS([value], lr=0.05, max_iter=80)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logits_tensor / value.clamp(0.05, 10), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(value.detach().clamp(0.05, 10).item())


def train_model(name: str, config: ExperimentConfig, loaders, class_counts, device: torch.device,
                reuse_checkpoint: bool = False, finetune_reused: bool = False) -> dict:
    model = create_model(name, config.pretrained).to(device)
    pos_weight = torch.tensor([class_counts["NORMAL"] / class_counts["PNEUMONIA"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    checkpoint = Path(config.output_dir) / "checkpoints" / f"{name}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best_loss, started = [], float("inf"), time.perf_counter()

    resumed_for_finetune = False
    if reuse_checkpoint and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        val_loss, val_labels, val_scores, val_logits = run_epoch(model, loaders["val"], criterion, device)
        if not finetune_reused or name == "baseline":
            print(f"Reused completed checkpoint for {name}.")
            return {"model": name, "checkpoint": str(checkpoint), "val_loss": val_loss,
                    "val_roc_auc": float(roc_auc_score(val_labels, val_scores)), "duration_seconds": 0.0,
                    "history": history, "labels": val_labels, "scores": val_scores, "logits": val_logits,
                    "val_metrics_05": calculate_metrics(val_labels, val_scores, 0.5)}
        best_loss, resumed_for_finetune = val_loss, True
        print(f"Loaded {name} checkpoint for selective fine-tuning.")

    stages = [] if resumed_for_finetune else [("frozen", config.frozen_epochs, config.frozen_lr, False)]
    if name != "baseline" and config.finetune_epochs > 0:
        stages.append(("finetune", config.finetune_epochs, config.finetune_lr, True))
    for stage, epochs, learning_rate, train_backbone in stages:
        set_backbone_trainable(model, name, train_backbone)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
        stopper = EarlyStopper(config.patience)
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            train_loss, train_labels, train_scores, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, val_labels, val_scores, _ = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step(val_loss)
            val_epoch_metrics = calculate_metrics(val_labels, val_scores, 0.5)
            row = {
                "model": name, "stage": stage, "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_accuracy": float(((train_scores >= 0.5) == train_labels).mean()),
                "val_accuracy": float(((val_scores >= 0.5) == val_labels).mean()),
                "val_roc_auc": float(roc_auc_score(val_labels, val_scores)),
                "val_precision": val_epoch_metrics["precision"], "val_recall": val_epoch_metrics["recall_sensitivity"],
                "val_specificity": val_epoch_metrics["specificity"], "val_f1": val_epoch_metrics["f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_duration_seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            print(json.dumps(row))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"state_dict": model.state_dict(), "model_name": name, "image_size": config.image_size}, checkpoint)
            if stopper.update(val_loss):
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    val_loss, val_labels, val_scores, val_logits = run_epoch(model, loaders["val"], criterion, device)
    return {
        "model": name, "checkpoint": str(checkpoint), "val_loss": val_loss,
        "val_roc_auc": float(roc_auc_score(val_labels, val_scores)),
        "duration_seconds": time.perf_counter() - started, "history": history,
        "labels": val_labels, "scores": val_scores, "logits": val_logits,
        "val_metrics_05": calculate_metrics(val_labels, val_scores, 0.5),
    }


def save_training_curves(runs: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for metric, filename, ylabel in (("accuracy", "accuracy_curve.png", "Accuracy"), ("loss", "loss_curve.png", "Loss")):
        columns = min(2, len(runs))
        rows = math.ceil(len(runs) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(12, 4.5 * rows), squeeze=False)
        flat_axes = axes.ravel()
        for ax, run in zip(flat_axes, runs):
            epochs = list(range(1, len(run["history"]) + 1))
            ax.plot(epochs, [row[f"train_{metric}"] for row in run["history"]], marker="o", label="Training")
            ax.plot(epochs, [row[f"val_{metric}"] for row in run["history"]], marker="o", label="Validation")
            ax.set(title=run["model"], xlabel="Epoch", ylabel=ylabel)
            ax.legend()
        for ax in flat_axes[len(runs):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(output / filename, dpi=200, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe PneumoAI model comparison.")
    parser.add_argument("--dataset", default="pneumonia_dataset/chest_xray")
    parser.add_argument("--manifest", default="results/split_manifest.csv")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--frozen-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true", help="Reuse completed per-model checkpoints when present.")
    parser.add_argument("--finetune-reused", action="store_true", help="Fine-tune an existing transfer-model checkpoint.")
    parser.add_argument("--comparison-output", default="results/model_comparison.csv")
    args = parser.parse_args()
    config = ExperimentConfig(dataset_dir=args.dataset, image_size=args.image_size, batch_size=args.batch_size,
                              frozen_epochs=args.frozen_epochs, finetune_epochs=args.finetune_epochs,
                              models=tuple(args.models), pretrained=not args.no_pretrained)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, counts = make_loaders(config.dataset_dir, config.image_size, config.batch_size, config.workers, args.manifest)
    runs = [train_model(name, config, loaders, counts, device, args.reuse_checkpoints, args.finetune_reused) for name in args.models]
    winner = max(runs, key=lambda item: item["val_roc_auc"])
    temperature = fit_temperature(winner["logits"], winner["labels"])
    calibrated = 1 / (1 + np.exp(-(winner["logits"] / temperature)))
    threshold, threshold_rows = optimize_threshold(winner["labels"], calibrated, config.min_sensitivity)
    val_metrics = calculate_metrics(winner["labels"], calibrated, threshold)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    checkpoint = torch.load(winner["checkpoint"], map_location="cpu", weights_only=True)
    torch.save(checkpoint, models_dir / "best_pneumonia_model.pth")
    metadata = {
        "classes": {"0": "NORMAL", "1": "PNEUMONIA"}, "positive_class": "PNEUMONIA",
        "image_size": [config.image_size, config.image_size], "preprocessing": "grayscale-to-RGB, resize, ImageNet normalization",
        "model": winner["model"], "threshold": threshold, "temperature": temperature,
        "uncertainty_margin": config.uncertainty_margin, "validation_metrics": val_metrics,
        "dataset_sizes": {split: len(loaders[split].dataset) for split in ("train", "val", "test")},
        "test_metrics": None, "gradcam_enabled": True, "seed": config.seed,
        "warning": "Research/education prototype; output is not a medical diagnosis.",
    }
    write_json(models_dir / "model_metadata.json", metadata)
    write_json(Path(config.output_dir) / "experiment_config.json", {**config.to_dict(), "python": platform.python_version(), "torch": torch.__version__, "device": str(device)})
    write_json(Path(config.output_dir) / "training_history.json", {run["model"]: run["history"] for run in runs})
    write_json(Path(config.output_dir) / "threshold_analysis.json", {"selected": threshold, "rows": threshold_rows})
    comparison_path = Path(args.comparison_output)
    comparison_path.parent.mkdir(exist_ok=True)
    save_training_curves(runs, Path("results"))
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "val_loss", "accuracy", "precision", "recall_sensitivity", "specificity", "f1", "val_roc_auc", "duration_seconds"])
        writer.writeheader()
        for run in runs:
            metrics = run["val_metrics_05"]
            writer.writerow({"model": run["model"], "val_loss": run["val_loss"], "accuracy": metrics["accuracy"],
                             "precision": metrics["precision"], "recall_sensitivity": metrics["recall_sensitivity"],
                             "specificity": metrics["specificity"], "f1": metrics["f1"],
                             "val_roc_auc": run["val_roc_auc"], "duration_seconds": run["duration_seconds"]})
    print(f"Selected {winner['model']} from validation ROC-AUC. Test set has not been evaluated.")


if __name__ == "__main__":
    main()
