from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

from training.dataset import CLASS_TO_INDEX, SUPPORTED_SUFFIXES, discover_records
from training.utils import write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32).ravel()
    return (values - values.mean()) / (values.std() + 1e-6)


def audit_dataset(dataset_dir: str | Path, output_dir: str | Path) -> dict:
    root, output = Path(dataset_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = discover_records(root)
    dimensions: list[tuple[int, int]] = []
    modes: Counter[str] = Counter()
    hashes: dict[str, list[dict]] = defaultdict(list)
    perceptual_hashes: dict[str, list[dict]] = defaultdict(list)
    corrupted: list[str] = []
    extensions: Counter[str] = Counter()

    for record in records:
        extensions[record.path.suffix.lower()] += 1
        try:
            with Image.open(record.path) as image:
                image.verify()
            with Image.open(record.path) as image:
                dimensions.append(image.size)
                modes[image.mode] += 1
                gray = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
                difference_hash = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
                perceptual_hashes[difference_hash].append({"path": str(record.path), "split": record.split, "class": record.class_name})
            hashes[sha256(record.path)].append({"path": str(record.path), "split": record.split, "class": record.class_name})
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            corrupted.append(f"{record.path}: {exc}")

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    cross_split_duplicates = [items for items in duplicate_groups if len({item["split"] for item in items}) > 1]
    cross_class_duplicates = [items for items in duplicate_groups if len({item["class"] for item in items}) > 1]
    perceptual_groups = [items for items in perceptual_hashes.values() if len(items) > 1]
    perceptual_cross_split = [items for items in perceptual_groups if len({item["split"] for item in items}) > 1]
    high_similarity_pairs = []
    for group in perceptual_cross_split:
        for first_index, first in enumerate(group):
            for second in group[first_index + 1:]:
                if first["split"] == second["split"]:
                    continue
                correlation = float(np.mean(normalized_pixels(Path(first["path"])) * normalized_pixels(Path(second["path"]))))
                if correlation >= 0.99:
                    high_similarity_pairs.append({"first": first, "second": second, "correlation": correlation})
    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_classes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = f"{record.class_name}:{record.patient_id}"
        patient_splits[key].add(record.split)
        patient_classes[record.patient_id].add(record.class_name)
    patient_leakage = {key: sorted(value) for key, value in patient_splits.items() if len(value) > 1}

    split_counts = {
        split: {name: sum(r.split == split and r.class_name == name for r in records) for name in CLASS_TO_INDEX}
        for split in ("train", "val", "test")
    }
    widths = [item[0] for item in dimensions]
    heights = [item[1] for item in dimensions]
    total_by_class = {name: sum(r.class_name == name for r in records) for name in CLASS_TO_INDEX}
    summary = {
        "dataset_path": str(root.resolve()),
        "total_images": len(records),
        "classes": CLASS_TO_INDEX,
        "counts_by_class": total_by_class,
        "counts_by_split": split_counts,
        "supported_formats": sorted(SUPPORTED_SUFFIXES),
        "observed_formats": dict(extensions),
        "image_dimensions": {
            "min": [min(widths), min(heights)] if dimensions else None,
            "max": [max(widths), max(heights)] if dimensions else None,
            "unique_sizes": len(set(dimensions)),
        },
        "image_modes": dict(modes),
        "corrupted_count": len(corrupted),
        "corrupted_files": corrupted,
        "exact_duplicate_groups": len(duplicate_groups),
        "duplicate_extra_files": sum(len(group) - 1 for group in duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_duplicates),
        "cross_class_duplicate_groups": len(cross_class_duplicates),
        "possible_perceptual_duplicate_groups": len(perceptual_groups),
        "possible_perceptual_cross_split_groups": len(perceptual_cross_split),
        "possible_perceptual_cross_split_details": perceptual_cross_split,
        "high_similarity_cross_split_pairs": high_similarity_pairs,
        "high_similarity_threshold": 0.99,
        "patient_level_cross_split_groups": len(patient_leakage),
        "patient_level_cross_split_details": patient_leakage,
        "cross_split_duplicate_details": cross_split_duplicates,
        "class_balance": {name: round(count / len(records), 6) for name, count in total_by_class.items()},
        "split_policy": "Official folders preserved. Any reported patient/hash overlap must be reviewed before training.",
    }
    write_json(output / "dataset_summary.json", summary)
    _save_samples(records, output / "sample_images.png")
    _print_summary(summary)
    return summary


def _save_samples(records, path: Path, samples_per_class: int = 4) -> None:
    fig, axes = plt.subplots(2, samples_per_class, figsize=(13, 6))
    for row, class_name in enumerate(CLASS_TO_INDEX):
        items = [r for r in records if r.class_name == class_name]
        step = max(1, len(items) // samples_per_class)
        for col, record in enumerate(items[::step][:samples_per_class]):
            with Image.open(record.path) as image:
                axes[row, col].imshow(image.convert("L"), cmap="gray")
            axes[row, col].set_title(f"{class_name} · {record.split}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle("Dataset samples (deterministic selection)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _print_summary(summary: dict) -> None:
    print("Dataset Summary\n" + "-" * 46)
    print(f"Total Images: {summary['total_images']}")
    for name, count in summary["counts_by_class"].items():
        print(f"{name}: {count}")
    print(f"Splits: {json.dumps(summary['counts_by_split'])}")
    print(f"Image Size Range: {summary['image_dimensions']['min']} to {summary['image_dimensions']['max']}")
    print(f"Modes: {summary['image_modes']}")
    print(f"Corrupted Images: {summary['corrupted_count']}")
    print(f"Exact Duplicate Groups: {summary['exact_duplicate_groups']}")
    print(f"Cross-Split Duplicate Groups: {summary['cross_split_duplicate_groups']}")
    print(f"Possible Perceptual Cross-Split Groups: {summary['possible_perceptual_cross_split_groups']}")
    print(f"High-Similarity Cross-Split Pairs (correlation >= 0.99): {len(summary['high_similarity_cross_split_pairs'])}")
    print(f"Patient IDs Across Splits: {summary['patient_level_cross_split_groups']}")
    print(f"Class Balance: {summary['class_balance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit chest X-ray data for quality and leakage.")
    parser.add_argument("--dataset", default="pneumonia_dataset/chest_xray")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    audit_dataset(args.dataset, args.output)
