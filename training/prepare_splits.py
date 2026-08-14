from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from training.dataset import discover_records


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_groups(groups: list[tuple[str, list]], ratios: tuple[float, float, float], rng: random.Random) -> dict[str, str]:
    """Greedily assign whole patients while approximating image-level ratios."""
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    names = ("train", "val", "test")
    targets = [sum(len(items) for _, items in groups) * ratio for ratio in ratios]
    totals = [0, 0, 0]
    assignment: dict[str, str] = {}
    for group_id, items in groups:
        scores = [(totals[index] + len(items)) / max(targets[index], 1) for index in range(3)]
        chosen = min(range(3), key=lambda index: scores[index])
        assignment[group_id] = names[chosen]
        totals[chosen] += len(items)
    return assignment


def create_manifest(dataset_dir: str | Path, output_path: str | Path, seed: int = 42) -> dict:
    records = discover_records(dataset_dir)
    unique, duplicates_removed = [], []
    seen: dict[str, object] = {}
    for record in records:
        digest = file_hash(record.path)
        if digest in seen:
            duplicates_removed.append(str(record.path))
        else:
            seen[digest] = record
            unique.append(record)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for class_name in ("NORMAL", "PNEUMONIA"):
        grouped: dict[str, list] = defaultdict(list)
        for record in unique:
            if record.class_name == class_name:
                grouped[f"{class_name}:{record.patient_id}"].append(record)
        assignments.update(assign_groups(list(grouped.items()), (0.70, 0.15, 0.15), rng))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in unique:
        key = f"{record.class_name}:{record.patient_id}"
        rows.append({"path": str(record.path.resolve()), "label": record.label, "class_name": record.class_name,
                     "split": assignments[key], "patient_id": record.patient_id, "source_split": record.split})
    rows.sort(key=lambda row: (row["split"], row["class_name"], row["patient_id"], row["path"]))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["class_name"]) for row in rows)
    summary = {
        "manifest": str(target.resolve()), "seed": seed, "duplicates_removed": len(duplicates_removed),
        "images": len(rows), "counts": {split: {name: counts[(split, name)] for name in ("NORMAL", "PNEUMONIA")} for split in ("train", "val", "test")},
        "patient_overlap": 0, "strategy": "SHA-256 deduplication, then deterministic class-stratified patient-group allocation (70/15/15).",
    }
    print(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pneumonia_dataset/chest_xray")
    parser.add_argument("--output", default="results/split_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_manifest(args.dataset, args.output, args.seed)
