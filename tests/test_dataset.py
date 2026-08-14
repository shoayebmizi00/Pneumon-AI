from __future__ import annotations

from collections import defaultdict

import numpy as np

from training.dataset import ChestXrayDataset, build_transform, load_manifest
from training.metrics import calculate_metrics, optimize_threshold


def test_manifest_has_no_patient_leakage() -> None:
    records = load_manifest("results/split_manifest.csv")
    partitions = defaultdict(set)
    for record in records:
        partitions[(record.class_name, record.patient_id)].add(record.split)
    assert len(records) == 5824
    assert all(len(splits) == 1 for splits in partitions.values())


def test_preprocessing_produces_expected_tensor() -> None:
    record = load_manifest("results/split_manifest.csv")[0]
    tensor, label, path = ChestXrayDataset([record], build_transform(224, training=False))[0]
    assert tensor.shape == (3, 224, 224)
    assert float(label) in {0.0, 1.0}
    assert path.endswith((".jpeg", ".jpg", ".png"))


def test_threshold_is_optimized_from_validation_inputs() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, rows = optimize_threshold(labels, scores, min_sensitivity=1.0)
    metrics = calculate_metrics(labels, scores, threshold)
    assert rows
    assert metrics["recall_sensitivity"] == 1.0
    assert metrics["f1"] == 1.0
