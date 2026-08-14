# PneumoAI — AI-Based Pneumonia Detection from Chest X-Rays

PneumoAI is an end-to-end university research project that audits a chest X-ray dataset, prevents patient-level leakage, compares convolutional neural networks, calibrates a selected classifier, evaluates it once on a held-out test set, serves predictions through FastAPI, and presents results in an accessible React interface.

> **Medical disclaimer:** This AI system is an educational/research prototype and does not provide a medical diagnosis. Predictions may be incorrect. Chest X-rays and medical symptoms should be reviewed by a qualified healthcare professional.

## Current dataset findings

The repository audit inspected every supplied image rather than relying on assumed dataset statistics.

| Finding | Actual value |
|---|---:|
| Total supplied images | 5,856 |
| NORMAL | 1,583 (27.03%) |
| PNEUMONIA | 4,273 (72.97%) |
| Original train / validation / test | 4,931 / 301 / 624 |
| Image size range | 384×127 to 2916×2713 |
| Grayscale / RGB | 5,573 / 283 |
| Corrupt images | 0 |
| Exact duplicate groups | 30 (32 extra files) |
| Exact duplicates across original splits | 0 |
| Coarse perceptual-hash groups across original splits | 118 |
| High-similarity pairs at ≥0.99 correlation | 0 |
| Patient groups crossing original splits | **172** |

The original folders therefore have patient-level leakage. The training pipeline does not use them directly. `training.prepare_splits` removes exact duplicates by SHA-256 and creates a fixed-seed, class-stratified, patient-grouped manifest:

| New partition | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,106 | 2,972 | 4,078 |
| Validation | 237 | 637 | 874 |
| Test | 236 | 636 | 872 |

Patient overlap in the new manifest is zero. The final test partition is not accessed by `training.train`; only `training.evaluate` loads it.

Generated evidence is stored in [results/dataset_summary.json](results/dataset_summary.json), [results/split_manifest.csv](results/split_manifest.csv), and [results/sample_images.png](results/sample_images.png).

## Objectives and features

- Classify an uploaded chest X-ray as `NORMAL` or `PNEUMONIA` pattern.
- Never phrase output as a diagnosis.
- Apply identical deterministic inference preprocessing to training validation/test inputs.
- Compare a baseline CNN, DenseNet121, EfficientNet-B0, ResNet50, and MobileNetV3.
- Use conservative training-only augmentation, class-weighted binary loss, early stopping, checkpoints, learning-rate reduction, and partial fine-tuning.
- Select by validation evidence, fit temperature scaling on validation logits, and select the decision threshold on validation data with a minimum-sensitivity constraint.
- Report accuracy, precision, sensitivity, specificity, F1, ROC–AUC, PR–AUC, Brier score, TN/FP/FN/TP, confusion matrix, ROC, precision-recall, calibration, and training curves.
- Generate Grad-CAM as an influence visualization, with explicit limits.
- Validate uploads, avoid disk persistence, rate limit inference, and return safe API errors.
- Warn on borderline scores and simple out-of-domain signals instead of visually forcing confidence.

## Model and preprocessing

Class order is fixed everywhere:

```text
0 = NORMAL
1 = PNEUMONIA
```

All images are converted safely to three identical grayscale-derived channels, resized to 224×224 (configurable), converted to tensors, then normalized with ImageNet mean and standard deviation for pretrained backbones. Training augmentation is limited to ±5° rotation, 2.5% translation, 0.95–1.05 scale, and small brightness/contrast changes. Validation, test, and inference receive no stochastic augmentation.

Class imbalance is handled with `BCEWithLogitsLoss(pos_weight=N_normal/N_pneumonia)` computed from the training partition. Models train in two stages: frozen backbone at `1e-3`, then the final backbone portion at `1e-5`. The baseline CNN trains end-to-end. AdamW, `ReduceLROnPlateau`, early stopping, and best-validation-loss checkpoints are used.

## Reliable evaluation policy

1. Audit all data and derive patient identifiers from filenames.
2. Remove exact duplicates globally.
3. Allocate entire patient groups to one fixed partition.
4. Train and compare models using only train and validation.
5. Fit temperature scaling and choose the operating threshold using validation only.
6. Save the selected architecture, weights, preprocessing, temperature, threshold, and uncertainty margin.
7. Run `training.evaluate` once. It refuses to overwrite an existing final report without `--force`.

The website loads these genuine final metrics from model metadata. If metadata is unavailable, it displays “Final evaluation pending” instead of sample numbers.

### Final selected experiment

DenseNet121 was selected after frozen-backbone comparison and one selective partial fine-tuning epoch at `1e-5`. Temperature scaling and the decision threshold (`0.19`) were fitted using validation data only. The frozen model and threshold were then evaluated on the 872-image untouched test partition.

| Test metric | Actual result |
|---|---:|
| Accuracy | 93.92% |
| Precision | 94.37% |
| Sensitivity / recall | 97.48% |
| Specificity | 84.32% |
| F1 score | 95.90% |
| ROC–AUC | 0.9811 |
| PR–AUC | 0.9898 |
| Brier score | 0.0549 |

Confusion matrix: TN 199, FP 37, FN 16, TP 620. These are research results on this dataset, not evidence of clinical effectiveness.

Validation comparison at the common 0.50 threshold (threshold-independent ROC–AUC was the primary selection signal):

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC–AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 91.30% | 98.11% | 89.80% | 95.36% | 93.77% | 0.9764 |
| **DenseNet121, fine-tuned** | **92.79%** | **98.98%** | 91.05% | **97.47%** | **94.85%** | **0.9839** |
| EfficientNet-B0 | 89.47% | 97.23% | 88.07% | 93.25% | 92.42% | 0.9681 |
| MobileNetV3 | 88.56% | 88.41% | **97.02%** | 65.82% | 92.51% | 0.9623 |

## Project structure

```text
backend/       FastAPI app, validation, model service
frontend/      React + Vite + Tailwind interface
training/      audit, split, datasets, models, training, metrics, evaluation, Grad-CAM
notebooks/     thin, reproducible notebook entry points
models/        selected checkpoint and metadata after training
results/       audit, split manifest, plots, metrics, comparisons
tests/         backend API contract and validation tests
design-system/ persisted UI/UX design guidance
```

## Setup

Python 3.12 and Node.js 22 are the recorded development versions.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm install
```

The frontend scripts invoke the Node entry points directly, which also works when a Windows parent directory contains `&`, as this workspace does.

## Dataset audit and leakage-safe split

From the repository root:

```powershell
python -m training.audit --dataset pneumonia_dataset/chest_xray --output results
python -m training.prepare_splits --dataset pneumonia_dataset/chest_xray --output results/split_manifest.csv --seed 42
```

## Training

Full comparison:

```powershell
python -m training.train --models baseline densenet121 efficientnet_b0 resnet50 mobilenet_v3
```

The default configuration uses 224×224 inputs, batch size 24, seed 42, 8 frozen epochs plus up to 8 fine-tuning epochs, and patience 3. ImageNet weights may download on first use. Checkpoints and histories are retained under `artifacts/`; comparison and training plots are written to `results/`.

After model selection and threshold calibration, evaluate the untouched test set exactly once:

```powershell
python -m training.evaluate
```

This produces `results/test_metrics.json`, confusion matrix, ROC curve, precision-recall curve, and reliability diagram, and adds the actual test metrics to `models/model_metadata.json`.

## Backend

```powershell
Copy-Item backend/.env.example backend/.env
uvicorn backend.main:app --reload --port 8000
```

Environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PATH` | PyTorch checkpoint | `models/best_pneumonia_model.pth` |
| `METADATA_PATH` | preprocessing/threshold metadata | `models/model_metadata.json` |
| `MAX_UPLOAD_SIZE` | byte limit | 10 MB |
| `ALLOWED_ORIGINS` | comma-separated CORS origins | `http://localhost:5173` |
| `REQUESTS_PER_MINUTE` | per-IP in-memory limit | 30 |

### API

`GET /api/health` reports `ready` only when the trained checkpoint is loaded.

`GET /api/performance` returns only metrics recorded in model metadata.

`POST /api/predict?gradcam=true` accepts multipart field `file` containing JPEG or PNG. A successful response contains `prediction`, calibrated `prediction_score`, separate `normal_score` and `pneumonia_score`, the recorded `decision_threshold`, an `uncertain` flag, an optional `domain_warning`, and an optional base64 PNG `gradcam_image`. Every numeric value is produced by the loaded model; the API contains no sample or hardcoded prediction values.

## Frontend

```powershell
cd frontend
Copy-Item .env.example .env
npm run dev
```

Set `VITE_API_URL` to the deployed backend origin. The interface supports drag/drop, keyboard selection, preview/removal, progress, timeout, retry, calibrated score bars, uncertainty, domain warnings, Grad-CAM, responsive navigation, visible focus, 44px targets, and reduced-motion preferences.

## Tests

```powershell
python -m pytest -q
cd frontend
npm test
npm run build
```

Backend tests cover health, missing/unsupported/corrupt files, and the successful response contract. Frontend tests cover the safety/upload surface and client-side file rejection. Additional model smoke cases should use known normal/pneumonia X-rays, RGB/grayscale variants, resized copies, corrupt files, and non-X-ray photographs; a domain warning is not a clinically validated OOD detector.

## Deployment

`docker-compose.yml`, `Dockerfile.backend`, and `frontend/Dockerfile` provide a container path. `frontend/vercel.json` and `render.yaml` provide Vercel SPA routing/security headers and a Render backend blueprint. Deploy the backend where the model fits in memory. Do not expose the API without HTTPS, strict origins, operational rate limiting, and an appropriate privacy review. Never log uploaded image bodies.

## Limitations

- This dataset is imbalanced and may not represent different hospitals, devices, ages, disease prevalences, or demographic groups.
- Filename-derived patient IDs are the strongest identifiers available; no DICOM metadata is present.
- Temperature scaling improves score calibration on the validation distribution but does not make a score a clinical probability.
- Grad-CAM indicates model influence, not a lesion, causal explanation, or diagnosis.
- The lightweight out-of-domain heuristic is only a warning and is not a validated chest-X-ray detector.
- Clinical use requires external validation, prospective evaluation, subgroup analysis, governance, and regulatory review.

## Future improvements

External multi-site validation, clinically reviewed labels, robust DICOM handling, stronger learned out-of-distribution detection, subgroup/fairness analysis, uncertainty ensembles, calibration monitoring, ONNX acceleration, and formal model/data cards.

## Author

University AI & ML Lab Project.
