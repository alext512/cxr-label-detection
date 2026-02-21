# CXR MIL Training (Grand X-Ray SLAM)

PyTorch codebase for study-level multi-label chest X-ray classification with MIL (multi-instance learning), patient-level CV splits, and configurable augmentations.

## Repository Structure
- `cxr_mil/`: model, data, metrics, and training code
- `cxr_mil/cli/train_cv.py`: CV training entrypoint
- `cxr_mil/cli/inspect_augmentations.py`: augmentation/transformation sampler
- `cxr_mil/cli/analyze_predictions.py`: OOF diagnostics (best/worst studies, per-label AUC)
- `configs/`: curated experiment YAML configs

## Setup
```bash
pip install -e .
```

## Data Layout
Expected dataset root:
- `<root>/train1.csv`
- `<root>/train1/` (images)

You can override with `data.csv_name` and `data.img_subdir` in YAML.

## Train
```bash
cxr-mil-train-cv --config configs/default.yaml --root /path/to/data
```

### Run hygiene outputs
Each run is now isolated under:

`runs/<run_name>/<seed>/fold_<k>/`

Saved artifacts include:
- resolved config dump (`resolved_config.json`)
- fold training log (`training_log.csv`, overwritten per run)
- best checkpoint (`best.pt`)
- per-fold OOF predictions (`oof_predictions.csv`)

Run-level artifacts:
- `cv_training_log.csv`
- `cv_summary.json`
- `oof_predictions_all_folds.csv` (if all folds were trained)
- `oof_metrics.json`

## Analyze predictions
```bash
cxr-mil-analyze-preds --oof runs/<run>/<seed>/oof_predictions_all_folds.csv --out-dir analysis_reports
```

Produces:
- per-label AUC table
- worst/best predicted studies by sample BCE
- summary metrics

## Inspect augmentations
```bash
cxr-mil-inspect-augs --config configs/exp_asl_cosine_ema.yaml --root /path/to/data --num-samples 8
```

Creates a grid image showing raw/train/val transformed samples.
