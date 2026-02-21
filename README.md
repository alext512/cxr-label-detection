# CXR MIL Training (Grand X-Ray SLAM)

PyTorch codebase for study-level multi-label chest X-ray classification with MIL
(multi-instance learning), patient-level CV splits, and configurable augmentations.

## Repository Structure
- `cxr_mil/`: model, data, metrics, and training code
- `cxr_mil/cli/train_cv.py`: CLI entrypoint for cross-validation training
- `configs/`: experiment YAML configs
- `run_project_in_jupyter.ipynb`: notebook runner

## Setup
```bash
pip install -e .
```

## Data Layout
Your dataset root is expected to contain:
- `<root>/train1.csv`
- `<root>/train1/` (images)

You can override names in YAML (`data.csv_name`, `data.img_subdir`).

## Train
With YAML config:
```bash
cxr-mil-train-cv --config configs/default.yaml --root /path/to/data
```

With direct args:
```bash
cxr-mil-train-cv --root /path/to/data --epochs 20
```

As a module:
```bash
python -m cxr_mil.train_cv --root /path/to/data
```

## Notebook
Use `run_project_in_jupyter.ipynb` for interactive runs. The notebook is already
clean (outputs removed) and suitable for GitHub.
