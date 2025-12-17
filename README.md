# Jointwise Model Development

A modular YOLO-based object detection pipeline for knee injury detection (ACL tears and meniscus tears) with ensemble learning capabilities.

## Features

- **Custom FPN Backbones**: Xception, ResNeXt, DenseNet, and EfficientNet architectures
- **Genetic Algorithm Tuning**: Automated hyperparameter optimization using DEAP
- **Ensemble Stacking**: Meta-learner fusion of multiple detector predictions
- **Comprehensive Evaluation**: AP, F1, precision/recall, and FROC analysis
- **Data Augmentation**: Automated class balancing with geometric transforms

## Project Structure

```
jointwise-model-development/
├── main.py                 # Unified CLI entry point
├── src/                    # Core modules
│   ├── __init__.py
│   ├── config.py           # Configuration constants
│   ├── models.py           # FPN backbone definitions
│   ├── utils.py            # Geometry & I/O utilities
│   ├── training.py         # Training & GA tuning
│   ├── stacking.py         # Ensemble meta-learner
│   ├── evaluation.py       # Metrics computation
│   ├── preparation.py      # CSV → YOLO conversion
│   └── augmentation.py     # Data augmentation
├── data/                   # Source data (CSV + PNG images)
├── datasets/yolo/          # Prepared YOLO format dataset
└── runs/                   # Training outputs & predictions
```

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended for training)

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python version and dependency management.

### 1. Install uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
git clone <repository-url>
cd jointwise-model-development

# Create virtual environment and install dependencies
uv sync

# Or install from requirements.txt
uv pip install -r requirements.txt
```

### 3. Verify Installation

```bash
uv run python -c "import ultralytics; import timm; print('Setup complete!')"
```

## Usage

The project provides a unified CLI with four subcommands:

### Prepare Dataset

Convert CSV annotations to YOLO format with subject-level stratified splitting:

```bash
# Default: uses data/knee.csv → datasets/yolo/
uv run python main.py prepare

# Custom paths
uv run python main.py prepare --csv data/annotations.csv --png-dir data/images --output datasets/yolo

# Force rebuild existing dataset
uv run python main.py prepare --force

# Adjust train/val/test split ratio (default: 50/25/25)
uv run python main.py prepare --train-frac 0.7
```

### Augment Training Data

Balance class distribution through data augmentation:

```bash
# Preview augmentation plan (dry run)
uv run python main.py augment --dry-run

# Augment to reach 10,000 images per class
uv run python main.py augment --target 10000

# Augment validation set instead
uv run python main.py augment --split val --target 5000
```

### Train Ensemble Models

Train all model families with optional GA hyperparameter tuning:

```bash
# Full training pipeline (includes GA tuning)
uv run python main.py train

# Skip GA tuning, use cached hyperparameters
uv run python main.py train --skip-ga

# Disable GA entirely (use defaults)
uv run python main.py train --ga-disable

# Reuse existing model checkpoints (inference only)
uv run python main.py train --reuse-models

# Reuse both models and meta-learner
uv run python main.py train --reuse-models --reuse-meta

# Evaluate confidence thresholds on validation
uv run python main.py train --eval-thresholds
```

### Evaluate Predictions

Compute metrics on test set predictions:

```bash
# Basic evaluation
uv run python main.py evaluate

# Custom IoU threshold
uv run python main.py evaluate --iou 0.3

# With FROC analysis
uv run python main.py evaluate --froc --froc-save results/froc.csv

# Save results to files
uv run python main.py evaluate --save-csv results/metrics.csv --save-json results/metrics.json

# Multi-IoU mAP (COCO-style)
uv run python main.py evaluate --map-ious 0.5,0.55,0.6,0.65,0.7,0.75

# Filter by confidence
uv run python main.py evaluate --min-conf 0.25

# Adaptive IoU based on object size
uv run python main.py evaluate --adaptive-iou
```

## Pipeline Overview

1. **Prepare**: Converts source annotations to YOLO format with proper train/val/test splits
2. **Augment**: Applies geometric transforms to balance underrepresented classes
3. **Train**: Trains 4 detector families, optionally tunes hyperparameters with GA, builds stacking meta-learner
4. **Evaluate**: Computes AP, F1, FROC curves on test predictions

## Configuration

Key parameters can be modified in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 2 | Number of detection classes |
| `IMG_SIZE` | 320 | Input image size |
| `BATCH` | 16 | Training batch size |
| `FINAL_EPOCHS` | 50 | Training epochs |
| `GA_ENABLE` | True | Enable GA hyperparameter tuning |
| `GA_GENERATIONS` | 10 | GA generations |
| `GA_POP_SIZE` | 6 | GA population size |

## Model Families

| Family | Backbone | FPN Features |
|--------|----------|--------------|
| Xception | Xception41 | 728, 1024, 2048 |
| DenseNet | DenseNet121 | 256, 512, 1024 |
| ResNeXt | ResNeXt50 | 512, 1024, 2048 |
| EfficientNet | EfficientNet-B0 | 40, 112, 320 |

## Output Structure

After training, outputs are saved to `runs/classic_train_stack/`:

```
runs/classic_train_stack/
├── xception_final/weights/best.pt    # Trained model weights
├── densenet_final/weights/best.pt
├── resnext_final/weights/best.pt
├── efficientnet_final/weights/best.pt
├── meta_stack.pkl                    # Stacking meta-learner
├── meta_class_thresholds.json        # Per-class confidence thresholds
├── ga_best_hparams.json              # Cached GA hyperparameters
└── stacked_test_json/                # Test predictions (JSON per image)
```

## License

[Add your license here]

## Citation

[Add citation information if applicable]
