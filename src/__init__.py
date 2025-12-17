"""
Jointwise Model Development Package

A modular YOLO-based object detection pipeline with:
- Dataset preparation and augmentation
- Custom FPN backbones (Xception, ResNeXt, DenseNet, EfficientNet)
- GA-based hyperparameter tuning
- Stacking ensemble with meta-learner
- Comprehensive evaluation metrics

Modules:
    config: Centralized configuration constants
    models: Custom FPN backbone definitions
    utils: Geometry utilities and I/O helpers
    training: Training logic and GA tuning
    stacking: Ensemble meta-learner and fusion
    evaluation: Metrics computation and FROC analysis
    preparation: Dataset conversion to YOLO format
    augmentation: Data augmentation utilities
"""

__version__ = "1.0.0"
