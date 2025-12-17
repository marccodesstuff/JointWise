"""
Configuration constants and project paths.

All tunable parameters and directory paths are centralized here
for easy modification and consistency across modules.
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
ROOT_DIR = Path.cwd()
PROJECT_NAME = "classic_train_stack"

# Data directories
IMAGES_DIR = Path("datasets/yolo/images").expanduser().resolve()
LABELS_DIR = Path("datasets/yolo/labels").expanduser().resolve()

# Output directories
TMP_DIR = ROOT_DIR / ".tmp_classic"
MODELS_DIR = TMP_DIR / "models"
PROJECT_DIR = ROOT_DIR / "runs" / PROJECT_NAME
DATA_YAML = TMP_DIR / "dataset.yaml"

# Model artifacts
META_MODEL_PATH = PROJECT_DIR / "meta_stack.pkl"
STACK_JSON_DIR = PROJECT_DIR / "stacked_test_json"
GA_CACHE_JSON = PROJECT_DIR / "ga_best_hparams.json"

# Evaluation defaults
STACK_JSON_DIR_DEFAULT = ROOT_DIR / "runs" / "classic_train_stack" / "stacked_test_json"
LABELS_TEST_DIR = ROOT_DIR / "datasets" / "yolo" / "labels" / "test"


# =============================================================================
# Model Configuration
# =============================================================================
NUM_CLASSES = 2
CLASS_NAMES = ["ACL Tear", "Meniscus Tear"]

# Image and training settings
IMG_SIZE = 320
BATCH = 16
FINAL_EPOCHS = 50
FINAL_PATIENCE = 10


# =============================================================================
# Genetic Algorithm (GA) Configuration
# =============================================================================
GA_ENABLE = True
GA_EVAL_EPOCHS = 3
GA_POP = 10
GA_GEN = 6
GA_SEED = 1337
GA_TUNE_ON_VAL = False


# =============================================================================
# Stacking / Meta-learner Configuration
# =============================================================================
GROUP_IOU = 0.45          # IoU threshold for clustering boxes across models
IOU_MATCH = 0.65          # IoU threshold to label meta group as positive
META_TOLERANCE_PX = 16.0  # Absolute pixel tolerance for center-distance fallback
META_TOLERANCE_REL = 0.25 # Relative tolerance (fraction of GT max dimension)
NMS_IOU = 0.5             # Final NMS threshold per class

# Confidence filtering
HIGH_PRECISION_CONF_THR = 0.2
TARGET_CLASS_PRECISION = 0.7

# Feature dimensions
STACK_STAT_FEATURES = 14  # model_count, sum, mean, max, min, std, median, top2_gap, max_pair_iou, area, width, height, diag, aspect
STACK_CLASS_FEATURES = NUM_CLASSES
STACK_EXTRA_FEATURES = STACK_STAT_FEATURES + STACK_CLASS_FEATURES


# =============================================================================
# Base Augmentation Settings
# =============================================================================
BASE_AUG = dict(
    mosaic=0.2,
    mixup=0.05,
    fliplr=0.2,
    hsv_h=0.003,
    hsv_s=0.15,
    hsv_v=0.15,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    shear=2.5,
    perspective=0.0005,
    copy_paste=0.05,
    multi_scale=False,
)


# =============================================================================
# GA Hyperparameter Search Space
# =============================================================================
HP_SPACE = {
    "optimizer": ["AdamW", "SGD"],
    "lr0": [1e-3, 2.5e-3, 5e-3],
    "lrf": [0.1, 0.2, 0.4],
    "momentum": [0.90, 0.93, 0.95],
    "weight_decay": [5e-5, 1e-4, 5e-4],
    "mosaic": [0.0, 0.1, 0.2, 0.3],
    "mixup": [0.0, 0.05, 0.10, 0.15],
    "fliplr": [0.1, 0.2, 0.3],
    "hsv_s": [0.10, 0.15, 0.20],
    "hsv_v": [0.10, 0.15, 0.20],
    "degrees": [0.0, 5.0, 10.0],
    "scale": [0.1, 0.2, 0.3],
    "translate": [0.05, 0.10],
    "shear": [2.5, 5.0],
    "copy_paste": [0.0, 0.05, 0.10],
}

HP_KEYS = list(HP_SPACE.keys())
HP_SIZES = [len(HP_SPACE[k]) for k in HP_KEYS]


# =============================================================================
# Directory Initialization
# =============================================================================
def ensure_directories():
    """Create necessary directories if they don't exist."""
    for p in (TMP_DIR, MODELS_DIR, PROJECT_DIR, STACK_JSON_DIR):
        p.mkdir(parents=True, exist_ok=True)
