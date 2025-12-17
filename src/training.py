"""
Training utilities including GA hyperparameter tuning.

This module handles:
- Model training with Ultralytics YOLO
- Genetic Algorithm (DEAP) hyperparameter optimization
- Training argument management
- Checkpoint and hyperparameter caching
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    raise ImportError("deap is required. Install with: pip install deap")

from ultralytics import YOLO

from .config import (
    IMG_SIZE, BATCH, FINAL_EPOCHS, FINAL_PATIENCE,
    GA_ENABLE, GA_EVAL_EPOCHS, GA_POP, GA_GEN, GA_SEED, GA_TUNE_ON_VAL,
    HP_SPACE, HP_KEYS, HP_SIZES, BASE_AUG,
    PROJECT_DIR, TMP_DIR, MODELS_DIR, DATA_YAML, GA_CACHE_JSON,
    IMAGES_DIR, LABELS_DIR, NUM_CLASSES,
)
from .utils import (
    get_device, list_images, simple_yaml_load,
    create_dataset_yaml, create_val_only_dataset_yaml,
)


# =============================================================================
# Training Argument Helpers
# =============================================================================
def merge_train_args(hp: Dict) -> Tuple[Dict, Dict]:
    """
    Split hyperparameters into trainer and augmentation arguments.
    
    Args:
        hp: Combined hyperparameters dict
    
    Returns:
        (trainer_args, augmentation_args) tuple
    """
    aug_keys = set(BASE_AUG.keys())
    trainer, aug = {}, BASE_AUG.copy()
    
    for k, v in (hp or {}).items():
        if k in aug_keys:
            aug[k] = v
        else:
            trainer[k] = v
    
    return trainer, aug


def get_trainer_defaults(overrides: Dict, *, patience: int) -> Dict:
    """
    Get default training arguments with optional overrides.
    
    Args:
        overrides: Dictionary of parameters to override
        patience: Early stopping patience
    
    Returns:
        Combined training arguments
    """
    base = {
        "optimizer": "AdamW",
        "lr0": 2.5e-3,
        "lrf": 0.1,
        "momentum": 0.93,
        "weight_decay": 5e-4,
        "patience": patience,
        "seed": 0,
        "pretrained": True,
        "deterministic": False,
        "val": True,
    }
    return {**base, **(overrides or {})}


# =============================================================================
# Model Training
# =============================================================================
def train_model(model_yaml: Path, data_yaml: Path, epochs: int, patience: int,
                run_name: str, hp: Optional[Dict] = None) -> Tuple[YOLO, Path]:
    """
    Train a YOLO model with given configuration.
    
    Args:
        model_yaml: Path to model architecture YAML
        data_yaml: Path to dataset YAML
        epochs: Maximum training epochs
        patience: Early stopping patience
        run_name: Name for the training run
        hp: Optional hyperparameters
    
    Returns:
        (trained_model, run_directory) tuple
    """
    y = YOLO(str(model_yaml))
    t_over, aug_over = merge_train_args(hp or {})
    kwargs = get_trainer_defaults(t_over, patience=patience)
    
    res = y.train(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        epochs=epochs,
        batch=BATCH,
        workers=min(8, os.cpu_count() or 8),
        device=get_device(),
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=True,
        plots=False,
        **aug_over,
        **kwargs
    )
    
    return y, Path(res.save_dir)


def evaluate_map(model: YOLO, split: str = "val", 
                 data_yaml: Optional[Path] = None) -> Dict[str, float]:
    """
    Evaluate model and return mAP metrics.
    
    Args:
        model: Trained YOLO model
        split: Dataset split to evaluate on
        data_yaml: Path to dataset YAML
    
    Returns:
        Dict with map50-95, map50, map75 values
    """
    r = model.val(
        data=str(data_yaml or DATA_YAML),
        split=split,
        imgsz=IMG_SIZE,
        device=get_device(),
        plots=False,
        save_json=False,
        verbose=False
    )
    
    out = {}
    try:
        out = {
            "map50-95": float(r.box.map),
            "map50": float(r.box.map50),
            "map75": float(r.box.map75),
        }
    except Exception:
        pass
    
    return out


# =============================================================================
# GA Cache Management
# =============================================================================
def read_ga_cache() -> Dict[str, Dict[str, object]]:
    """Load cached GA hyperparameters from JSON."""
    if not GA_CACHE_JSON.exists():
        return {}
    try:
        return json.loads(GA_CACHE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_ga_cache(cache: Dict[str, Dict[str, object]]) -> None:
    """Save GA hyperparameters to JSON cache."""
    try:
        GA_CACHE_JSON.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"[GA][cache] Updated {GA_CACHE_JSON}")
    except Exception as e:
        print(f"[GA][cache][WARN] Failed to write cache: {e}")


def extract_hp_from_args_yaml(args_kv: Dict[str, object]) -> Dict[str, object]:
    """
    Extract relevant hyperparameters from Ultralytics args.yaml.
    
    Args:
        args_kv: Parsed args.yaml content
    
    Returns:
        Dict with hyperparameters matching our HP_SPACE
    """
    hp_keys = set(HP_SPACE.keys())
    out: Dict[str, object] = {}
    
    for k in hp_keys:
        if k in args_kv:
            out[k] = args_kv[k]
    
    # Also include augmentation keys
    aug_keys = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "shear", "perspective", "fliplr", "mosaic", "mixup", "copy_paste"]
    for k in aug_keys:
        if k in args_kv and k not in out:
            out[k] = args_kv[k]
    
    return out


def load_previous_ga_from_runs(family: str) -> Optional[Dict[str, object]]:
    """
    Try to load hyperparameters from previous training runs.
    
    Args:
        family: Model family name (e.g., 'xception')
    
    Returns:
        Hyperparameters dict or None if not found
    """
    candidates = [
        PROJECT_DIR / f"{family}_final" / "args.yaml",
        PROJECT_DIR / f"{family}_GAeval" / "args.yaml",
    ]
    
    for p in candidates:
        if p.exists():
            kv = simple_yaml_load(p)
            hp = extract_hp_from_args_yaml(kv)
            if hp:
                print(f"[GA][reuse] Loaded previous HPs for {family} from {p}")
                return hp
    
    return None


# =============================================================================
# GA Hyperparameter Conversion
# =============================================================================
def gene_to_hp(gene: List[int]) -> Dict:
    """
    Convert GA gene (list of indices) to hyperparameter dict.
    
    Args:
        gene: List of indices into HP_SPACE options
    
    Returns:
        Hyperparameters dict
    """
    hp = {}
    for idx, k in enumerate(HP_KEYS):
        hp[k] = HP_SPACE[k][gene[idx] % len(HP_SPACE[k])]
    return hp


# =============================================================================
# Genetic Algorithm
# =============================================================================
def run_ga_for_family(family: str, yml_path: Path, *,
                      force_run: bool = False,
                      prefer_reuse: bool = True,
                      predict_boxes_fn=None) -> Dict:
    """
    Run GA hyperparameter search for a model family.
    
    Args:
        family: Model family name
        yml_path: Path to model YAML
        force_run: Force GA even if cached values exist
        prefer_reuse: Try to reuse cached/previous hyperparameters
        predict_boxes_fn: Optional function for prediction diagnostics
    
    Returns:
        Best hyperparameters dict
    """
    # Try reuse first unless forced
    if prefer_reuse and not force_run:
        cache = read_ga_cache()
        if family in cache and cache[family]:
            print(f"[GA][reuse] Using cached best HPs for {family} from {GA_CACHE_JSON}")
            return cache[family]
        
        prev = load_previous_ga_from_runs(family)
        if prev:
            print(f"[GA][reuse] Using HPs from previous runs for {family}")
            return prev

    if not GA_ENABLE and not force_run:
        print(f"[GA] Disabled, using defaults (and BASE_AUG) for {family}.")
        return {}

    # Set seeds for reproducibility
    random.seed(GA_SEED)
    np.random.seed(GA_SEED)

    # Choose dataset YAML for GA
    ga_data_yaml = DATA_YAML
    if GA_TUNE_ON_VAL:
        ga_yaml_path = TMP_DIR / f"ga_val_{family}.yaml"
        create_val_only_dataset_yaml(IMAGES_DIR, LABELS_DIR, ga_yaml_path, nc=NUM_CLASSES)
        ga_data_yaml = ga_yaml_path
        print(f"[GA] {family}: training on validation split for HP search -> {ga_data_yaml}")

    # Create DEAP structures (check if already created)
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Register gene generators
    for i, size in enumerate(HP_SIZES):
        toolbox.register(f"gene_{i}", random.randrange, size)

    def make_individual():
        return creator.Individual([
            toolbox.__getattribute__(f"gene_{i}")() 
            for i in range(len(HP_SIZES))
        ])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        """Fitness function: train briefly and return mAP50."""
        hp = gene_to_hp(individual)
        run_name = f"{family}_GAeval"
        
        y, _ = train_model(
            yml_path, ga_data_yaml,
            epochs=GA_EVAL_EPOCHS,
            patience=max(1, GA_EVAL_EPOCHS - 1),
            run_name=run_name,
            hp=hp
        )
        
        metrics = evaluate_map(y, split="val", data_yaml=ga_data_yaml)
        
        # Optional diagnostics
        if predict_boxes_fn is not None:
            try:
                val_dir = IMAGES_DIR / "val"
                sample_imgs = list_images(val_dir)[:24]
                pred_map = predict_boxes_fn(y, sample_imgs, conf=0.001, iou=0.5)
                total_preds = sum(len(v) for v in pred_map.values())
                per_img = total_preds / max(1, len(sample_imgs)) if sample_imgs else 0.0
                hist = {c: 0 for c in range(NUM_CLASSES)}
                for vs in pred_map.values():
                    for cl, _, _ in vs:
                        if cl in hist:
                            hist[cl] += 1
                print(f"[GA][{family}] preds/img={per_img:.2f} total={total_preds} "
                      f"class_hist={hist} map50={metrics.get('map50', 0.0):.4f}")
            except Exception:
                pass
        
        score = float(metrics.get("map50", 0.0))
        return (score,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, 
                     low=[0] * len(HP_SIZES),
                     up=[s - 1 for s in HP_SIZES], 
                     indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run evolution
    pop = toolbox.population(n=GA_POP)
    hof = tools.HallOfFame(1)
    
    algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.5, mutpb=0.3, ngen=GA_GEN,
        halloffame=hof, verbose=True
    )

    # Extract best
    best_gene = list(hof[0])
    best_hp = gene_to_hp(best_gene)
    print(f"[GA] {family} best hp: {best_hp}")
    
    # Persist to cache
    cache = read_ga_cache()
    cache[family] = best_hp
    write_ga_cache(cache)
    
    return best_hp


# =============================================================================
# Family Training Pipeline
# =============================================================================
def write_yaml_file(name: str, text: str) -> Path:
    """Write model YAML text to a file."""
    p = MODELS_DIR / f"{name}.yaml"
    p.write_text(text)
    return p


def train_family(family: str, yaml_text: str, *,
                 force_ga: bool = False, skip_ga: bool = False,
                 predict_boxes_fn=None) -> Tuple[str, Path, Dict]:
    """
    Complete training pipeline for a model family.
    
    Args:
        family: Model family name
        yaml_text: Model YAML configuration
        force_ga: Force running GA
        skip_ga: Skip GA and use cached/default values
        predict_boxes_fn: Optional prediction function for diagnostics
    
    Returns:
        (family_name, checkpoint_path, hyperparameters) tuple
    """
    yml = write_yaml_file(family, yaml_text)
    
    # GA tuning (with reuse)
    best_hp: Dict[str, object] = {}
    
    if skip_ga:
        # Try reuse only
        cache = read_ga_cache()
        best_hp = cache.get(family) or load_previous_ga_from_runs(family) or {}
        if best_hp:
            print(f"[GA][skip] Using reused HPs for {family}: {best_hp}")
        else:
            print(f"[GA][skip] No reused HPs found for {family}; using defaults.")
    else:
        best_hp = run_ga_for_family(
            family, yml,
            force_run=force_ga,
            prefer_reuse=True,
            predict_boxes_fn=predict_boxes_fn
        )
    
    # Final training with early stopping
    run_name = f"{family}_final"
    y, run_dir = train_model(
        yml, DATA_YAML,
        epochs=FINAL_EPOCHS,
        patience=FINAL_PATIENCE,
        run_name=run_name,
        hp=best_hp
    )
    
    # Get best checkpoint
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "weights" / "last.pt"
    
    print(f"[{family}] Final checkpoint -> {ckpt}")
    return family, ckpt, best_hp
