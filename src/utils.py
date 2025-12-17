"""
Utility functions for geometry, I/O, and data handling.

This module provides common helper functions used across the pipeline:
- Bounding box geometry (IoU, coordinate conversions)
- YOLO label file I/O
- Dataset YAML generation
- Image listing utilities
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

from .config import CLASS_NAMES


# =============================================================================
# Device Utilities
# =============================================================================
def get_device() -> int | str:
    """Return the appropriate device for training/inference."""
    return 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# Bounding Box Geometry
# =============================================================================
def xywhn_to_xyxy(cx: float, cy: float, w: float, h: float, 
                  W: int, H: int) -> Tuple[float, float, float, float]:
    """
    Convert normalized YOLO format (center x, y, width, height) to pixel XYXY.
    
    Args:
        cx, cy: Normalized center coordinates (0-1)
        w, h: Normalized width and height (0-1)
        W, H: Image dimensions in pixels
    
    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return x1, y1, x2, y2


def xyxy_to_xywhn(x1: float, y1: float, x2: float, y2: float,
                  W: int, H: int) -> Tuple[float, float, float, float]:
    """
    Convert pixel XYXY format to normalized YOLO format.
    
    Args:
        x1, y1, x2, y2: Pixel coordinates
        W, H: Image dimensions in pixels
    
    Returns:
        Tuple of (cx, cy, w, h) normalized (0-1)
    """
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return cx, cy, w, h


def iou_xyxy(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """
    Calculate Intersection over Union for two XYXY boxes.
    
    Args:
        a, b: Boxes as (x1, y1, x2, y2)
    
    Returns:
        IoU value between 0 and 1
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    # Intersection
    xi1 = max(ax1, bx1)
    yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2)
    yi2 = min(ay2, by2)
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    
    # Union
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    
    return inter / union if union > 0 else 0.0


def center_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """
    Calculate Euclidean distance between box centers.
    
    Args:
        a, b: Boxes as (x1, y1, x2, y2)
    
    Returns:
        Distance in pixels
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    
    return math.hypot(acx - bcx, acy - bcy)


def expand_box(box: Tuple[float, ...], px: float) -> Tuple[float, float, float, float]:
    """
    Expand a box by a fixed number of pixels on each side.
    
    Args:
        box: (x1, y1, x2, y2) coordinates
        px: Number of pixels to expand
    
    Returns:
        Expanded box coordinates
    """
    x1, y1, x2, y2 = box
    return (x1 - px, y1 - px, x2 + px, y2 + px)


def box_area(box: Tuple[float, ...]) -> float:
    """Calculate the area of a box in XYXY format."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_dimensions(box: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    """
    Calculate box dimensions.
    
    Returns:
        (width, height, diagonal, aspect_ratio)
    """
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    diag = math.hypot(width, height)
    aspect = width / height if height > 1e-6 else 0.0
    return width, height, diag, aspect


# =============================================================================
# YOLO Label I/O
# =============================================================================
def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO format labels from a text file.
    
    Args:
        label_path: Path to label .txt file
    
    Returns:
        List of (class_id, cx, cy, w, h) tuples (normalized)
    """
    if not label_path.exists():
        return []
    
    out = []
    for ln in label_path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) >= 5:
            try:
                cl = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                out.append((cl, cx, cy, w, h))
            except ValueError:
                pass
    return out


def write_yolo_labels(label_path: Path, 
                      labels: List[Tuple[int, float, float, float, float]]) -> None:
    """
    Write YOLO format labels to a text file.
    
    Args:
        label_path: Output path
        labels: List of (class_id, cx, cy, w, h) tuples
    """
    lines = [f"{cl} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" 
             for cl, cx, cy, w, h in labels]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def yolo_label_path_for_image(img_path: Path, labels_root: Path) -> Path:
    """Get the label file path corresponding to an image."""
    return labels_root / (img_path.stem + ".txt")


# =============================================================================
# Dataset YAML Generation
# =============================================================================
def create_dataset_yaml(images_dir: Path, labels_dir: Path, 
                        out_yaml: Path, nc: int) -> Path:
    """
    Create a dataset YAML file for Ultralytics training.
    
    Args:
        images_dir: Path to images directory (containing train/val/test subdirs)
        labels_dir: Path to labels directory
        out_yaml: Output YAML path
        nc: Number of classes
    
    Returns:
        Path to created YAML file
    """
    # Validate directories exist
    for split in ("train", "val", "test"):
        if not (images_dir / split).exists():
            raise FileNotFoundError(f"Missing images/{split}")
        if not (labels_dir / split).exists():
            raise FileNotFoundError(f"Missing labels/{split}")
    
    root_posix = images_dir.parent.as_posix()
    data = f"""
path: "{root_posix}"
train: images/train
val: images/val
test: images/test
names: {json.dumps(CLASS_NAMES)}
nc: {nc}
""".strip()
    
    out_yaml.write_text(data + "\n")
    return out_yaml


def create_val_only_dataset_yaml(images_dir: Path, labels_dir: Path,
                                  out_yaml: Path, nc: int) -> Path:
    """
    Create a dataset YAML where both train and val point to validation split.
    
    Useful for GA evaluation when HP search should use only validation data.
    """
    for split in ("val", "test"):
        if not (images_dir / split).exists():
            raise FileNotFoundError(f"Missing images/{split}")
    
    root_posix = images_dir.parent.as_posix()
    data = f"""
path: "{root_posix}"
train: images/val
val: images/val
test: images/test
names: {json.dumps(CLASS_NAMES)}
nc: {nc}
""".strip()
    
    out_yaml.write_text(data + "\n")
    return out_yaml


# =============================================================================
# File Listing Utilities
# =============================================================================
def list_images(dir_path: Path) -> List[str]:
    """
    List all image files in a directory.
    
    Args:
        dir_path: Directory to scan
    
    Returns:
        Sorted list of image paths as strings
    """
    imgs = sorted([
        *dir_path.glob("*.png"),
        *dir_path.glob("*.jpg"),
        *dir_path.glob("*.jpeg")
    ])
    return [str(p) for p in imgs]


def get_image_size(img_path: str) -> Tuple[int, int]:
    """
    Get image dimensions without loading the full image.
    
    Args:
        img_path: Path to image file
    
    Returns:
        (width, height) tuple
    """
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except Exception:
        # Fallback for missing images
        return (1, 1)


# =============================================================================
# Cache Management
# =============================================================================
def clear_dataset_caches(labels_root: Path) -> None:
    """Remove Ultralytics dataset cache files."""
    for split in ("train", "val", "test"):
        cache_file = labels_root / f"{split}.cache"
        if cache_file.exists():
            try:
                cache_file.unlink()
                print(f"[DATA] Removed cache: {cache_file}")
            except Exception:
                pass


# =============================================================================
# YAML Key-Value Parser
# =============================================================================
def simple_yaml_load(path: Path) -> Dict[str, object]:
    """
    Simple YAML key-value parser for flat configuration files.
    
    Handles basic types (strings, numbers, booleans, null).
    Not suitable for nested structures.
    """
    if not path.exists():
        return {}
    
    out: Dict[str, object] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        
        # Remove inline comments
        if " #" in v:
            v = v.split(" #", 1)[0].strip()
        
        # Remove quotes
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        
        # Parse value type
        low = v.lower()
        if low in {"null", "none"}:
            val: object = None
        elif low in {"true", "false"}:
            val = (low == "true")
        else:
            # Try numeric
            try:
                if any(ch in v for ch in [".", "e", "E"]):
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                val = v
        
        out[k] = val
    
    return out
