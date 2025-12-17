"""
Data augmentation utilities for YOLO datasets.

This module provides functions to augment training data by applying
various transforms (flips, rotations, perspective) while preserving
bounding box annotations.
"""

from __future__ import annotations

import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import cv2
import albumentations as A


# =============================================================================
# Configuration
# =============================================================================
DATASET_ROOT = Path("datasets/yolo")
DEFAULT_TARGET = 20000
DEFAULT_SEED = 42


# =============================================================================
# Label I/O
# =============================================================================
def load_label_lines(path: Path) -> List[str]:
    """Load non-empty lines from a label file."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_label_line(line: str) -> Tuple[int, Tuple[float, float, float, float]]:
    """Parse YOLO label line into (class_id, (x, y, w, h))."""
    parts = line.split()
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:5])
    return cls, (x, y, w, h)


def save_label(path: Path, items: List[Tuple[int, Tuple[float, float, float, float]]]) -> None:
    """Save labels to file in YOLO format."""
    lines = [
        f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        for cls, box in items
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# =============================================================================
# Augmentation Pipeline
# =============================================================================
def create_augmentation_pipeline(img_size: int = 320) -> A.Compose:
    """
    Create Albumentations augmentation pipeline.
    
    Args:
        img_size: Target image size for resize
        
    Returns:
        Configured Albumentations Compose pipeline
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=45, p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            ),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
            A.Affine(shear=(-10, 10), cval=(0, 0, 0), p=0.5),
            A.Resize(height=img_size, width=img_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,  # Drop boxes with <20% visibility after transform
        ),
    )


def apply_augmentation(
    img,
    boxes: List[Tuple[float, float, float, float]],
    classes: List[int],
    aug: A.Compose
) -> Tuple:
    """
    Apply augmentation to image and bounding boxes.
    
    Args:
        img: OpenCV image array
        boxes: List of (x, y, w, h) in YOLO format
        classes: List of class IDs
        aug: Albumentations pipeline
        
    Returns:
        Tuple of (augmented_image, augmented_boxes, augmented_classes)
    """
    result = aug(image=img, bboxes=boxes, class_labels=classes)
    return result["image"], result["bboxes"], result["class_labels"]


# =============================================================================
# Dataset Analysis
# =============================================================================
def build_class_index(
    img_dir: Path,
    lbl_dir: Path
) -> Tuple[Dict[int, List[Tuple[Path, Path]]], List[Tuple[Path, Path]]]:
    """
    Build index of images per class.
    
    Args:
        img_dir: Directory containing images
        lbl_dir: Directory containing label files
        
    Returns:
        Tuple of (class_images, all_images) where:
        - class_images: {class_id: [(img_path, lbl_path), ...]}
        - all_images: [(img_path, lbl_path), ...]
    """
    class_images: Dict[int, List[Tuple[Path, Path]]] = defaultdict(list)
    all_images: List[Tuple[Path, Path]] = []

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lines = load_label_lines(lbl_path)

        if not lines:
            continue

        classes_in_img = set(parse_label_line(line)[0] for line in lines)

        for cls in classes_in_img:
            class_images[cls].append((img_path, lbl_path))

        all_images.append((img_path, lbl_path))

    return class_images, all_images


def compute_augmentation_plan(
    class_images: Dict[int, List],
    target: int
) -> List[Tuple[int, int, int]]:
    """
    Compute augmentation plan to reach target per class.
    
    Args:
        class_images: Dictionary of class_id -> image list
        target: Target number of images per class
        
    Returns:
        List of (class_id, current_count, needed_count)
    """
    plan = []

    for cls, imgs in class_images.items():
        current = len(imgs)
        if current < target:
            plan.append((cls, current, target - current))

    return plan


# =============================================================================
# Main Augmentation Function
# =============================================================================
def augment_dataset(
    split: str = "train",
    target: int = DEFAULT_TARGET,
    seed: int = DEFAULT_SEED,
    dry_run: bool = False,
    root: Optional[Path] = None
) -> int:
    """
    Augment dataset to balance class distribution.
    
    Args:
        split: Dataset split to augment ("train", "val", "test")
        target: Target number of images per class
        seed: Random seed for reproducibility
        dry_run: If True, only report plan without writing files
        root: Dataset root directory (defaults to DATASET_ROOT)
        
    Returns:
        Number of augmented images created
    """
    random.seed(seed)

    if root is None:
        root = DATASET_ROOT

    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    if not (img_dir.exists() and lbl_dir.exists()):
        raise FileNotFoundError(f"Split not found: {img_dir}")

    # Build class index
    print(f"Analyzing {split} split...")
    class_images, all_images = build_class_index(img_dir, lbl_dir)

    counts = {cls: len(imgs) for cls, imgs in class_images.items()}
    print(f"Current counts per class: {counts}")

    # Compute plan
    plan = compute_augmentation_plan(class_images, target)

    if not plan:
        print("All classes meet or exceed target. No augmentation needed.")
        return 0

    print("\nAugmentation plan:")
    print("-" * 40)
    for cls, current, needed in plan:
        print(f"  Class {cls}: {current} -> {current + needed} (+{needed})")

    if dry_run:
        print("\nDry-run mode: no files written")
        return 0

    # Create augmentation pipeline
    aug = create_augmentation_pipeline()

    # Determine starting index for new files
    def parse_tail_index(stem: str) -> int:
        try:
            return int(stem.split("_")[-1])
        except Exception:
            return 0

    next_index = max([parse_tail_index(p.stem) for p, _ in all_images] + [0]) + 1

    # Perform augmentation
    print("\nAugmenting...")
    saved = 0

    for cls, current, needed in plan:
        src_list = class_images[cls]

        for _ in range(needed):
            src_img_path, src_lbl_path = random.choice(src_list)

            # Load image
            img = cv2.imread(str(src_img_path))
            if img is None:
                continue

            # Load labels
            lines = load_label_lines(src_lbl_path)
            parsed = [parse_label_line(line) for line in lines]

            if not parsed:
                continue

            class_labels = [c for c, _ in parsed]
            boxes = [b for _, b in parsed]

            # Apply augmentation
            aug_img, aug_boxes, aug_classes = apply_augmentation(
                img, boxes, class_labels, aug
            )

            # Skip if all boxes were removed
            if not aug_boxes:
                continue

            # Save augmented image and labels
            new_name = f"aug_{next_index:06d}.png"
            dst_img = img_dir / new_name
            dst_lbl = lbl_dir / (Path(new_name).stem + ".txt")

            cv2.imwrite(str(dst_img), aug_img)
            save_label(dst_lbl, list(zip(aug_classes, aug_boxes)))

            next_index += 1
            saved += 1

    print(f"\nAugmentation complete. Created {saved} new images.")
    return saved
