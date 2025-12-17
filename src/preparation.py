"""
Dataset preparation utilities.

This module converts CSV annotations to YOLO format with proper
subject-level stratified splitting.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_PNG_DIR = Path("data/png-output")
DEFAULT_CSV_PATH = Path("data/knee.csv")
DEFAULT_OUTPUT_ROOT = Path("datasets/yolo")
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.5
IMG_SIZE = 320

# Class mapping: label text -> class ID
CLASSES = {
    "Ligament - ACL Low Grade sprain": 0,
    "Ligament - ACL High Grade Sprain": 0,
    "Meniscus Tear": 1,
}


# =============================================================================
# Coordinate Conversion
# =============================================================================
def yolo_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert from top-left x,y,width,height to YOLO normalized format.
    
    Args:
        x, y: Top-left corner coordinates
        w, h: Width and height
        img_w, img_h: Image dimensions
        
    Returns:
        Tuple of (center_x, center_y, norm_width, norm_height) in 0-1 range
    """
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def find_image(png_dir: Path, file_id: str, slice_idx: int) -> Path:
    """
    Construct path to image file.
    
    Args:
        png_dir: Directory containing PNG images
        file_id: Patient/file identifier
        slice_idx: Slice index number
        
    Returns:
        Path to the image file
    """
    fname = f"{file_id}_{slice_idx:03d}.png"
    return png_dir / fname


# =============================================================================
# Label Writing
# =============================================================================
def write_labels(group: pd.DataFrame, out_txt: Path, img_w: int = IMG_SIZE, img_h: int = IMG_SIZE) -> None:
    """
    Write YOLO format label file from DataFrame group.
    
    Args:
        group: DataFrame with x, y, width, height, class_id columns
        out_txt: Output path for label file
        img_w, img_h: Image dimensions
    """
    lines = []
    for _, row in group.iterrows():
        cls = int(row["class_id"])
        cx, cy, nw, nh = yolo_bbox(
            row["x"], row["y"], row["width"], row["height"],
            img_w, img_h
        )
        # Clamp to valid range
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))


# =============================================================================
# Subject-Level Splitting
# =============================================================================
def create_subject_splits(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = DEFAULT_SEED
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Create subject-level stratified splits.
    
    Ensures all slices from same subject go to same split.
    
    Args:
        df: DataFrame with 'file' and 'class_id' columns
        train_frac: Fraction for training set
        seed: Random seed
        
    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects) as sets
    """
    # Determine subject-level label using max class presence
    subj_groups = df.groupby("file")
    subj_labels = subj_groups["class_id"].max().reset_index()
    subj_labels = subj_labels.rename(columns={"class_id": "subj_label"})
    
    temp_frac = 1.0 - train_frac
    
    # First split: train vs temp
    train_subj, temp_subj = train_test_split(
        subj_labels,
        test_size=temp_frac,
        random_state=seed,
        stratify=subj_labels["subj_label"]
    )
    
    # Second split: val and test from temp (equal parts)
    try:
        val_subj, test_subj = train_test_split(
            temp_subj,
            test_size=0.5,
            random_state=seed,
            stratify=temp_subj["subj_label"]
        )
    except Exception:
        # Fallback if stratification fails (small dataset)
        val_subj, test_subj = train_test_split(
            temp_subj,
            test_size=0.5,
            random_state=seed
        )
    
    return (
        set(train_subj["file"].tolist()),
        set(val_subj["file"].tolist()),
        set(test_subj["file"].tolist())
    )


# =============================================================================
# Main Preparation Function
# =============================================================================
def prepare_dataset(
    png_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False
) -> Dict[str, int]:
    """
    Prepare YOLO dataset from CSV annotations.
    
    Args:
        png_dir: Directory containing source PNG images
        csv_path: Path to annotations CSV
        output_root: Output directory for YOLO dataset
        train_frac: Fraction for training split
        seed: Random seed for splitting
        force_rebuild: If True, delete existing output first
        
    Returns:
        Dictionary with counts per split
    """
    # Set defaults
    if png_dir is None:
        png_dir = DEFAULT_PNG_DIR
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT
    
    # Validate inputs
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    # Handle rebuild
    if force_rebuild and output_root.exists():
        print(f"Removing existing dataset: {output_root}")
        shutil.rmtree(output_root)
    
    # Create output directories
    img_out = output_root / "images"
    lbl_out = output_root / "labels"
    
    for split in ["train", "val", "test"]:
        (img_out / split).mkdir(parents=True, exist_ok=True)
        (lbl_out / split).mkdir(parents=True, exist_ok=True)
    
    # Load and validate CSV
    print(f"Loading annotations from {csv_path}")
    df = pd.read_csv(csv_path)
    
    needed_cols = {"file", "slice", "x", "y", "width", "height", "label"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    # Map labels to class IDs
    df["class_id"] = df["label"].map(CLASSES)
    df = df.dropna(subset=["class_id"])
    
    print(f"Loaded {len(df)} annotations across {df['file'].nunique()} subjects")
    
    # Build image paths
    df["slice"] = df["slice"].astype(int)
    df["img_path"] = df.apply(
        lambda r: str(find_image(png_dir, r["file"], r["slice"])),
        axis=1
    )
    
    # Filter to existing images
    before = len(df)
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    after = len(df)
    
    if after < before:
        print(f"Filtered to existing images: {before} -> {after} annotations")
    
    if df.empty:
        raise ValueError("No valid annotations found")
    
    # Create splits
    print(f"Creating splits (train={train_frac:.0%}, val={0.5*(1-train_frac):.0%}, test={0.5*(1-train_frac):.0%})")
    train_subj, val_subj, test_subj = create_subject_splits(df, train_frac, seed)
    
    print(f"Subjects: train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)}")
    
    # Process images
    img_groups = df.groupby("img_path")
    counts = {"train": 0, "val": 0, "test": 0}
    
    for img_path, group in img_groups:
        img_path = Path(img_path)
        subj_id = str(group.iloc[0]["file"])
        
        # Determine split
        if subj_id in train_subj:
            split = "train"
        elif subj_id in val_subj:
            split = "val"
        else:
            split = "test"
        
        dst_img = img_out / split / img_path.name
        dst_lbl = lbl_out / split / (img_path.stem + ".txt")
        
        # Copy image if not exists
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        
        # Write labels
        write_labels(group, dst_lbl)
        counts[split] += 1
    
    # Write info file
    readme = output_root / "README.txt"
    readme.write_text(
        f"YOLO dataset prepared from {csv_path.name}\n"
        f"Train: {counts['train']} images\n"
        f"Val: {counts['val']} images\n"
        f"Test: {counts['test']} images\n"
    )
    
    print(f"\nDataset prepared: {output_root}")
    print(f"  Train: {counts['train']} images")
    print(f"  Val: {counts['val']} images")
    print(f"  Test: {counts['test']} images")
    
    return counts
