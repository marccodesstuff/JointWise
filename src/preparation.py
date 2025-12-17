# Dataset preparation utilities.
# Converts CSV annotations to YOLO format with subject-level stratified splitting.

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_PNG_DIR = Path("data/png-output")
DEFAULT_CSV_PATH = Path("data/knee.csv")
DEFAULT_OUTPUT_ROOT = Path("datasets/yolo")
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.5
IMG_SIZE = 320

CLASSES = {
    "Ligament - ACL Low Grade sprain": 0,
    "Ligament - ACL High Grade Sprain": 0,
    "Meniscus Tear": 1,
}


# Convert from top-left x,y,width,height to YOLO normalized format.
def yolo_bbox(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int
) -> Tuple[float, float, float, float]:
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


# Construct path to image file.
def find_image(png_dir: Path, file_id: str, slice_idx: int) -> Path:
    fname = f"{file_id}_{slice_idx:03d}.png"
    return png_dir / fname


# Write YOLO format label file from DataFrame group.
def write_labels(group: pd.DataFrame, out_txt: Path, img_w: int = IMG_SIZE, img_h: int = IMG_SIZE) -> None:
    lines = []
    for _, row in group.iterrows():
        cls = int(row["class_id"])
        cx, cy, nw, nh = yolo_bbox(
            row["x"], row["y"], row["width"], row["height"],
            img_w, img_h
        )
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))


# Create subject-level stratified splits.
def create_subject_splits(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = DEFAULT_SEED
) -> Tuple[Set[str], Set[str], Set[str]]:
    subj_groups = df.groupby("file")
    subj_labels = subj_groups["class_id"].max().reset_index()
    subj_labels = subj_labels.rename(columns={"class_id": "subj_label"})
    
    temp_frac = 1.0 - train_frac
    
    train_subj, temp_subj = train_test_split(
        subj_labels,
        test_size=temp_frac,
        random_state=seed,
        stratify=subj_labels["subj_label"]
    )
    
    try:
        val_subj, test_subj = train_test_split(
            temp_subj,
            test_size=0.5,
            random_state=seed,
            stratify=temp_subj["subj_label"]
        )
    except Exception:
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


# Prepare YOLO dataset from CSV annotations.
def prepare_dataset(
    png_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    output_root: Optional[Path] = None,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False
) -> Dict[str, int]:
    if png_dir is None:
        png_dir = DEFAULT_PNG_DIR
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT
    
    if not png_dir.exists():
        raise FileNotFoundError(f"PNG directory not found: {png_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    if force_rebuild and output_root.exists():
        print(f"Removing existing dataset: {output_root}")
        shutil.rmtree(output_root)
    
    img_out = output_root / "images"
    lbl_out = output_root / "labels"
    
    for split in ["train", "val", "test"]:
        (img_out / split).mkdir(parents=True, exist_ok=True)
        (lbl_out / split).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {csv_path}")
    df = pd.read_csv(csv_path)
    
    needed_cols = {"file", "slice", "x", "y", "width", "height", "label"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    df["class_id"] = df["label"].map(CLASSES)
    df = df.dropna(subset=["class_id"])
    
    print(f"Loaded {len(df)} annotations across {df['file'].nunique()} subjects")
    
    df["slice"] = df["slice"].astype(int)
    df["img_path"] = df.apply(
        lambda r: str(find_image(png_dir, r["file"], r["slice"])),
        axis=1
    )
    
    before = len(df)
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()
    after = len(df)
    
    if after < before:
        print(f"Filtered to existing images: {before} -> {after} annotations")
    
    if df.empty:
        raise ValueError("No valid annotations found")
    
    print(f"Creating splits (train={train_frac:.0%}, val={0.5*(1-train_frac):.0%}, test={0.5*(1-train_frac):.0%})")
    train_subj, val_subj, test_subj = create_subject_splits(df, train_frac, seed)
    
    print(f"Subjects: train={len(train_subj)}, val={len(val_subj)}, test={len(test_subj)}")
    
    img_groups = df.groupby("img_path")
    counts = {"train": 0, "val": 0, "test": 0}
    
    for img_path, group in img_groups:
        img_path = Path(img_path)
        subj_id = str(group.iloc[0]["file"])
        
        if subj_id in train_subj:
            split = "train"
        elif subj_id in val_subj:
            split = "val"
        else:
            split = "test"
        
        dst_img = img_out / split / img_path.name
        dst_lbl = lbl_out / split / (img_path.stem + ".txt")
        
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        
        write_labels(group, dst_lbl)
        counts[split] += 1
    
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
