import os
import math
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


CLASSES = {
    "Ligament - ACL Low Grade sprain": 0,
    "Ligament - ACL High Grade Sprain": 0,
    "Meniscus Tear": 1,
}


def yolo_bbox(x, y, w, h, img_w, img_h):
    # Convert from top-left x,y,width,height to YOLO cx,cy,nw,nh normalized
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def find_image(png_dir: Path, file_id: str, slice_idx: int) -> Path:
    fname = f"{file_id}_{slice_idx:03d}.png"
    return png_dir / fname


def main():
    png_dir = Path("data/png-output/png-output")
    assert png_dir.exists(), f"PNG_PATH not found: {png_dir}"

    csv_path = Path("data/knee.csv")
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    out_root = Path("datasets/yolo")
    # Optional clean rebuild to avoid mixing different split strategies
    force_rebuild = os.getenv("FORCE_REBUILD", "0") in ("1", "true", "True")
    if force_rebuild and out_root.exists():
        shutil.rmtree(out_root)
    img_out = out_root / "images"
    lbl_out = out_root / "labels"
    for split in ["train", "val", "test"]:
        (img_out / split).mkdir(parents=True, exist_ok=True)
        (lbl_out / split).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic validation and filtering
    needed_cols = {"file", "slice", "x", "y", "width", "height", "label"}
    missing = needed_cols - set(df.columns)
    assert not missing, f"Missing columns in CSV: {missing}"

    # Map labels
    df["class_id"] = df["label"].map(CLASSES)
    df = df.dropna(subset=["class_id"])  # drop irrelevant labels if any

    # Build image list with grouped boxes per image
    df["slice"] = df["slice"].astype(int)
    df["img_path"] = df.apply(lambda r: str(find_image(png_dir, r["file"], r["slice"])), axis=1)
    # Remove rows for missing images
    df = df[df["img_path"].apply(lambda p: Path(p).exists())].copy()

    # Subject-level split: group by 'file' so all slices/images for a subject go to same split
    # Determine a subject-level label using max class presence within that subject
    subj_groups = df.groupby("file")
    subj_labels = subj_groups["class_id"].max().reset_index().rename(columns={"class_id": "subj_label"})

    # Desired splits: train 70%, val 15%, test 15%
    # We'll perform a two-stage stratified split: first train vs temp (30%), then val/test split the temp equally.
    SEED = int(os.getenv("SPLIT_SEED", "42"))
    train_frac = float(os.getenv("TRAIN_FRAC", "0.7"))
    temp_frac = 1.0 - train_frac  # 0.3

    # First split train vs temp
    train_subj, temp_subj = train_test_split(
        subj_labels, test_size=temp_frac, random_state=SEED, stratify=subj_labels["subj_label"]
    )

    # Split temp into val and test equally to get 0.15/0.15
    # If temp_subj is small, train_test_split with stratify may fail; fall back to non-stratified split
    try:
        val_subj, test_subj = train_test_split(
            temp_subj, test_size=0.5, random_state=SEED, stratify=temp_subj["subj_label"]
        )
    except Exception:
        val_subj, test_subj = train_test_split(
            temp_subj, test_size=0.5, random_state=SEED
        )

    train_subj_set = set(train_subj["file"].tolist())
    val_subj_set = set(val_subj["file"].tolist())
    test_subj_set = set(test_subj["file"].tolist())

    # For file copying and label writing, still iterate per image path but map by its subject (file)
    img_groups = df.groupby("img_path")

    # Function to write label file
    def write_labels(group: pd.DataFrame, out_txt: Path):
        # Images are 320x320 by requirement
        img_w = 320
        img_h = 320
        lines = []
        for _, r in group.iterrows():
            cls = int(r["class_id"])
            cx, cy, nw, nh = yolo_bbox(r["x"], r["y"], r["width"], r["height"], img_w, img_h)
            # clamp
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))

    # Copy images and write labels
    for img_path, group in img_groups:
        img_path = Path(img_path)
        # Extract subject id from first row's 'file' since group shares same image
        subj_id = str(group.iloc[0]["file"])
        if subj_id in train_subj_set:
            split = "train"
        elif subj_id in val_subj_set:
            split = "val"
        else:
            split = "test"
        dst_img = img_out / split / img_path.name
        dst_lbl = lbl_out / split / (img_path.stem + ".txt")
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        write_labels(group, dst_lbl)

    # Write a small info file
    (out_root / "README.txt").write_text("YOLO dataset prepared from knee.csv.\n")
    print("Dataset prepared under datasets/yolo")


if __name__ == "__main__":
    main()
