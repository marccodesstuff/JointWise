#!/usr/bin/env python3
"""
Augment YOLO dataset to reach target images per class, updating labels too.

Usage (dry-run):
  PYTHONPATH=$(pwd) ./bin/python src/augment_dataset.py --split train --target 200 --dry-run

Real run (careful):
  PYTHONPATH=$(pwd) ./bin/python src/augment_dataset.py --split train --target 10000
"""
import argparse
import random
from pathlib import Path
from collections import defaultdict

import cv2
import albumentations as A


def load_label_lines(path: Path):
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def parse_label_line(line: str):
    parts = line.split()
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:5])
    # ignore optional conf or keypoints if present
    return cls, (x, y, w, h)


def save_label(path: Path, items):
    # items: list of tuples (cls, (x,y,w,h))
    lines = [f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for c, b in items]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def augment(img, boxes, classes, aug):
    # Albumentations expects bboxes=[(x,y,w,h), ...] in 0..1 for format="yolo"
    out = aug(image=img, bboxes=boxes, class_labels=classes)
    return out["image"], out["bboxes"], out["class_labels"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--target", type=int, default=25000, help="Target images per class")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; just report plan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    root = Path("datasets/yolo")
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    assert img_dir.exists() and lbl_dir.exists(), f"Split not found: {img_dir}"

    # Build index per class
    class_images = defaultdict(list)
    all_images = []
    for p in img_dir.iterdir():
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        lbl = lbl_dir / (p.stem + ".txt")
        lines = load_label_lines(lbl)
        if not lines:
            continue
        classes_in_img = set(parse_label_line(l)[0] for l in lines)
        for c in classes_in_img:
            class_images[c].append((p, lbl))
        all_images.append((p, lbl))

    counts = {c: len(v) for c, v in class_images.items()}
    print("Current counts per class:", counts)

    # Aug pipeline (bbox-safe ops)
    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
            A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
            A.Affine(shear=(-10, 10), cval=(0, 0, 0), p=0.5),
            A.Resize(height=320, width=320),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,   # drop tiny/mostly-occluded boxes after aug
            # min_area=16,        # optionally require min area in pixels (set if you want)
        ),
    )

    # Plan
    plan = []
    for cls, imgs in class_images.items():
        cur = len(imgs)
        if cur >= args.target:
            continue
        plan.append((cls, cur, args.target - cur))

    if not plan:
        print("All classes meet or exceed target.")
        return

    print("Augmentation plan (class, current, needed):")
    for p in plan:
        print(p)

    if args.dry_run:
        print("Dry-run: no files will be written")
        return

    # Next index for new filenames
    def parse_tail_index(stem: str) -> int:
        # try to parse *_000123 pattern; fall back to 0
        try:
            return int(stem.split("_")[-1])
        except Exception:
            return 0

    next_index = max([parse_tail_index(p.stem) for p, _ in all_images] + [0]) + 1

    # Augment
    saved = 0
    for cls, cur, need in plan:
        src_list = class_images[cls]
        for _ in range(need):
            src_img_path, src_lbl_path = random.choice(src_list)

            img = cv2.imread(str(src_img_path))
            if img is None:
                continue

            # load yolo boxes for this image
            lines = load_label_lines(src_lbl_path)
            parsed = [parse_label_line(l) for l in lines]
            if not parsed:
                continue
            class_labels = [c for c, _ in parsed]
            boxes = [b for _, b in parsed]

            aug_img, aug_boxes, aug_classes = augment(img, boxes, class_labels, aug)

            # If every box got dropped (e.g., cropped out), skip this sample
            if not aug_boxes:
                continue

            # Save
            new_name = f"aug_{next_index:06d}.png"
            dst_img = img_dir / new_name
            dst_lbl = lbl_dir / (dst_img.stem + ".txt")

            cv2.imwrite(str(dst_img), aug_img)
            save_label(dst_lbl, list(zip(aug_classes, aug_boxes)))

            next_index += 1
            saved += 1

    print(f"Augmentation complete. Saved {saved} new images with updated labels.")


if __name__ == "__main__":
    main()
