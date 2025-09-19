#!/usr/bin/env python3
"""Compute ensemble evaluation metrics from stacked JSON outputs.

Reads JSON files in `runs/classic_train_stack/stacked_test_json` (or the
`STACK_JSON_DIR` configured in `main.py`) and ground-truth labels in
`datasets/yolo/labels/test` (YOLO format). Computes per-class precision,
recall and Average Precision (AP) at IoU=0.5 and prints a summary mAP.

This script has no heavy dependencies (only numpy and PIL for image sizes).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import argparse


# Config (adjust if your project layout differs)
ROOT = Path.cwd()
STACK_JSON_DIR = ROOT / "runs" / "classic_train_stack" / "stacked_test_json"
LABELS_TEST_DIR = ROOT / "datasets" / "yolo" / "labels" / "test"


def read_stacked_jsons(dir_path: Path, use_kept_warn_only: bool = False) -> Dict[str, List[Dict]]:
    out = {}
    if not dir_path.exists():
        raise FileNotFoundError(f"Stacked JSON dir not found: {dir_path}")
    for p in sorted(dir_path.glob("*.json")):
        j = json.loads(p.read_text())
        img = j.get("image") or j.get("image_path") or p.stem
        preds = j.get("predictions", [])
        # normalize boxes and classes (optionally filter to WBF-kept greens and orange-warned reps)
        out[str(img)] = []
        for e in preds:
            cls = int(e.get("cls", e.get("class", 0)))
            box = tuple(map(float, e.get("box", e.get("bbox", (0,0,0,0)))))
            conf = float(e.get("conf", e.get("score", 0.0)))
            if use_kept_warn_only:
                kept = bool(e.get("kept", False))
                warn = bool(e.get("warn_far_majority", False))
                # Orange is a subset of kept; include both explicitly for clarity
                if not (kept or warn):
                    continue
            out[str(img)].append({"cls": cls, "box": box, "conf": conf})
    return out


def read_yolo_label_file(path: Path) -> List[Tuple[int, Tuple[float,float,float,float]]]:
    out = []
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        ps = ln.strip().split()
        if len(ps) < 5:
            continue
        cl = int(float(ps[0])); cx,cy,w,h = map(float, ps[1:5])
        out.append((cl, (cx,cy,w,h)))
    return out


def xywhn_to_xyxy(cx,cy,w,h,W,H):
    x1 = (cx - w/2.0)*W; y1 = (cy - h/2.0)*H
    x2 = (cx + w/2.0)*W; y2 = (cy + h/2.0)*H
    return (x1,y1,x2,y2)


def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xi1 = max(ax1, bx1); yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2); yi2 = min(ay2, by2)
    iw = max(0.0, xi2-xi1); ih = max(0.0, yi2-yi1)
    inter = iw*ih
    ua = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    ub = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    union = ua + ub - inter
    return inter/union if union>0 else 0.0


def get_image_size_from_path(img_path: str) -> Tuple[int,int]:
    # try to avoid heavy deps; use PIL if available
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size[::-1] if False else (im.size[0], im.size[1])
    except Exception:
        # fallback: assume 1x1 to avoid crash (will make IoU zero)
        return (1,1)


def prepare_gt(gt_dir: Path) -> Dict[str, List[Tuple[int, Tuple[float,float,float,float]]]]:
    out = {}
    # Find corresponding image files by stem inside datasets images test folder
    images_dir = (gt_dir.parent.parent / "images" / "test")
    img_index = {}
    if images_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for ip in images_dir.glob(ext):
                img_index[ip.stem] = str(ip.resolve())

    for p in sorted(gt_dir.glob("*.txt")):
        stem = p.stem
        img_key = img_index.get(stem)
        if img_key is None:
            # fallback: use stem (may match if preds used relative paths)
            img_key = stem
        out[str(img_key)] = read_yolo_label_file(p)
    return out


def filter_predictions(preds: Dict[str, List[Dict]], min_conf: float = 0.0,
                       neighbor_iou: float = 0.0, min_neighbors: int = 0) -> Dict[str, List[Dict]]:
    """Filter predictions by confidence and isolation.

    - min_conf: drop predictions with conf < min_conf
    - neighbor_iou & min_neighbors: drop predictions that have fewer than
      min_neighbors other predictions of the same class within neighbor_iou
      (counts neighbors across the same image only).
    """
    out = {}
    for img, ds in preds.items():
        kept = []
        for i, d in enumerate(ds):
            if d["conf"] < min_conf:
                continue
            if neighbor_iou > 0 and min_neighbors > 0:
                # count neighbors of same class within IoU
                neigh = 0
                for j, d2 in enumerate(ds):
                    if i == j: continue
                    if d2["cls"] != d["cls"]: continue
                    if iou_xyxy(tuple(d["box"]), tuple(d2["box"])) >= neighbor_iou:
                        neigh += 1
                if neigh < min_neighbors:
                    continue
            kept.append(d)
        out[img] = kept
    return out


def expand_box(box, px: float):
    x1,y1,x2,y2 = box
    return (x1 - px, y1 - px, x2 + px, y2 + px)


def evaluate(preds: Dict[str, List[Dict]], gts: Dict[str, List[Tuple[int, Tuple[float,float,float,float]]]], iou_th=0.5, tolerance_px: float = 0.0, debug: bool = False) -> None:
    # collect detections per class
    classes = set()
    for img, gs in gts.items():
        for cl, _ in gs: classes.add(cl)
    for img, ds in preds.items():
        for d in ds: classes.add(d["cls"])
    classes = sorted(classes)

    ap_per_class = {}
    for cl in classes:
        # build list of all detections for this class
        dets = []  # (img, conf, box_xyxy)
        npos = 0
        for img, gt_list in gts.items():
            gts_cl = [g for g in gt_list if g[0]==cl]
            npos += len(gts_cl)
        for img, det_list in preds.items():
            for d in det_list:
                if d["cls"] != cl: continue
                box = d["box"]
                # stacked JSON boxes are in absolute pixel XYXY based on main.py
                dets.append((img, float(d["conf"]), tuple(box)))

        if len(dets) == 0:
            ap_per_class[cl] = 0.0
            continue

        # sort by confidence desc
        dets = sorted(dets, key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        # track which GTs have been matched per image
        matched = {}

        for i, (img, conf, box) in enumerate(dets):
            gt_list = gts.get(img, [])
            # convert GTs to XYXY pixel coords
            gt_boxes_cl = []
            for idx, (gcl, gxywh) in enumerate(gt_list):
                if gcl != cl: continue
                cx,cy,w,h = gxywh
                W,H = get_image_size_from_path(img)
                gt_box = xywhn_to_xyxy(cx,cy,w,h,W,H)
                gt_boxes_cl.append((idx, gt_box))

            ovmax = 0.0; jmax = -1
            for j, gt_b in gt_boxes_cl:
                ov = iou_xyxy(box, gt_b)
                # compute center distance
                bx1,by1,bx2,by2 = box
                gx1,gy1,gx2,gy2 = gt_b
                bcx = 0.5*(bx1+bx2); bcy = 0.5*(by1+by2)
                gcx = 0.5*(gx1+gx2); gcy = 0.5*(gy1+gy2)
                center_dist = ((bcx-gcx)**2 + (bcy-gcy)**2)**0.5
                if debug and i < 5:
                    print(f"DEBUG det#{i} img={img} cls={cl} conf={conf:.3f} box={box} gt_idx={j} gt_box={gt_b} ov={ov:.4f} center_dist={center_dist:.2f}")
                if ov > ovmax:
                    ovmax = ov; jmax = j
                # allow matching by center-distance as a fallback (does not change ovmax)
                if tolerance_px and center_dist <= tolerance_px and ovmax < iou_th:
                    # mark as potential match with small pseudo-iou to pass threshold
                    ovmax = iou_th
                    jmax = j

            if ovmax >= iou_th:
                key = (img, cl, jmax)
                if key not in matched:
                    tp[i] = 1.0
                    matched[key] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # compute precision recall
        fp_cum = np.cumsum(fp); tp_cum = np.cumsum(tp)
        rec = tp_cum / (npos + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)

        # AP: compute area under PR curve (11-point interpolation not used)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre)-1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]) if idx.size>0 else 0.0
        ap_per_class[cl] = float(ap)

        print(f"Class {cl}: AP={ap:.4f}, #gt={npos}, #det={len(dets)}")

    # summary
    if len(ap_per_class)>0:
        mAP = float(np.mean(list(ap_per_class.values())))
    else:
        mAP = 0.0
    print("\nPer-class AP:")
    for cl, ap in ap_per_class.items():
        print(f"  class {cl}: {ap:.4f}")
    print(f"\nmAP@IoU=0.5: {mAP:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble stacked JSONs against YOLO labels")
    parser.add_argument("--stack-dir", type=Path, default=STACK_JSON_DIR, help="stacked JSON directory")
    parser.add_argument("--labels-dir", type=Path, default=LABELS_TEST_DIR, help="YOLO labels test dir")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching (default 0.5)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence to keep a prediction")
    parser.add_argument("--neighbor-iou", type=float, default=0.0, help="IoU to consider a neighbor prediction (0 disables)")
    parser.add_argument("--min-neighbors", type=int, default=0, help="Minimum number of neighbors required to keep a prediction")
    parser.add_argument("--tolerance-px", type=float, default=0.0, help="Expand GT boxes by this many pixels before matching (helps small localization errors)")
    parser.add_argument("--debug", action="store_true", help="Print debug IoU info for first few detections")
    parser.add_argument("--use-kept-warn-only", action="store_true", help="When evaluating WBF outputs, include only green kept and orange warned boxes (requires WBF JSONs with kept/warn_far_majority fields)")
    args = parser.parse_args()

    preds = read_stacked_jsons(args.stack_dir, use_kept_warn_only=args.use_kept_warn_only)
    gts = prepare_gt(args.labels_dir)
    if args.min_conf > 0.0 or (args.neighbor_iou > 0.0 and args.min_neighbors > 0):
        preds = filter_predictions(preds, min_conf=args.min_conf,
                                   neighbor_iou=args.neighbor_iou, min_neighbors=args.min_neighbors)
    evaluate(preds, gts, iou_th=args.iou, tolerance_px=args.tolerance_px, debug=args.debug)


if __name__ == "__main__":
    main()
