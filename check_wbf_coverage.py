#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import argparse
import csv


def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xi1 = max(ax1, bx1); yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2); yi2 = min(ay2, by2)
    iw = max(0.0, xi2-xi1); ih = max(0.0, yi2-yi1)
    inter = iw*ih
    ua = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    ub = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    union = ua + ub - inter
    return inter/union if union>0 else 0.0


def yolo_to_xyxy(cx, cy, w, h, W, H):
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return (max(0.0,x1), max(0.0,y1), min(W,x2), min(H,y2))


def read_labels(lbl_path: Path, img_path: Path) -> List[Tuple[float,float,float,float,int]]:
    if not lbl_path.exists():
        return []
    try:
        W,H = Image.open(img_path).size
    except Exception:
        return []
    out=[]
    for ln in lbl_path.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0])); cx,cy,w,h = map(float, parts[1:5])
        out.append((*yolo_to_xyxy(cx,cy,w,h, W,H), cls))
    return out


def match_greedy(preds, gts, iou_thr):
    """Greedy 1:1 matching: sort preds by area desc to reduce small-box preference; return matched indices.
    preds, gts are lists of boxes (xyxy). Returns (matched_pred_idx_set, matched_gt_idx_set).
    """
    used_gt = set()
    used_pr = set()
    # sort preds by area descending
    order = sorted(range(len(preds)), key=lambda i: (preds[i][2]-preds[i][0])*(preds[i][3]-preds[i][1]), reverse=True)
    for i in order:
        best_j = -1; best_iou = 0.0
        for j in range(len(gts)):
            if j in used_gt: continue
            v = iou(preds[i], gts[j])
            if v >= iou_thr and v > best_iou:
                best_iou = v; best_j = j
        if best_j >= 0:
            used_pr.add(i); used_gt.add(best_j)
    return used_pr, used_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', type=Path, default=Path('runs/classic_train_stack/stacked_test_json_wbf'))
    ap.add_argument('--images-dir', type=Path, default=Path('datasets/yolo/images/test'))
    ap.add_argument('--labels-dir', type=Path, default=Path('datasets/yolo/labels/test'))
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--out-csv', type=Path, default=Path('runs/classic_train_stack/wbf_coverage.csv'))
    args = ap.parse_args()

    files = sorted(args.json_dir.glob('*.json'))
    covered = 0
    total_with_gt = 0
    total_gt = 0
    total_kept = 0
    matched_gt_total = 0
    matched_kept_total = 0

    rows = []
    for p in files:
        j = json.loads(p.read_text())
        img_path = Path(j.get('image', args.images_dir / (p.stem + '.png')))
        if not img_path.exists():
            img_path = args.images_dir / (p.stem + '.png')
            if not img_path.exists():
                img_path = args.images_dir / (p.stem + '.jpg')
        lbl_path = args.labels_dir / (p.stem + '.txt')
        gts_all = [b for b in read_labels(lbl_path, img_path)]
        gts = [gt[:4] for gt in gts_all if gt[-1] == 1]
        if not gts:
            # no class-1 GT in this image
            rows.append([p.name, 0, 0, 0, 0, 1])
            continue
        total_with_gt += 1
        total_gt += len(gts)
        preds = j.get('predictions', [])
        kept1 = [tuple(pr['box']) for pr in preds if int(pr.get('cls', pr.get('class',0)))==1 and pr.get('kept', False)]
        total_kept += len(kept1)
        # image-level coverage (at least one match)
        hit = 0
        for kb in kept1:
            if any(iou(kb, gt) >= args.iou for gt in gts):
                hit = 1; break
        covered += hit
        # 1:1 matching for precision/recall
        m_pr, m_gt = match_greedy(kept1, gts, args.iou)
        matched_kept_total += len(m_pr)
        matched_gt_total += len(m_gt)
        rows.append([p.name, len(kept1), len(gts), len(m_pr), len(m_gt), hit])

    cov = covered / max(1,total_with_gt)
    recall = matched_gt_total / max(1,total_gt)
    precision = matched_kept_total / max(1,total_kept)
    print(f"Images with class-1 GT: {total_with_gt}")
    print(f"At-least-one match coverage @ IoU>={args.iou}: {covered} ({cov:.3f})")
    print(f"Total class-1 GT boxes: {total_gt}; matched: {matched_gt_total} -> recall: {recall:.3f}")
    print(f"Total kept class-1 boxes: {total_kept}; matched: {matched_kept_total} -> precision: {precision:.3f}")

    # write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file','kept_count','gt_count','matched_kept','matched_gt','covered'])
        w.writerows(rows)
    print(f"Wrote per-image coverage CSV -> {args.out_csv}")

if __name__ == '__main__':
    main()
