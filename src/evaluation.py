# Evaluation metrics for ensemble object detection.
# Provides per-class AP, best F1, adaptive IoU, FROC analysis, and mAP computation.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .config import LABELS_TEST_DIR, STACK_JSON_DIR_DEFAULT
from .utils import iou_xyxy, xywhn_to_xyxy, expand_box, get_image_size


# Read stacked prediction JSON files.
def read_stacked_jsons(dir_path: Path, 
                       use_kept_warn_only: bool = False) -> Dict[str, List[Dict]]:
    out = {}
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Stacked JSON dir not found: {dir_path}")
    
    for p in sorted(dir_path.glob("*.json")):
        j = json.loads(p.read_text())
        img = j.get("image") or j.get("image_path") or p.stem
        preds = j.get("predictions", [])
        
        img_key = Path(str(img)).stem
        out[str(img_key)] = []
        
        for e in preds:
            cls = int(e.get("cls", e.get("class", 0)))
            box = tuple(map(float, e.get("box", e.get("bbox", (0, 0, 0, 0)))))
            conf = float(e.get("conf", e.get("score", 0.0)))
            
            if use_kept_warn_only and conf <= 0.5:
                continue
            
            if use_kept_warn_only:
                kept = bool(e.get("kept", False))
                warn = bool(e.get("warn_far_majority", False))
                if not (kept or warn):
                    continue
            
            out[str(img_key)].append({"cls": cls, "box": box, "conf": conf})
    
    return out


# Read YOLO label file and return list of (class, (cx,cy,w,h)).
def read_yolo_label_file(path: Path) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    out = []
    if not path.exists():
        return out
    
    for ln in path.read_text().splitlines():
        ps = ln.strip().split()
        if len(ps) < 5:
            continue
        cl = int(float(ps[0]))
        cx, cy, w, h = map(float, ps[1:5])
        out.append((cl, (cx, cy, w, h)))
    
    return out


# Load ground truth labels from directory.
def prepare_gt(gt_dir: Path) -> Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]]:
    out = {}
    
    images_dir = gt_dir.parent.parent / "images" / "test"
    img_index = {}
    if images_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for ip in images_dir.glob(ext):
                img_index[ip.stem] = str(ip.resolve())

    for p in sorted(gt_dir.glob("*.txt")):
        stem = p.stem
        out[str(stem)] = read_yolo_label_file(p)
    
    return out


# Filter predictions by confidence and isolation.
def filter_predictions(preds: Dict[str, List[Dict]], 
                       min_conf: float = 0.0,
                       neighbor_iou: float = 0.0, 
                       min_neighbors: int = 0) -> Dict[str, List[Dict]]:
    out = {}
    
    for img, ds in preds.items():
        kept = []
        
        for i, d in enumerate(ds):
            if d["conf"] < min_conf:
                continue
            
            if neighbor_iou > 0 and min_neighbors > 0:
                neigh = 0
                for j, d2 in enumerate(ds):
                    if i == j:
                        continue
                    if d2["cls"] != d["cls"]:
                        continue
                    if iou_xyxy(tuple(d["box"]), tuple(d2["box"])) >= neighbor_iou:
                        neigh += 1
                
                if neigh < min_neighbors:
                    continue
            
            kept.append(d)
        
        out[img] = kept
    
    return out


# Decide if prediction is TP using adaptive IoU threshold.
def adaptive_iou_detection(
    pred_box: Tuple[float, ...],
    gt_box: Tuple[float, ...],
    object_size: float = None,
    cls: int = None,
    class_iou_map: Dict[int, Tuple[float, float]] = None,
    min_iou: float = 0.1,
    max_iou: float = 0.6,
    method: str = "linear"
) -> Tuple[bool, float, float]:
    iou = iou_xyxy(pred_box, gt_box)
    
    if object_size is None:
        gx1, gy1, gx2, gy2 = gt_box
        gw = max(0.0, gx2 - gx1)
        gh = max(0.0, gy2 - gy1)
        object_size = max(gw, gh)
    
    if class_iou_map is not None and cls is not None and cls in class_iou_map:
        cmin, cmax = class_iou_map[cls]
    else:
        cmin, cmax = float(min_iou), float(max_iou)
    
    SMALL_PX = 24.0
    LARGE_PX = 128.0
    
    if method == "logistic":
        x = (object_size - (SMALL_PX + LARGE_PX) / 2.0) / ((LARGE_PX - SMALL_PX) / 6.0)
        frac = 1.0 / (1.0 + np.exp(-x))
    else:
        frac = (object_size - SMALL_PX) / max(1e-6, (LARGE_PX - SMALL_PX))
        frac = min(max(frac, 0.0), 1.0)
    
    threshold = cmin + frac * (cmax - cmin)
    is_tp = iou >= threshold
    
    return bool(is_tp), float(iou), float(threshold)


# Evaluate predictions against ground truth.
def evaluate(
    preds: Dict[str, List[Dict]],
    gts: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]],
    iou_th: float = 0.5,
    tolerance_px: float = 0.0,
    tolerance_rel: float = 0.0,
    debug: bool = False,
    adaptive_iou: bool = False,
    class_iou_map: Dict[int, Tuple[float, float]] = None,
    pred_expand_px: float = 0.0,
    pred_expand_rel: float = 0.0,
    pred_expand_mode: str = "expand_gt",
) -> Dict:
    classes = set()
    for img, gs in gts.items():
        for cl, _ in gs:
            classes.add(cl)
    for img, ds in preds.items():
        for d in ds:
            classes.add(d["cls"])
    classes = sorted(classes)

    ap_per_class = {}
    per_class_results: Dict[int, Dict] = {}
    thresholds_per_class: Dict[int, List[float]] = {cl: [] for cl in classes}
    total_gt = 0
    total_det = 0
    
    if pred_expand_px and float(pred_expand_px) > 0.0:
        print(f"NOTE: pred_expand_px={pred_expand_px}, pred_expand_mode={pred_expand_mode}")
    
    for cl in classes:
        dets = []
        npos = 0
        
        for img, gt_list in gts.items():
            gts_cl = [g for g in gt_list if g[0] == cl]
            npos += len(gts_cl)
        
        for img, det_list in preds.items():
            for d in det_list:
                if d["cls"] != cl:
                    continue
                box = d["box"]
                dets.append((img, float(d["conf"]), tuple(box)))

        total_gt += npos
        total_det += len(dets)
        
        if len(dets) == 0:
            ap_per_class[cl] = 0.0
            per_class_results[cl] = {
                "AP": 0.0, "n_gt": int(npos), "n_det": 0,
                "best_f1": 0.0, "best_prec": 0.0, "best_rec": 0.0,
                "best_conf": None, "tp": 0, "fp": 0, "fn": int(npos),
            }
            continue

        dets = sorted(dets, key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        confs = np.array([d[1] for d in dets], dtype=float)

        matched = {}

        for i, (img, conf, box) in enumerate(dets):
            gt_list = gts.get(img, [])
            box_orig = tuple(box)
            box_used = box_orig
            
            gt_boxes_cl = []
            for idx, (gcl, gxywh) in enumerate(gt_list):
                if gcl != cl:
                    continue
                cx, cy, w, h = gxywh
                W, H = _get_image_size_for_eval(img)
                gt_box = xywhn_to_xyxy(cx, cy, w, h, W, H)
                gt_boxes_cl.append((idx, gt_box))

            ovmax = 0.0
            jmax = -1
            
            for j, gt_b in gt_boxes_cl:
                px = _compute_expansion_px(box_used, pred_expand_px, pred_expand_rel)
                ov = _compute_iou_with_expansion(
                    box_used, gt_b, px, pred_expand_mode
                )
                
                if ov > ovmax:
                    ovmax = ov
                    jmax = j

            thr = float(iou_th)
            matched_gt_box = None
            
            if jmax != -1:
                for jj, gbb in gt_boxes_cl:
                    if jj == jmax:
                        matched_gt_box = gbb
                        break
                
                if adaptive_iou and matched_gt_box is not None:
                    gx1, gy1, gx2, gy2 = matched_gt_box
                    gw = max(0.0, gx2 - gx1)
                    gh = max(0.0, gy2 - gy1)
                    obj_size = max(gw, gh)
                    _, _, thr = adaptive_iou_detection(
                        box_used, matched_gt_box,
                        object_size=obj_size, cls=cl,
                        class_iou_map=class_iou_map
                    )

            thresholds_per_class[cl].append(float(thr))

            if ovmax >= thr:
                key = (img, cl, jmax)
                if key not in matched:
                    tp[i] = 1.0
                    matched[key] = True
                else:
                    fp[i] = 1.0
            else:
                if _check_center_distance_match(
                    box, matched_gt_box, jmax, tolerance_px, tolerance_rel
                ):
                    key = (img, cl, jmax)
                    if key not in matched:
                        tp[i] = 1.0
                        matched[key] = True
                    else:
                        fp[i] = 1.0
                else:
                    fp[i] = 1.0

        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        rec = tp_cum / (npos + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        
        for idx in range(len(mpre) - 1, 0, -1):
            mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
        
        change_idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]) if change_idx.size > 0 else 0.0
        ap_per_class[cl] = float(ap)
        
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = int(np.argmax(f1)) if f1.size > 0 else 0
        best_f1 = float(f1[best_idx]) if f1.size > 0 else 0.0
        best_prec = float(prec[best_idx]) if prec.size > 0 else 0.0
        best_rec = float(rec[best_idx]) if rec.size > 0 else 0.0
        best_conf = float(confs[best_idx]) if confs.size > 0 else None
        
        per_class_results[cl] = {
            "AP": float(ap),
            "n_gt": int(npos),
            "n_det": int(len(dets)),
            "best_f1": best_f1,
            "best_prec": best_prec,
            "best_rec": best_rec,
            "best_conf": best_conf,
            "tp": int(tp_cum[best_idx] if tp_cum.size > 0 else 0),
            "fp": int(fp_cum[best_idx] if fp_cum.size > 0 else 0),
            "fn": int(npos - (tp_cum[best_idx] if tp_cum.size > 0 else 0)),
        }
        
        mean_thr = float(np.mean(thresholds_per_class[cl])) if thresholds_per_class[cl] else float(iou_th)
        print(f"Class {cl}: AP={ap:.4f}, #gt={npos}, #det={len(dets)} | "
              f"bestF1={best_f1:.4f} (P={best_prec:.3f}, R={best_rec:.3f}, "
              f"conf>={best_conf if best_conf is not None else 'n/a'}) meanIoU={mean_thr:.3f}")

    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    
    print("\nPer-class AP:")
    for cl, ap in ap_per_class.items():
        print(f"  class {cl}: {ap:.4f}")
    
    print(f"\nmAP@IoU={iou_th}: {mAP:.4f}")

    valid_cls = [cl for cl in classes if per_class_results.get(cl, {}).get("n_gt", 0) > 0]
    macro_f1 = float(np.mean([per_class_results[cl]["best_f1"] for cl in valid_cls])) if valid_cls else 0.0
    print(f"Macro best-F1: {macro_f1:.4f}")

    return {
        "per_class": per_class_results,
        "mAP": mAP,
        "iou_th": float(iou_th),
        "macro_best_f1": macro_f1,
        "total_gt": int(total_gt),
        "total_det": int(total_det),
        "classes": classes,
    }


# Get image size for evaluation.
def _get_image_size_for_eval(img_stem: str) -> Tuple[int, int]:
    try:
        from PIL import Image
        
        if not any(sep in str(img_stem) for sep in ("/", "\\")):
            images_dir = LABELS_TEST_DIR.parent.parent / "images" / "test"
            if images_dir.exists():
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = images_dir / (str(img_stem) + ext)
                    if candidate.exists():
                        with Image.open(candidate) as im:
                            return im.size
        
        with Image.open(img_stem) as im:
            return im.size
    except Exception:
        return (1, 1)


# Compute expansion in pixels.
def _compute_expansion_px(box: Tuple, pred_expand_px: float, 
                          pred_expand_rel: float) -> float:
    if pred_expand_rel and float(pred_expand_rel) > 0.0:
        bx1, by1, bx2, by2 = box
        bw = max(0.0, bx2 - bx1)
        bh = max(0.0, by2 - by1)
        return float(pred_expand_rel) * max(bw, bh)
    elif pred_expand_px and float(pred_expand_px) > 0.0:
        return float(pred_expand_px)
    return 0.0


# Compute IoU with optional box expansion.
def _compute_iou_with_expansion(box: Tuple, gt_box: Tuple, 
                                 px: float, mode: str) -> float:
    if px <= 0.0:
        return iou_xyxy(box, gt_box)
    
    if mode == "expand_gt":
        return iou_xyxy(box, expand_box(gt_box, px))
    elif mode == "expand_pred":
        return iou_xyxy(expand_box(box, px), gt_box)
    elif mode == "intersect":
        expanded = expand_box(box, px)
        ix1 = max(expanded[0], gt_box[0])
        iy1 = max(expanded[1], gt_box[1])
        ix2 = min(expanded[2], gt_box[2])
        iy2 = min(expanded[3], gt_box[3])
        return 1.0 if (ix2 > ix1 and iy2 > iy1) else 0.0
    else:
        return iou_xyxy(box, expand_box(gt_box, px))


# Check if box matches GT by center distance.
def _check_center_distance_match(box: Tuple, gt_box: Optional[Tuple],
                                  jmax: int, tolerance_px: float,
                                  tolerance_rel: float) -> bool:
    if jmax == -1 or gt_box is None:
        return False
    
    tol_eff = 0.0
    if tolerance_px and tolerance_px > 0.0:
        tol_eff = max(tol_eff, float(tolerance_px))
    if tolerance_rel and tolerance_rel > 0.0:
        gw = max(0.0, gt_box[2] - gt_box[0])
        gh = max(0.0, gt_box[3] - gt_box[1])
        tol_eff = max(tol_eff, float(tolerance_rel) * max(gw, gh))
    
    if tol_eff <= 0.0:
        return False
    
    bx1, by1, bx2, by2 = box
    gx1, gy1, gx2, gy2 = gt_box
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    gcx = 0.5 * (gx1 + gx2)
    gcy = 0.5 * (gy1 + gy2)
    
    center_dist = ((bcx - gcx) ** 2 + (bcy - gcy) ** 2) ** 0.5
    return center_dist <= tol_eff


# Compute FROC (Free-Response ROC) curve.
def compute_froc(
    preds: Dict[str, List[Dict]],
    gts: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]],
    iou_th: float = 0.3,
    fppi_points: Tuple[float, ...] = (0.25, 0.5, 1, 2, 4, 8)
) -> Dict:
    images = sorted(set(list(preds.keys()) + list(gts.keys())))
    n_images = len(images) if len(images) > 0 else 1

    gt_boxes_per_image = {}
    total_gt = 0
    
    for img in images:
        gt_boxes = []
        for cl, xywh in gts.get(img, []):
            cx, cy, w, h = xywh
            W, H = _get_image_size_for_eval(img)
            gt_box = xywhn_to_xyxy(cx, cy, w, h, W, H)
            gt_boxes.append((cl, tuple(gt_box)))
        gt_boxes_per_image[img] = gt_boxes
        total_gt += len(gt_boxes)

    det_list = []
    for img, ds in preds.items():
        for d in ds:
            det_list.append((
                img, float(d["conf"]), 
                int(d["cls"]), tuple(map(float, d["box"]))
            ))

    if len(det_list) == 0:
        return {"fppi": [], "sens": [], "points": {}}

    confs = sorted({c for (_, c, _, _) in det_list}, reverse=True)

    fppi_vals = []
    sens_vals = []

    for thr in confs:
        matched_gt = {img: [False] * len(gt_boxes_per_image.get(img, [])) 
                      for img in images}
        tp = 0
        fp = 0
        
        for img, conf, cls, box in det_list:
            if conf < thr:
                continue
            
            best_iou = 0.0
            best_idx = -1
            gts_here = gt_boxes_per_image.get(img, [])
            
            for idx, (gcl, gbox) in enumerate(gts_here):
                if matched_gt[img][idx]:
                    continue
                iou = iou_xyxy(box, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_idx != -1 and best_iou >= iou_th:
                matched_gt[img][best_idx] = True
                tp += 1
            else:
                fp += 1

        sens = float(tp) / float(total_gt) if total_gt > 0 else 0.0
        fppi = float(fp) / float(n_images)
        fppi_vals.append(fppi)
        sens_vals.append(sens)

    order = np.argsort(np.array(fppi_vals))
    fppi_sorted = np.array(fppi_vals)[order]
    sens_sorted = np.array(sens_vals)[order]
    
    for i in range(1, len(sens_sorted)):
        if sens_sorted[i] < sens_sorted[i - 1]:
            sens_sorted[i] = sens_sorted[i - 1]

    fppi_points_arr = np.array(list(fppi_points), dtype=float)
    
    if fppi_sorted.size == 0:
        interp_sens = np.zeros_like(fppi_points_arr)
    else:
        interp_sens = np.interp(
            fppi_points_arr, fppi_sorted, sens_sorted,
            left=sens_sorted[0] if sens_sorted.size > 0 else 0.0,
            right=sens_sorted[-1] if sens_sorted.size > 0 else 0.0
        )

    points = {float(pp): float(s) 
              for pp, s in zip(fppi_points_arr.tolist(), interp_sens.tolist())}
    
    return {
        "fppi": fppi_sorted.tolist(),
        "sens": sens_sorted.tolist(),
        "points": points,
        "total_gt": total_gt,
        "n_images": n_images
    }
