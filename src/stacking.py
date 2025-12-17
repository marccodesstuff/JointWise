# Stacking ensemble and meta-learner module.
# Handles prediction aggregation, box grouping, meta-learner training, and NMS.

from __future__ import annotations

import math
import pickle
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

from ultralytics import YOLO

from .config import (
    IMG_SIZE, NUM_CLASSES,
    GROUP_IOU, IOU_MATCH, META_TOLERANCE_PX, META_TOLERANCE_REL, NMS_IOU,
    HIGH_PRECISION_CONF_THR, TARGET_CLASS_PRECISION,
    STACK_STAT_FEATURES, STACK_EXTRA_FEATURES,
)
from .utils import (
    get_device, iou_xyxy, center_distance, xywhn_to_xyxy,
    read_yolo_labels, yolo_label_path_for_image,
)


# Run inference on a batch of images.
def predict_boxes(model: YOLO, image_paths: List[str], 
                  conf: float = 0.001, iou: float = 0.5
                  ) -> Dict[str, List[Tuple[int, float, Tuple[float, float, float, float]]]]:
    out = {}
    batch_size = 16
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        results = model.predict(
            source=batch,
            imgsz=IMG_SIZE,
            conf=conf,
            iou=iou,
            device=get_device(),
            verbose=False
        )
        
        for img, r in zip(batch, results):
            H, W = r.orig_shape[:2]
            preds = []
            
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy().astype(int)
                
                for (x1, y1, x2, y2), cf, cl in zip(boxes, confs, clses):
                    preds.append((
                        int(cl),
                        float(cf),
                        (float(x1), float(y1), float(x2), float(y2))
                    ))
            except Exception:
                pass
            
            out[img] = preds
    
    return out


# Group predictions by IoU across models for stacking.
def group_boxes_across_models(
    per_model: List[Dict[str, List[Tuple[int, float, Tuple[float, float, float, float]]]]],
    image_paths: List[str]
) -> Dict[str, List[Dict]]:
    num_models = len(per_model)
    out = {}
    
    for img in image_paths:
        groups = []
        
        flat = []
        for m_idx, pdict in enumerate(per_model):
            for (cl, cf, b) in pdict.get(img, []):
                flat.append((m_idx, cl, cf, b))

        unique_classes = sorted(set(f[1] for f in flat))
        
        for cl in unique_classes:
            items = [(m, cf, b) for (m, ccl, cf, b) in flat if ccl == cl]
            used = [False] * len(items)
            
            for i, (m_i, cf_i, b_i) in enumerate(items):
                if used[i]:
                    continue
                
                group = [(m_i, cf_i, b_i)]
                used[i] = True
                
                for j, (m_j, cf_j, b_j) in enumerate(items):
                    if used[j]:
                        continue
                    if iou_xyxy(b_i, b_j) > GROUP_IOU:
                        used[j] = True
                        group.append((m_j, cf_j, b_j))
                
                xs = [g[2][0] for g in group]
                ys = [g[2][1] for g in group]
                xe = [g[2][2] for g in group]
                ye = [g[2][3] for g in group]
                avg_box = (
                    float(np.mean(xs)),
                    float(np.mean(ys)),
                    float(np.mean(xe)),
                    float(np.mean(ye))
                )
                
                feat = _build_group_features(group, avg_box, num_models, cl)
                
                groups.append({
                    "cls": cl,
                    "members": group,
                    "box": avg_box,
                    "feat": feat
                })
        
        out[img] = groups
    
    return out


# Build feature vector for a prediction group.
def _build_group_features(group: List[Tuple], avg_box: Tuple, 
                          num_models: int, cls: int) -> List[float]:
    feat = [0.0] * num_models
    for (m, cf, _) in group:
        feat[m] = max(feat[m], float(cf))
    
    non_zero = [c for c in feat if c > 0.0]
    if not non_zero:
        non_zero = [0.0]
    
    model_count = float(len([c for c in non_zero if c > 0]))
    sum_conf = float(sum(non_zero))
    mean_conf = sum_conf / len(non_zero)
    max_conf = float(max(non_zero))
    min_conf = float(min(non_zero))
    std_conf = float(np.std(non_zero)) if len(non_zero) > 1 else 0.0
    median_conf = float(np.median(non_zero))
    
    top_sorted = sorted(non_zero, reverse=True)
    top1 = top_sorted[0]
    top2 = top_sorted[1] if len(top_sorted) > 1 else 0.0
    conf_gap = float(top1 - top2)
    
    max_pair_iou = 0.0
    if len(group) > 1:
        pair_ious = [iou_xyxy(a[2], b[2]) for a, b in combinations(group, 2)]
        if pair_ious:
            max_pair_iou = float(max(pair_ious))
    
    bx1, by1, bx2, by2 = avg_box
    width = max(0.0, bx2 - bx1)
    height = max(0.0, by2 - by1)
    area = float(width * height)
    diag = float(math.hypot(width, height))
    aspect = float(width / height) if height > 1e-6 else 0.0
    
    class_one_hot = [1.0 if cls == idx else 0.0 for idx in range(NUM_CLASSES)]
    
    agg_feats = [
        model_count, sum_conf, mean_conf, max_conf, min_conf,
        std_conf, median_conf, conf_gap, max_pair_iou,
        area, width, height, diag, aspect,
    ] + class_one_hot
    
    return feat + agg_feats


# Build training data for the meta-learner.
def build_meta_labels(groups: Dict[str, List[Dict]], 
                      labels_root: Path
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    cls_ids: List[int] = []
    feature_len: Optional[int] = None

    for img, gs in groups.items():
        gts_norm = read_yolo_labels(yolo_label_path_for_image(Path(img), labels_root))
        
        try:
            from PIL import Image
            with Image.open(img) as im:
                width, height = im.size
        except Exception:
            continue

        gts: List[Tuple[int, float, float, float, float]] = []
        for (gt_cls, cx, cy, w, h) in gts_norm:
            x1, y1, x2, y2 = xywhn_to_xyxy(cx, cy, w, h, width, height)
            gts.append((gt_cls, x1, y1, x2, y2))

        for group in gs:
            cls_id = group["cls"]
            box = group["box"]
            is_pos = False

            for (gt_cls, gx1, gy1, gx2, gy2) in gts:
                if gt_cls != cls_id:
                    continue

                gt_box = (gx1, gy1, gx2, gy2)
                
                if iou_xyxy(box, gt_box) >= IOU_MATCH:
                    is_pos = True
                    break

                tolerance = 0.0
                if META_TOLERANCE_PX > 0.0:
                    tolerance = max(tolerance, float(META_TOLERANCE_PX))
                if META_TOLERANCE_REL > 0.0:
                    gw = max(0.0, gx2 - gx1)
                    gh = max(0.0, gy2 - gy1)
                    tolerance = max(tolerance, float(META_TOLERANCE_REL) * max(gw, gh))

                if tolerance > 0.0 and center_distance(box, gt_box) <= tolerance:
                    is_pos = True
                    break

            if feature_len is None:
                feature_len = len(group["feat"])
            
            X.append(group["feat"])
            y.append(1 if is_pos else 0)
            cls_ids.append(cls_id)

    if not X:
        dim = feature_len or 0
        return (
            np.zeros((0, dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int64),
        np.array(cls_ids, dtype=np.int64),
    )


# Create a new meta-learner pipeline.
def create_meta_learner() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")),
    ])


# Save meta-learner to pickle file.
def save_meta_learner(meta: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[STACK] Saved meta model -> {path}")


# Load meta-learner from pickle file.
def load_meta_learner(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# Apply meta-learner to grouped predictions.
def apply_meta_on_groups(groups: Dict[str, List[Dict]], 
                         meta: Any) -> Dict[str, List[Dict]]:
    out = {}
    
    for img, gs in groups.items():
        preds = []
        
        if len(gs) == 0:
            out[img] = preds
            continue
        
        X = np.array([g["feat"] for g in gs], dtype=np.float32)
        
        if hasattr(meta, "predict_proba"):
            ps = meta.predict_proba(X)[:, 1]
        else:
            ps = meta.decision_function(X)
        
        for g, p in zip(gs, ps):
            preds.append({
                "cls": g["cls"],
                "box": g["box"],
                "conf": float(p)
            })
        
        out[img] = preds
    
    return out


# Fallback scoring: average per-model confidences.
def apply_fallback_averaging(groups: Dict[str, List[Dict]], 
                             num_models: int) -> Dict[str, List[Dict]]:
    out = {}
    
    for img, gs in groups.items():
        preds = []
        for g in gs:
            base_feats = g["feat"][:num_models]
            score = float(sum(base_feats)) / max(1, len(base_feats))
            preds.append({
                "cls": g["cls"],
                "box": g["box"],
                "conf": score
            })
        out[img] = preds
    
    return out


# Apply class-wise NMS to predictions.
def nms_by_class(preds: List[Dict], iou_thr: float = NMS_IOU) -> List[Dict]:
    out = []
    
    for cl in sorted(set(p["cls"] for p in preds)):
        items = [p for p in preds if p["cls"] == cl]
        items = sorted(items, key=lambda x: x["conf"], reverse=True)
        
        kept = []
        for e in items:
            b = e["box"]
            if all(iou_xyxy(b, k["box"]) <= iou_thr for k in kept):
                kept.append(e)
        
        out.extend(kept)
    
    return out


# Filter predictions by per-class confidence thresholds.
def filter_by_confidence(preds: List[Dict], 
                         thresholds: Dict[int, float]) -> List[Dict]:
    filtered = []
    default_thr = HIGH_PRECISION_CONF_THR
    
    for p in preds:
        cls_id = int(p["cls"])
        thr = thresholds.get(cls_id, default_thr)
        if p["conf"] >= thr:
            filtered.append(p)
    
    return filtered


# Compute per-class score thresholds that satisfy a precision target.
def compute_class_thresholds(
    meta_model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cls_ids: np.ndarray,
    precision_target: float = TARGET_CLASS_PRECISION,
    default_thr: float = HIGH_PRECISION_CONF_THR
) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
    thresholds: Dict[int, float] = {c: default_thr for c in range(NUM_CLASSES)}
    diagnostics: Dict[int, Dict[str, float]] = {}

    if X.shape[0] == 0:
        return thresholds, diagnostics

    if not hasattr(meta_model, "predict_proba"):
        raise AttributeError("Meta model must have predict_proba method")

    scores = meta_model.predict_proba(X)[:, 1]

    for class_id in range(NUM_CLASSES):
        cls_mask = cls_ids == class_id
        
        if not np.any(cls_mask):
            diagnostics[class_id] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, 
                "threshold": default_thr
            }
            continue

        cls_scores = scores[cls_mask]
        cls_labels = y[cls_mask]
        
        order = np.argsort(cls_scores)[::-1]
        cls_scores_sorted = cls_scores[order]
        cls_labels_sorted = cls_labels[order]

        tp_cum = np.cumsum(cls_labels_sorted)
        fp_cum = np.cumsum(1 - cls_labels_sorted)
        denom = tp_cum + fp_cum
        
        precision = np.divide(
            tp_cum, denom, 
            out=np.zeros_like(tp_cum, dtype=float), 
            where=denom > 0
        )
        
        total_pos = max(1, int(cls_labels.sum()))
        recall = tp_cum / total_pos
        
        f1 = np.divide(
            2 * precision * recall, 
            precision + recall,
            out=np.zeros_like(precision, dtype=float), 
            where=(precision + recall) > 0
        )

        candidate_thresholds = np.concatenate([cls_scores_sorted, [0.0]])
        precision = np.concatenate([precision, [precision[-1] if precision.size else 0.0]])
        recall = np.concatenate([recall, [recall[-1] if recall.size else 0.0]])
        f1 = np.concatenate([f1, [f1[-1] if f1.size else 0.0]])

        satisfying = np.where(precision >= precision_target)[0]
        if satisfying.size > 0:
            best_idx = satisfying[np.argmax(f1[satisfying])]
        else:
            best_idx = int(np.argmax(f1))

        chosen_thr = float(candidate_thresholds[best_idx])
        thresholds[class_id] = chosen_thr
        
        diagnostics[class_id] = {
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1[best_idx]),
            "threshold": chosen_thr,
            "target_precision": precision_target,
        }

    return thresholds, diagnostics


# Evaluate different confidence thresholds on validation set.
def evaluate_confidence_thresholds(
    fused_preds: Dict[str, List[Dict]],
    val_imgs: List[str],
    confidence_thresholds: Optional[List[float]] = None
) -> Tuple[Dict, float]:
    if confidence_thresholds is None:
        confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    results = {}
    
    for conf_thr in confidence_thresholds:
        print(f"[EVAL] Testing confidence threshold: {conf_thr}")
        
        total_preds = 0
        total_filtered = 0
        
        for img in val_imgs:
            preds = fused_preds.get(img, [])
            filtered_preds = [p for p in preds if p["conf"] >= conf_thr]
            
            total_preds += len(preds)
            total_filtered += len(filtered_preds)
        
        retention_rate = total_filtered / max(1, total_preds)
        
        results[conf_thr] = {
            "total_predictions": total_preds,
            "filtered_predictions": total_filtered,
            "retention_rate": retention_rate
        }
        
        print(f"[EVAL] Conf {conf_thr}: {total_filtered}/{total_preds} retained ({retention_rate:.3f})")
    
    recommended_thr = 0.8
    for thr in sorted(confidence_thresholds, reverse=True):
        if results[thr]["retention_rate"] >= 0.3:
            recommended_thr = thr
            break
    
    print(f"\n[EVAL] Recommended threshold: {recommended_thr}")
    print(f"[EVAL] Retains {results[recommended_thr]['retention_rate']:.1%} of predictions")
    
    return results, recommended_thr
