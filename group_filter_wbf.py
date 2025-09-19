#!/usr/bin/env python3
"""Weighted Boxes Fusion (WBF) grouping for stacked JSON predictions (class 1 only).

Goals:
- Eliminate redundant boxes by clustering overlapping predictions (IoU >= iou_thr)
  and fusing them into a single representative box per cluster.
- Do NOT use ground truth for the elimination logic (GT is only for visualization overlay).
- Leave class 0 predictions unchanged in output JSONs (no kept/group_id modifications for class 0).

Outputs:
- JSON copies to out-json-dir: class 1 predictions get added fields: kept (True for the chosen
  representative per cluster, False otherwise) and group_id (cluster index). The kept prediction's
  box is updated to the fused coordinates.
- Visualizations to out-vis-dir: class 0 boxes yellow; class 1 kept fused box green; other class 1 boxes red; GT on top (blue).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import random
from PIL import Image, ImageDraw, ImageFont


def read_stacked_jsons(dir_path: Path) -> Dict[str, Tuple[Path, List[Dict]]]:
    out = {}
    if not dir_path.exists():
        raise FileNotFoundError(f"Stacked JSON dir not found: {dir_path}")
    for p in sorted(dir_path.glob("*.json")):
        j = json.loads(p.read_text())
        img = j.get("image") or p.stem
        preds = j.get("predictions", [])
        normalized = []
        for e in preds:
            cls = int(e.get("cls", e.get("class", 0)))
            box = tuple(map(float, e.get("box", e.get("bbox", (0,0,0,0)))))
            conf = float(e.get("conf", e.get("score", 0.0)))
            normalized.append({"cls": cls, "box": box, "conf": conf, "orig": e})
        out[str(img)] = (p, normalized)
    return out


def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xi1 = max(ax1, bx1); yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2); yi2 = min(ay2, by2)
    iw = max(0.0, xi2-xi1); ih = max(0.0, yi2-yi1)
    inter = iw*ih
    ua = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    ub = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    union = ua + ub - inter
    return inter/union if union>0 else 0.0


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[float,float,float,float]:
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w/2.0; y1 = cy - h/2.0; x2 = cx + w/2.0; y2 = cy + h/2.0
    return (max(0.0, x1), max(0.0, y1), min(W, x2), min(H, y2))


def draw_overlay(img_path: Path, out_path: Path,
                 boxes: List[Tuple[float,float,float,float]],
                 classes: List[int],
                 kept_mask: List[bool],
                 labels_path: Path | None = None,
                 processed_mask: List[bool] | None = None,
                 elim_reasons: List[str] | None = None,
                 warn_far_mask: List[bool] | None = None):
    try:
        im = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"[WARN] Could not open {img_path}: {e}")
        return
    W,H = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # predictions first
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = b
        eliminated = False
        if elim_reasons is not None and i < len(elim_reasons):
            eliminated = elim_reasons[i] is not None
        if eliminated:
            col = (255,0,255)  # magenta = eliminated (edge/isolated/far_epicenter)
        else:
            proc = processed_mask[i] if processed_mask is not None and i < len(processed_mask) else False
            if not proc:
                col = (255,255,0)  # yellow = unprocessed (skipped by WBF or other class unaffected)
            else:
                # green kept, red not-kept; but allow orange override for WBF-kept warnings
                if kept_mask[i]:
                    warn = warn_far_mask[i] if warn_far_mask is not None and i < len(warn_far_mask) else False
                    # orange = WBF-kept rep flagged as far from majority; green otherwise
                    col = (255,165,0) if warn else (0,255,0)
                else:
                    col = (255,0,0)
        draw.rectangle([x1,y1,x2,y2], outline=col, width=2)

    # ground truth on top
    if labels_path is not None and labels_path.exists():
        try:
            txt = labels_path.read_text().strip().splitlines()
            for ln in txt:
                if not ln.strip():
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    cx,cy,w,h = map(float, parts[1:5])
                except Exception:
                    continue
                gx1,gy1,gx2,gy2 = _yolo_to_xyxy(cx,cy,w,h, W, H)
                draw.rectangle([gx1,gy1,gx2,gy2], outline=(0,0,255), width=3)
                if font is not None:
                    draw.text((gx1+3, gy1+1), str(cls), fill=(0,0,255), font=font)
        except Exception as e:
            print(f"[WARN] Failed reading/drawing labels {labels_path}: {e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def wbf_fuse_once(boxes: List[Tuple[float,float,float,float]],
                  scores: List[float],
                  iou_thr: float,
                  conf_power: float = 1.0) -> List[Dict]:
    """Greedy WBF-style clustering. Returns list of clusters; each item is a dict with:
       - members: indices of boxes in the cluster
       - fused_box: fused XYXY
       - fused_score: aggregated score (mean of member scores)
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    clusters: List[Dict] = []
    for i in order:
        b = boxes[i]; s = scores[i]
        matched = None
        for c in clusters:
            if iou_xyxy(b, c['fused_box']) >= iou_thr:
                matched = c; break
        w = max(1e-6, s ** conf_power)
        if matched is None:
            clusters.append({
                'members': [i],
                'sum_w': w,
                'sum_x1': b[0]*w, 'sum_y1': b[1]*w, 'sum_x2': b[2]*w, 'sum_y2': b[3]*w,
                'fused_box': b,
                'scores': [s]
            })
        else:
            matched['members'].append(i)
            matched['sum_w'] += w
            matched['sum_x1'] += b[0]*w; matched['sum_y1'] += b[1]*w
            matched['sum_x2'] += b[2]*w; matched['sum_y2'] += b[3]*w
            sw = matched['sum_w']
            fx1 = matched['sum_x1']/sw; fy1 = matched['sum_y1']/sw
            fx2 = matched['sum_x2']/sw; fy2 = matched['sum_y2']/sw
            # ensure proper ordering
            matched['fused_box'] = (min(fx1, fx2), min(fy1, fy2), max(fx1, fx2), max(fy1, fy2))
            matched['scores'].append(s)
    # add fused_score
    for c in clusters:
        c['fused_score'] = sum(c['scores'])/len(c['scores']) if c['scores'] else 0.0
    return clusters


def connected_components(indices: List[int],
                         boxes: List[Tuple[float,float,float,float]],
                         iou_thr: float) -> List[List[int]]:
    """Return connected components (as lists of original indices) using IoU>=iou_thr as edges."""
    if not indices:
        return []
    n = len(indices)
    visited = [False]*n
    comps: List[List[int]] = []
    for a in range(n):
        if visited[a]:
            continue
        stack = [a]
        visited[a] = True
        comp_local = []
        while stack:
            i = stack.pop()
            comp_local.append(indices[i])
            bi = boxes[indices[i]]
            for j in range(n):
                if visited[j]:
                    continue
                bj = boxes[indices[j]]
                if iou_xyxy(bi, bj) >= iou_thr:
                    visited[j] = True
                    stack.append(j)
        comps.append(comp_local)
    return comps

def box_center(b: Tuple[float,float,float,float]) -> Tuple[float,float]:
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def process_class(
    target_cls: int,
    boxes: List[Tuple[float,float,float,float]],
    confs: List[float],
    classes: List[int],
    img_size: Tuple[int,int] | None,
    iou_thr: float,
    min_members: int,
    conf_power: float,
    max_area_rel: float,
    max_area_px: float,
    edge_margin_rel: float,
    keep_top_k_components: int,
    small_epicenter_radius_rel: float,
    high_conf_keep_thr: float | None,
    # Component selection beyond top-K
    comp_keep_frac: float = 0.0,
    comp_keep_metric: str = 'size',  # 'size' or 'conf'
    far_majority_warn_rel: float = 0.0,
    # Big-box promotion knobs (only applied when target_cls==1 via caller):
    big_promo_enable: bool = False,
    big_promo_iou: float = 0.4,
    big_promo_min_conf: float = 0.4,
    big_promo_max_area_rel: float = 0.15,
    big_promo_max_area_px: float = 0.0,
) -> Tuple[List[bool], List[bool], List[str | None], List[Dict], List[int]]:
    """Return (keep_mask, processed_mask, elim_reasons, clusters_meta) for a single class.
    clusters_meta is a list of dicts with mapping information to update JSON outside this function.
    """
    W = H = None
    if img_size is not None:
        W, H = img_size
    idx = [i for i,c in enumerate(classes) if c == target_cls]
    keep_mask = [False]*len(boxes)
    processed_mask = [False]*len(boxes)
    elim_reasons: List[str | None] = [None]*len(boxes)
    clusters_meta: List[Dict] = []
    pre_keep: List[int] = []
    if not idx:
        return keep_mask, processed_mask, elim_reasons, clusters_meta, pre_keep

    def area_xyxy(b):
        return max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))

    # Edge eliminations
    edge_elims = set()
    if W is not None and H is not None and edge_margin_rel and edge_margin_rel > 0:
        mx = edge_margin_rel * W; my = edge_margin_rel * H
        for i in idx:
            x1,y1,x2,y2 = boxes[i]
            if x1 <= mx or y1 <= my or (W - x2) <= mx or (H - y2) <= my:
                elim_reasons[i] = 'edge'
                edge_elims.add(i)

    # High-confidence pre-keep (still disallow edges)
    if high_conf_keep_thr is not None:
        for i in idx:
            if i in edge_elims:
                continue
            if confs[i] >= high_conf_keep_thr:
                keep_mask[i] = True
                processed_mask[i] = True
                pre_keep.append(i)

    # Small-box candidate selection for WBF
    thr_px = None
    if max_area_px and max_area_px > 0:
        thr_px = max_area_px
    if max_area_rel and max_area_rel > 0 and W is not None and H is not None:
        rel_px = max_area_rel * (W*H)
        thr_px = min(thr_px, rel_px) if thr_px is not None else rel_px
    if thr_px is not None:
        candidate_idx = [i for i in idx if i not in edge_elims and i not in pre_keep and area_xyxy(boxes[i]) <= thr_px]
    else:
        candidate_idx = [i for i in idx if i not in edge_elims and i not in pre_keep]

    # Build epicenters using selected IoU components among non-edge boxes for this class
    idx_noedge = [i for i in idx if i not in edge_elims]
    kept_members = set(); epicenters: List[Tuple[float,float]] = []
    dominant_center: Tuple[float,float] | None = None
    if idx_noedge:
        comps_all = connected_components(idx_noedge, boxes, iou_thr)
        comps_all.sort(key=lambda comp: len(comp), reverse=True)
        # Start with top-K (if <=0, keep all initially)
        if keep_top_k_components is not None and keep_top_k_components > 0:
            keep_k = max(1, keep_top_k_components)
            kept_components = comps_all[:keep_k]
        else:
            kept_components = list(comps_all)
        # Optionally add components that are within a fraction of the dominant component
        if comp_keep_frac and comp_keep_frac > 0.0 and comps_all:
            # Compute metric per component
            def comp_metric(comp: List[int]) -> float:
                if comp_keep_metric == 'conf':
                    return sum(confs[j] for j in comp)
                # default: size
                return float(len(comp))
            metrics = [comp_metric(c) for c in comps_all]
            max_m = max(metrics) if metrics else 0.0
            thr_m = comp_keep_frac * max_m
            extra = [c for c, m in zip(comps_all, metrics) if m >= thr_m]
            # Union
            kept_components_ids = {id(c) for c in kept_components}
            for c in extra:
                if id(c) not in kept_components_ids:
                    kept_components.append(c)
        for comp in kept_components:
            kept_members.update(comp)
            if comp:
                cx = cy = 0.0
                for j in comp:
                    bx, by = box_center(boxes[j]); cx += bx; cy += by
                cx /= len(comp); cy /= len(comp)
                epicenters.append((cx, cy))
        # dominant component center (largest by size among all comps)
        if comps_all:
            dom = comps_all[0]
            cx = cy = 0.0
            for j in dom:
                bx, by = box_center(boxes[j]); cx += bx; cy += by
            dominant_center = (cx/len(dom), cy/len(dom)) if dom else None

    # Eliminate small boxes not in selected components (minor components)
    if candidate_idx and kept_members:
        not_in_topk = [i for i in candidate_idx if i not in kept_members]
        for i in not_in_topk:
            elim_reasons[i] = 'minor_component'
        candidate_idx = [i for i in candidate_idx if i in kept_members]

    # Eliminate small boxes far from epicenters
    if candidate_idx and epicenters and W is not None and H is not None and small_epicenter_radius_rel and small_epicenter_radius_rel > 0:
        radius_px = small_epicenter_radius_rel * min(W, H)
        def dist2(p, q):
            return (p[0]-q[0])**2 + (p[1]-q[1])**2
        radius2 = radius_px*radius_px
        kept = []
        for i in candidate_idx:
            c = box_center(boxes[i])
            if any(dist2(c, e) <= radius2 for e in epicenters):
                kept.append(i)
            else:
                elim_reasons[i] = 'far_epicenter'
        candidate_idx = kept

    # mark processed
    for i in candidate_idx:
        processed_mask[i] = True

    # WBF
    if candidate_idx:
        b_cand = [boxes[i] for i in candidate_idx]
        s_cand = [confs[i] for i in candidate_idx]
        clusters = wbf_fuse_once(b_cand, s_cand, iou_thr=iou_thr, conf_power=conf_power)
        valid_clusters = [c for c in clusters if len(c['members']) >= min_members]
        for gid, c in enumerate(valid_clusters, start=1):
            members_global = [candidate_idx[m] for m in c['members']]
            rep = max(members_global, key=lambda i: confs[i])
            keep_mask[rep] = True
            clusters_meta.append({'gid': gid, 'rep': rep, 'members': members_global, 'fused_box': c['fused_box']})

    # Far-majority warning: flag WBF representatives far from dominant component center (or global center)
    if far_majority_warn_rel and far_majority_warn_rel > 0.0 and W is not None and H is not None and clusters_meta:
        # If no dominant center was found (no comps), fallback to overall mean of non-edge boxes
        ref_center = dominant_center
        if ref_center is None and idx_noedge:
            cx = cy = 0.0
            for j in idx_noedge:
                bx, by = box_center(boxes[j]); cx += bx; cy += by
            ref_center = (cx/len(idx_noedge), cy/len(idx_noedge)) if idx_noedge else None
        if ref_center is not None:
            R = far_majority_warn_rel * min(W, H)
            R2 = R*R
            def dist2p(p,q):
                return (p[0]-q[0])**2 + (p[1]-q[1])**2
            for meta in clusters_meta:
                c = box_center(meta['fused_box'])
                meta['warn_far'] = dist2p(c, ref_center) > R2

    # Big-box promotion: keep large, non-edge boxes that align with fused clusters
    # Applicable when enabled by caller; intended primarily for class 1
    if big_promo_enable and W is not None and H is not None:
        # Collect fused boxes to align with
        fused_boxes = [m['fused_box'] for m in clusters_meta]
        if fused_boxes:
            # Determine "large" candidates = eligible indices not already processed/kept, not edges
            # Start from all indices of this class excluding edge eliminations and pre-kept
            base = [i for i in idx if i not in edge_elims and i not in pre_keep]
            # Exclude small-box WBF candidates
            large_candidates = [i for i in base if i not in candidate_idx]

            # Apply an optional upper bound on area for promotions
            big_thr_px = None
            if big_promo_max_area_px and big_promo_max_area_px > 0:
                big_thr_px = big_promo_max_area_px
            if big_promo_max_area_rel and big_promo_max_area_rel > 0:
                big_rel_px = big_promo_max_area_rel * (W*H)
                big_thr_px = min(big_thr_px, big_rel_px) if big_thr_px is not None else big_rel_px

            def contains_point(b, p):
                x1,y1,x2,y2 = b; x,y = p
                return (x1 <= x <= x2) and (y1 <= y <= y2)

            for i in large_candidates:
                if processed_mask[i] or keep_mask[i]:
                    continue
                if big_promo_min_conf is not None and confs[i] < big_promo_min_conf:
                    continue
                a = area_xyxy(boxes[i])
                if big_thr_px is not None and a > big_thr_px:
                    continue
                b = boxes[i]
                promote = False
                for fb in fused_boxes:
                    if iou_xyxy(b, fb) >= big_promo_iou:
                        promote = True; break
                    # or contains fused center
                    if contains_point(b, box_center(fb)):
                        promote = True; break
                if promote:
                    keep_mask[i] = True
                    processed_mask[i] = True
    return keep_mask, processed_mask, elim_reasons, clusters_meta, pre_keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack-dir', type=Path, default=Path('runs/classic_train_stack/stacked_test_json'))
    ap.add_argument('--out-json-dir', type=Path, default=Path('runs/classic_train_stack/stacked_test_json_wbf'))
    ap.add_argument('--out-vis-dir', type=Path, default=Path('runs/classic_train_stack/vis_wbf'))
    ap.add_argument('--images-root', type=Path, default=Path('datasets/yolo/images/test'))
    ap.add_argument('--labels-root', type=Path, default=Path('datasets/yolo/labels/test'))
    ap.add_argument('--iou', type=float, default=0.5, help='IoU threshold for clustering')
    ap.add_argument('--min-members', type=int, default=2, help='minimum cluster size to keep')
    ap.add_argument('--conf-power', type=float, default=1.0, help='exponent for confidence weights')
    ap.add_argument('--sample-vis', type=int, default=200)
    ap.add_argument('--sample-random', action='store_true')
    # size filtering: only boxes with area <= thresholds will be considered for WBF
    ap.add_argument('--max-area-rel', type=float, default=0.0, help='max box area as fraction of image area to include in WBF (0 disables)')
    ap.add_argument('--max-area-px', type=float, default=0.0, help='max box area in pixels to include in WBF (0 disables)')
    # pre-WBF elimination of edge-near and isolated boxes
    ap.add_argument('--edge-margin-rel', type=float, default=0.03, help='relative margin from image edges; class-1 boxes crossing into this margin are eliminated pre-WBF')
    ap.add_argument('--keep-top-k-components', type=int, default=1, help='keep only boxes belonging to the top-K largest IoU-connected components (class 1) before WBF; set 0 to disable top-K limiting')
    ap.add_argument('--component-keep-frac', type=float, default=0.0, help='also keep any IoU component whose metric is at least this fraction of the dominant component (0 disables)')
    ap.add_argument('--component-keep-metric', type=str, choices=['size','conf'], default='size', help='metric to compare components when using --component-keep-frac')
    ap.add_argument('--small-epicenter-radius-rel', type=float, default=0.08, help='for small boxes, eliminate if center is farther than this relative radius from the nearest epicenter (top-K component centroid). Radius is fraction of min(W,H).')
    ap.add_argument('--high-conf-keep', type=float, default=None, help='if set, pre-keep class-1 boxes with conf >= this (non-edge only)')
    # warn: WBF representatives far from majority
    ap.add_argument('--far-majority-warn-rel', type=float, default=0.0, help='flag WBF-kept reps whose fused center is farther than this fraction of min(W,H) from the dominant component center (0 disables)')
    # big box promotion (class 1)
    ap.add_argument('--big-promo', action='store_true', help='promote large non-edge class-1 boxes that align with fused clusters (IoU or center containment)')
    ap.add_argument('--big-promo-iou', type=float, default=0.4, help='IoU threshold between large box and fused cluster box to promote')
    ap.add_argument('--big-promo-min-conf', type=float, default=0.4, help='minimum confidence for a large box to be eligible for promotion')
    ap.add_argument('--big-promo-max-area-rel', type=float, default=0.15, help='maximum relative area for a large box to be eligible for promotion (0 disables)')
    ap.add_argument('--big-promo-max-area-px', type=float, default=0.0, help='maximum pixel area for a large box to be eligible for promotion (0 disables)')
    # class-0 specific relaxed settings (defaults looser than class-1)
    ap.add_argument('--c0-iou', type=float, default=None, help='IoU for class-0 clustering (defaults to --iou if None)')
    ap.add_argument('--c0-min-members', type=int, default=1, help='min cluster size for class-0 (default 1 to be lenient)')
    ap.add_argument('--c0-max-area-rel', type=float, default=0.05, help='small-box threshold for class-0 (relative area)')
    ap.add_argument('--c0-max-area-px', type=float, default=0.0, help='small-box threshold for class-0 (pixels)')
    ap.add_argument('--c0-keep-top-k-components', type=int, default=1, help='top-K components to keep for class-0 epicenters (same edge criteria); set 0 to disable')
    ap.add_argument('--c0-small-epicenter-radius-rel', type=float, default=0.12, help='radius for small class-0 boxes near epicenters (rel).')
    ap.add_argument('--c0-component-keep-frac', type=float, default=0.0, help='component keep fraction for class-0 (0 disables)')
    ap.add_argument('--c0-component-keep-metric', type=str, choices=['size','conf'], default='size', help='component metric for class-0 when using fraction keep')
    ap.add_argument('--c0-far-majority-warn-rel', type=float, default=0.0, help='flag WBF-kept reps far from dominant center for class-0 (0 disables)')
    ap.add_argument('--c0-high-conf-keep', type=float, default=None, help='if set, pre-keep class-0 boxes with conf >= this (non-edge only)')
    args = ap.parse_args()

    imgs = read_stacked_jsons(args.stack_dir)
    args.out_json_dir.mkdir(parents=True, exist_ok=True)
    args.out_vis_dir.mkdir(parents=True, exist_ok=True)

    keys = list(imgs.keys())
    render_keys = None
    if args.sample_random:
        k = min(args.sample_vis, len(keys))
        render_keys = set(random.sample(keys, k))

    processed = 0
    for img_key, (src_json_path, preds) in imgs.items():
        boxes = [p['box'] for p in preds]
        confs = [p['conf'] for p in preds]
        classes = [int(p.get('cls', p.get('class', 0))) for p in preds]

        # Class 1 indices
        idx1 = [i for i,c in enumerate(classes) if c == 1]
        keep_mask = [False]*len(boxes)
        processed_mask = [False]*len(boxes)
        elim_reasons: List[str | None] = [None]*len(boxes)

        # Determine image size once
        img_path = Path(json.loads(src_json_path.read_text()).get('image', args.images_root / (Path(src_json_path).stem + '.png')))
        if not img_path.exists():
            img_path = args.images_root / (Path(src_json_path).stem + '.png')
            if not img_path.exists():
                img_path = args.images_root / (Path(src_json_path).stem + '.jpg')
        W = H = None
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            W = H = None

        # Process class 1 (original stricter settings)
        keep1, proc1, elim1, meta1, pre1 = process_class(
            1, boxes, confs, classes, (W,H) if W and H else None,
            args.iou, args.min_members, args.conf_power,
            args.max_area_rel, args.max_area_px,
            args.edge_margin_rel, args.keep_top_k_components, args.small_epicenter_radius_rel,
            args.high_conf_keep,
            args.component_keep_frac, args.component_keep_metric,
            args.far_majority_warn_rel,
            args.big_promo, args.big_promo_iou, args.big_promo_min_conf,
            args.big_promo_max_area_rel, args.big_promo_max_area_px,
        )

        # Process class 0 (looser thresholds, but same edge/isolated logic)
        keep0, proc0, elim0, meta0, pre0 = process_class(
            0, boxes, confs, classes, (W,H) if W and H else None,
            args.c0_iou if args.c0_iou is not None else args.iou,
            args.c0_min_members,
            args.conf_power,
            args.c0_max_area_rel, args.c0_max_area_px,
            args.edge_margin_rel, args.c0_keep_top_k_components, args.c0_small_epicenter_radius_rel,
            args.c0_high_conf_keep,
            args.c0_component_keep_frac, args.c0_component_keep_metric,
            args.c0_far_majority_warn_rel,
            False, 0.0, 0.0, 0.0, 0.0,  # disable big promo for class 0 by default
        )

        # Merge masks and reasons
        keep_mask = [k1 or k0 for k1,k0 in zip(keep1, keep0)]
        processed_mask = [p1 or p0 for p1,p0 in zip(proc1, proc0)]
        elim_reasons = [e1 if e1 else e0 for e1,e0 in zip(elim1, elim0)]

        # Update JSON once with both classes' changes
        out_j = json.loads(src_json_path.read_text())
        out_preds = out_j.get('predictions', [])
        gid_counter = 0
        warn_far_mask = [False]*len(out_preds)
        for meta in (meta1 + meta0):
            gid_counter += 1
            rep = meta['rep']
            members_global = meta['members']
            if 0 <= rep < len(out_preds):
                out_preds[rep]['box'] = list(meta['fused_box'])
                out_preds[rep]['kept'] = True
                out_preds[rep]['group_id'] = gid_counter
                if meta.get('warn_far'):
                    out_preds[rep]['warn_far_majority'] = True
                    warn_far_mask[rep] = True
            for m in members_global:
                if m == rep: continue
                if 0 <= m < len(out_preds):
                    out_preds[m]['kept'] = False
                    out_preds[m]['group_id'] = gid_counter
        # mark pre-kept explicitly
        for i in (pre1 + pre0):
            if 0 <= i < len(out_preds):
                out_preds[i]['kept'] = True
        # annotate eliminations
        for i, reason in enumerate(elim_reasons):
            if reason and 0 <= i < len(out_preds):
                out_preds[i]['kept'] = False
                out_preds[i]['elim_reason'] = reason

        # write JSON
        out_path = args.out_json_dir / src_json_path.name
        out_path.write_text(json.dumps(out_j, indent=2))

        # Visualization
        do_render = (processed < args.sample_vis) if not args.sample_random else (img_key in render_keys)
        if do_render:
            img_file = Path(img_key)
            if not img_file.exists():
                img_file = args.images_root / (Path(img_key).stem + '.png')
                if not img_file.exists():
                    img_file = args.images_root / (Path(img_key).stem + '.jpg')
            labels_path = args.labels_root / (Path(img_key).stem + '.txt')
            vis_path = args.out_vis_dir / f"vis_{Path(img_key).stem}.png"
            draw_overlay(img_file, vis_path, boxes, classes, keep_mask, labels_path=labels_path, processed_mask=processed_mask, elim_reasons=elim_reasons, warn_far_mask=warn_far_mask)
            processed += 1

    print(f"Processed {len(imgs)} JSONs. WBF outputs -> {args.out_json_dir} | visuals -> {args.out_vis_dir}")


if __name__ == '__main__':
    main()
