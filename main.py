#!/usr/bin/env python3
# Jointwise Model Development CLI.
# Unified command-line interface for prepare, augment, train, and evaluate commands.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


# Run dataset preparation.
def cmd_prepare(args):
    from src.preparation import prepare_dataset
    
    prepare_dataset(
        png_dir=args.png_dir,
        csv_path=args.csv,
        output_root=args.output,
        train_frac=args.train_frac,
        seed=args.seed,
        force_rebuild=args.force,
    )


# Run data augmentation.
def cmd_augment(args):
    from src.augmentation import augment_dataset
    
    augment_dataset(
        split=args.split,
        target=args.target,
        seed=args.seed,
        dry_run=args.dry_run,
    )


# Run training pipeline.
def cmd_train(args):
    from ultralytics import YOLO
    
    from src.config import (
        IMAGES_DIR, LABELS_DIR, DATA_YAML, PROJECT_DIR,
        META_MODEL_PATH, STACK_JSON_DIR, NUM_CLASSES,
        GA_ENABLE, HIGH_PRECISION_CONF_THR, IOU_MATCH, NMS_IOU,
        TARGET_CLASS_PRECISION, STACK_EXTRA_FEATURES,
        ensure_directories,
    )
    from src.models import (
        XCEPTION_YAML, RESNEXT_YAML, DENSENET_YAML, EFFICIENTNET_YAML,
        register_custom_modules,
    )
    from src.utils import list_images, create_dataset_yaml, clear_dataset_caches
    from src.training import train_family
    from src.stacking import (
        predict_boxes, group_boxes_across_models,
        build_meta_labels, create_meta_learner,
        save_meta_learner, load_meta_learner,
        apply_meta_on_groups, apply_fallback_averaging,
        nms_by_class, filter_by_confidence,
        compute_class_thresholds, evaluate_confidence_thresholds,
    )
    
    FAMILIES = [
        ("xception", XCEPTION_YAML),
        ("resnext", RESNEXT_YAML),
        ("densenet", DENSENET_YAML),
        ("efficientnet", EFFICIENTNET_YAML),
    ]
    
    ga_enabled = GA_ENABLE
    precision_thr = HIGH_PRECISION_CONF_THR
    iou_match_thr = IOU_MATCH
    
    if args.ga_disable:
        ga_enabled = False
    if args.precision_conf is not None:
        precision_thr = args.precision_conf
    if args.iou_match is not None:
        iou_match_thr = args.iou_match
    
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  GA enabled: {ga_enabled}")
    print(f"  IOU_MATCH threshold: {iou_match_thr}")
    print(f"  High-precision confidence: {precision_thr}")
    print(f"  Reuse base models: {args.reuse_models}")
    print(f"  Reuse meta-learner: {args.reuse_meta}")
    print(f"  Threshold evaluation: {args.eval_thresholds}")
    print("=" * 60)
    
    ensure_directories()
    register_custom_modules()
    
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise RuntimeError(
            "Dataset not found. Run 'python main.py prepare' first."
        )
    
    create_dataset_yaml(IMAGES_DIR, LABELS_DIR, DATA_YAML, nc=NUM_CLASSES)
    clear_dataset_caches(LABELS_DIR)
    
    print("\n" + "=" * 60)
    print("PHASE 1: MODEL TRAINING")
    print("=" * 60)
    
    trained = []
    
    if args.reuse_models:
        print("[TRAIN] Reusing existing checkpoints...")
        for name, _ in FAMILIES:
            weights_dir = PROJECT_DIR / f"{name}_final" / "weights"
            candidates = [weights_dir / "best.pt", weights_dir / "last.pt"]
            ckpt_path = next((p for p in candidates if p.exists()), None)
            
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No checkpoint for '{name}'. Run training first."
                )
            
            print(f"[TRAIN][reuse] {name} -> {ckpt_path}")
            trained.append((name, ckpt_path, {}))
    else:
        for name, yaml_text in FAMILIES:
            fam, ckpt, hp = train_family(
                name, yaml_text,
                force_ga=args.force_ga,
                skip_ga=args.skip_ga,
                predict_boxes_fn=predict_boxes
            )
            trained.append((fam, ckpt, hp))
    
    print("\n" + "=" * 60)
    print("PHASE 2: LOADING MODELS")
    print("=" * 60)
    
    yolos = []
    for fam, ckpt, _ in trained:
        print(f"Loading {fam} from {ckpt}")
        yolos.append((fam, YOLO(str(ckpt))))
    
    num_models = len(FAMILIES)
    
    class_thresholds: Dict[int, float] = {
        cls: precision_thr for cls in range(NUM_CLASSES)
    }
    thresholds_path = PROJECT_DIR / "meta_class_thresholds.json"
    
    print("\n" + "=" * 60)
    print("PHASE 3: META-LEARNER")
    print("=" * 60)
    
    meta: Optional[Any] = None
    meta_loaded = False
    
    if args.reuse_meta and META_MODEL_PATH.exists():
        meta = load_meta_learner(META_MODEL_PATH)
        
        expected_dim = num_models + STACK_EXTRA_FEATURES
        model_dim = None
        
        if hasattr(meta, "n_features_in_"):
            model_dim = getattr(meta, "n_features_in_")
        elif hasattr(meta, "coef_"):
            model_dim = meta.coef_.shape[1]
        
        if model_dim is not None and model_dim != expected_dim:
            print(f"[STACK][warn] Dimension mismatch; retraining")
            meta = None
        else:
            meta_loaded = True
            print(f"[STACK] Reused meta-learner from {META_MODEL_PATH}")
            
            if thresholds_path.exists():
                try:
                    loaded = json.loads(thresholds_path.read_text())
                    class_thresholds.update({
                        int(k): float(v)
                        for k, v in loaded.get("thresholds", {}).items()
                    })
                except Exception as e:
                    print(f"[STACK][warn] Failed to load thresholds: {e}")
    
    elif args.reuse_meta:
        print(f"[STACK][warn] --reuse-meta but model not found; training new")
    
    meta_needs_thresholds = meta_loaded and not thresholds_path.exists()
    need_val_preds = (not meta_loaded) or args.eval_thresholds or meta_needs_thresholds
    
    val_imgs: List[str] = []
    val_groups: Dict[str, List[Dict]] = {}
    X_val = None
    y_val = None
    cls_val = None
    
    if need_val_preds:
        print("\n[STACK] Generating validation predictions...")
        val_imgs = list_images(IMAGES_DIR / "val")
        
        per_model_preds_val = []
        for fam, model in yolos:
            print(f"  Predicting val with {fam}...")
            per_model_preds_val.append(predict_boxes(model, val_imgs, conf=0.001, iou=0.5))
        
        val_groups = group_boxes_across_models(per_model_preds_val, val_imgs)
        X_val, y_val, cls_val = build_meta_labels(val_groups, LABELS_DIR / "val")
    
    if not meta_loaded:
        if not val_groups or X_val is None or X_val.shape[0] == 0:
            print("[STACK] No validation groups; using fallback averaging")
            meta = None
        else:
            print(f"[STACK] Training meta-learner: X={X_val.shape}")
            
            meta = create_meta_learner()
            meta.fit(X_val, y_val)
            save_meta_learner(meta, META_MODEL_PATH)
            
            class_thresholds, thr_metrics = compute_class_thresholds(
                meta, X_val, y_val, cls_val,
                TARGET_CLASS_PRECISION, precision_thr
            )
            
            thresholds_payload = {
                "thresholds": class_thresholds,
                "target_precision": TARGET_CLASS_PRECISION,
                "metrics": thr_metrics,
            }
            thresholds_path.write_text(json.dumps(thresholds_payload, indent=2))
    
    elif meta_loaded and meta_needs_thresholds and X_val is not None:
        class_thresholds, thr_metrics = compute_class_thresholds(
            meta, X_val, y_val, cls_val,
            TARGET_CLASS_PRECISION, precision_thr
        )
        
        thresholds_payload = {
            "thresholds": class_thresholds,
            "metrics": thr_metrics,
        }
        thresholds_path.write_text(json.dumps(thresholds_payload, indent=2))
    
    print("\n[STACK] Active per-class thresholds:")
    for cls_id in range(NUM_CLASSES):
        print(f"    Class {cls_id}: {class_thresholds.get(cls_id, precision_thr):.4f}")
    
    if args.eval_thresholds and val_groups and meta is not None:
        print("\n" + "=" * 60)
        print("PHASE 4: THRESHOLD EVALUATION")
        print("=" * 60)
        
        fused_val = apply_meta_on_groups(val_groups, meta)
        eval_results, recommended_thr = evaluate_confidence_thresholds(fused_val, val_imgs)
        
        eval_file = PROJECT_DIR / "confidence_threshold_evaluation.json"
        with open(eval_file, "w") as f:
            json.dump({
                "evaluation_results": eval_results,
                "recommended_threshold": recommended_thr,
            }, f, indent=2)
        print(f"[STACK] Saved evaluation to {eval_file}")
    
    print("\n" + "=" * 60)
    print("PHASE 5: TEST INFERENCE")
    print("=" * 60)
    
    test_imgs = list_images(IMAGES_DIR / "test")
    
    per_model_preds_test = []
    for fam, model in yolos:
        print(f"  Predicting test with {fam}...")
        per_model_preds_test.append(predict_boxes(model, test_imgs, conf=0.001, iou=0.5))
    
    test_groups = group_boxes_across_models(per_model_preds_test, test_imgs)
    
    if meta is None:
        fused = apply_fallback_averaging(test_groups, num_models)
    else:
        fused = apply_meta_on_groups(test_groups, meta)
    
    for img in test_imgs:
        preds = fused.get(img, [])
        filtered = filter_by_confidence(preds, class_thresholds)
        final = nms_by_class(filtered, iou_thr=NMS_IOU)
        
        output_path = STACK_JSON_DIR / f"{Path(img).stem}.json"
        output_path.write_text(json.dumps({
            "image": img,
            "predictions": final
        }, indent=2))
    
    print(f"\n[STACK] Wrote predictions to {STACK_JSON_DIR}")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


# Run evaluation metrics.
def cmd_evaluate(args):
    import csv
    import numpy as np
    
    from src.config import STACK_JSON_DIR_DEFAULT, LABELS_TEST_DIR
    from src.evaluation import (
        read_stacked_jsons, prepare_gt, filter_predictions,
        evaluate, compute_froc,
    )
    from src.utils import xywhn_to_xyxy
    
    print(f"Loading predictions from {args.stack_dir}")
    preds = read_stacked_jsons(args.stack_dir, use_kept_warn_only=args.use_kept_warn_only)
    
    print(f"Loading ground truth from {args.labels_dir}")
    gts = prepare_gt(args.labels_dir)
    
    if args.only_labelled:
        labelled = set(gts.keys())
        before = len(preds)
        preds = {k: v for k, v in preds.items() if k in labelled}
        print(f"Filtered to labelled images: {before} -> {len(preds)}")
    
    class_map = None
    if args.class_iou_map is not None:
        try:
            raw = json.loads(args.class_iou_map.read_text())
            class_map = {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
        except Exception as e:
            print(f"Warning: failed to load class_iou_map: {e}")
    
    if args.min_conf > 0.0 or (args.neighbor_iou > 0.0 and args.min_neighbors > 0):
        preds = filter_predictions(
            preds, min_conf=args.min_conf,
            neighbor_iou=args.neighbor_iou, min_neighbors=args.min_neighbors
        )
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    results = evaluate(
        preds, gts,
        iou_th=args.iou,
        tolerance_px=args.tolerance_px,
        tolerance_rel=args.tolerance_rel,
        debug=args.debug,
        adaptive_iou=args.adaptive_iou,
        class_iou_map=class_map,
        pred_expand_px=args.expand_pred_px,
        pred_expand_rel=args.expand_pred_rel,
        pred_expand_mode=args.expand_pred_mode,
    )
    
    if args.map_ious is not None:
        try:
            iou_list = [float(x) for x in args.map_ious.split(",")]
        except Exception:
            iou_list = []
        
        if iou_list:
            print(f"\nComputing mAP across IoUs: {iou_list}")
            per_class_map_collect = {}
            
            for iou_val in iou_list:
                res_iou = evaluate(
                    preds, gts, iou_th=float(iou_val),
                    tolerance_px=args.tolerance_px, debug=False,
                    adaptive_iou=False, class_iou_map=class_map,
                )
                for cl, stats in res_iou["per_class"].items():
                    per_class_map_collect.setdefault(cl, []).append(stats.get("AP", 0.0))
            
            per_class_map = {
                cl: float(np.mean(vals)) if vals else 0.0
                for cl, vals in per_class_map_collect.items()
            }
            
            print("\nPer-class mAP (mean over IoUs):")
            for cl in sorted(per_class_map.keys()):
                print(f"  class {cl}: {per_class_map[cl]:.4f}")
            
            if args.map_save is not None:
                args.map_save.parent.mkdir(parents=True, exist_ok=True)
                with args.map_save.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["class", "mAP"])
                    for cl in sorted(per_class_map.keys()):
                        w.writerow([cl, f"{per_class_map[cl]:.6f}"])
    
    if args.froc:
        try:
            fppi_points = tuple(float(x) for x in args.froc_fppi.split(","))
        except Exception:
            fppi_points = (0.25, 0.5, 1, 2, 4, 8)
        
        print("\n" + "=" * 60)
        print("FROC ANALYSIS")
        print("=" * 60)
        
        froc_res = compute_froc(preds, gts, iou_th=float(args.iou), fppi_points=fppi_points)
        
        print(f"Images: {froc_res['n_images']}, Total GT: {froc_res['total_gt']}")
        for pp, sens in froc_res["points"].items():
            print(f"  FPPI={pp:.3f} -> Sensitivity={sens:.4f}")
        
        if args.froc_save is not None:
            args.froc_save.parent.mkdir(parents=True, exist_ok=True)
            with args.froc_save.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["FPPI", "Sensitivity"])
                for x, y in zip(froc_res["fppi"], froc_res["sens"]):
                    w.writerow([f"{x:.6f}", f"{y:.6f}"])
    
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.save_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "AP", "n_gt", "n_det", "best_f1", "best_prec", "best_rec"])
            for cl in results["classes"]:
                r = results["per_class"].get(cl)
                if r is None:
                    continue
                w.writerow([
                    cl, f"{r['AP']:.6f}", r["n_gt"], r["n_det"],
                    f"{r['best_f1']:.6f}", f"{r['best_prec']:.6f}", f"{r['best_rec']:.6f}"
                ])
        print(f"Saved CSV to {args.save_csv}")
    
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON to {args.save_json}")


# Build argument parser with subcommands.
def build_parser():
    parser = argparse.ArgumentParser(
        description="Jointwise Model Development CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    p_prepare = subparsers.add_parser("prepare", help="Prepare YOLO dataset from CSV")
    p_prepare.add_argument("--png-dir", type=Path, default=None,
                           help="Source PNG directory")
    p_prepare.add_argument("--csv", type=Path, default=None,
                           help="Annotations CSV path")
    p_prepare.add_argument("--output", type=Path, default=None,
                           help="Output directory for YOLO dataset")
    p_prepare.add_argument("--train-frac", type=float, default=0.5,
                           help="Training set fraction (default: 0.5)")
    p_prepare.add_argument("--seed", type=int, default=42,
                           help="Random seed for splitting")
    p_prepare.add_argument("--force", action="store_true",
                           help="Force rebuild (delete existing)")
    p_prepare.set_defaults(func=cmd_prepare)
    
    p_augment = subparsers.add_parser("augment", help="Augment training data")
    p_augment.add_argument("--split", choices=["train", "val", "test"], default="train",
                           help="Split to augment")
    p_augment.add_argument("--target", type=int, default=20000,
                           help="Target images per class")
    p_augment.add_argument("--seed", type=int, default=42,
                           help="Random seed")
    p_augment.add_argument("--dry-run", action="store_true",
                           help="Preview plan without writing")
    p_augment.set_defaults(func=cmd_augment)
    
    p_train = subparsers.add_parser("train", help="Train ensemble models")
    p_train.add_argument("--skip-ga", action="store_true",
                         help="Skip GA, reuse cached hyperparameters")
    p_train.add_argument("--force-ga", action="store_true",
                         help="Force GA even if cached")
    p_train.add_argument("--ga-disable", action="store_true",
                         help="Disable GA entirely")
    p_train.add_argument("--reuse-models", action="store_true",
                         help="Skip training, use existing checkpoints")
    p_train.add_argument("--reuse-meta", action="store_true",
                         help="Skip meta-learner training")
    p_train.add_argument("--precision-conf", type=float, default=None,
                         help="High-precision confidence threshold")
    p_train.add_argument("--iou-match", type=float, default=None,
                         help="IoU match threshold for meta-learner")
    p_train.add_argument("--eval-thresholds", action="store_true",
                         help="Evaluate confidence thresholds")
    p_train.set_defaults(func=cmd_train)
    
    p_eval = subparsers.add_parser("evaluate", help="Evaluate predictions")
    p_eval.add_argument("--stack-dir", type=Path, default=None,
                        help="Directory with stacked JSON predictions")
    p_eval.add_argument("--labels-dir", type=Path, default=None,
                        help="Directory with YOLO label files")
    p_eval.add_argument("--save-csv", type=Path, default=None,
                        help="Save metrics to CSV")
    p_eval.add_argument("--save-json", type=Path, default=None,
                        help="Save metrics to JSON")
    p_eval.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold (default: 0.5)")
    p_eval.add_argument("--min-conf", type=float, default=0.0,
                        help="Minimum confidence threshold")
    p_eval.add_argument("--neighbor-iou", type=float, default=0.0,
                        help="IoU for neighbor counting")
    p_eval.add_argument("--min-neighbors", type=int, default=0,
                        help="Minimum neighbors required")
    p_eval.add_argument("--tolerance-px", type=float, default=0.0,
                        help="Pixel tolerance for center matching")
    p_eval.add_argument("--tolerance-rel", type=float, default=0.0,
                        help="Relative tolerance")
    p_eval.add_argument("--use-kept-warn-only", action="store_true",
                        help="Include only kept/warned predictions")
    p_eval.add_argument("--only-labelled", action="store_true",
                        help="Keep only images with GT")
    p_eval.add_argument("--adaptive-iou", action="store_true",
                        help="Use adaptive IoU based on size")
    p_eval.add_argument("--class-iou-map", type=Path, default=None,
                        help="JSON with per-class IoU bounds")
    p_eval.add_argument("--expand-pred-px", type=float, default=0.0,
                        help="Expand predictions by pixels")
    p_eval.add_argument("--expand-pred-rel", type=float, default=0.0,
                        help="Expand predictions by relative amount")
    p_eval.add_argument("--expand-pred-mode", type=str, default="expand_gt",
                        choices=["expand_gt", "expand_pred", "intersect"],
                        help="Mode for prediction expansion")
    p_eval.add_argument("--froc", action="store_true",
                        help="Compute FROC curve")
    p_eval.add_argument("--froc-fppi", type=str, default="0.25,0.5,1,2,4,8",
                        help="FPPI points for FROC")
    p_eval.add_argument("--froc-save", type=Path, default=None,
                        help="Save FROC curve to CSV")
    p_eval.add_argument("--map-ious", type=str, default=None,
                        help="Comma-separated IoUs for mAP")
    p_eval.add_argument("--map-save", type=Path, default=None,
                        help="Save per-class mAP to CSV")
    p_eval.add_argument("--debug", action="store_true",
                        help="Print debug information")
    p_eval.set_defaults(func=cmd_evaluate)
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "evaluate":
        from src.config import STACK_JSON_DIR_DEFAULT, LABELS_TEST_DIR
        if args.stack_dir is None:
            args.stack_dir = STACK_JSON_DIR_DEFAULT
        if args.labels_dir is None:
            args.labels_dir = LABELS_TEST_DIR
    
    args.func(args)


if __name__ == "__main__":
    main()
