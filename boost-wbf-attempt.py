#!/usr/bin/env python3
"""
Leakage-Safe Sequential Ensembles per Architecture + Final Voting/WBF
---------------------------------------------------------------------

Implements the diagram:
    [Boost Xception]
    [Boost ResNeXt]     >  => Calibrate => Weighted WBF on test
    [Boost DenseNet]
    [Boost EfficientNet]

Key features
- OOF (out-of-fold) residual computation inside the TRAIN split to avoid leakage
- Optional residual_pool mode (small holdout from train used only to compute residuals)
- Optional KD pseudo-label pretrain between stages (set KD_PRE_EPOCHS=0 to turn off)
- Detect-only warm start (re-uses Detect head if shapes match)
- Per-family mAP on validation → temperature scaling → final weighted WBF fusion
- NEW: Hyperparameter tuning (random search) per family before boosting
- NEW: Round-level early stopping (no fixed number of rounds required)

Requires: ultralytics, timm, torch
Optional: scikit-learn (for KFold). If not present, falls back to residual_pool mode.

Dataset layout:
  IMAGES_DIR/{train,val,test}/*.(png|jpg|jpeg)
  LABELS_DIR/{train,val,test}/*.txt  (YOLO labels)

NOTE: Keep classes consistent across families. All families use the same Detect head spec.
"""

from __future__ import annotations
import os, sys, json, random, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------- Optional installs ----------
def _pip_install(pkg: str):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

try:
    import timm  # type: ignore
except Exception:
    _pip_install("timm")
    import timm  # type: ignore

try:
    from sklearn.model_selection import KFold  # type: ignore
    _HAS_SK = True
except Exception:
    _HAS_SK = False

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn import tasks as ytasks

# =========================
# User Configuration
# =========================
IMAGES_DIR = Path("datasets/yolo/images").expanduser().resolve()
LABELS_DIR = Path("datasets/yolo/labels").expanduser().resolve()

NUM_CLASSES = 2
CLASS_NAMES = ["ACL Tear", "Meniscus Tear"]

# --------- Boosting rounds (no fixed number; early stopping controls it) ----------
MAX_ROUNDS_PER_FAMILY = 1000      # hard cap for safety; set high
ROUND_ES_PATIENCE = 3             # stop if no mAP50 improvement for these many rounds
ROUND_MIN_DELTA = 0.01            # minimum mAP50 gain to reset patience (e.g., 0.1%)

# --------- Epoch-level early stopping inside each Ultralytics train ----------
TRAIN_PATIENCE_EPOCHS = 10        # early-stop training if no improvement for N epochs

# Residual computation (anti-leakage)
RESIDUAL_MODE = "oof"             # "oof" or "pool"
K_FOLDS = 5                       # if RESIDUAL_MODE == "oof"
RESIDUAL_POOL_FRAC = 0.15         # if RESIDUAL_MODE == "pool"
RESIDUAL_CLIP = (0.2, 5.0)
RESIDUAL_SHRINK = 0.4             # shrinkage towards 1.0

# KD / Pseudo-labeling
KD_PRE_EPOCHS = 0                 # set 0 to disable KD
KD_CONF = 0.05
KD_IOU = 0.6
KD_MAX_DETS = 300
KD_OVERWRITE_EMPTY = True

# Train & eval (default baseline; tuning can override many of these)
IMG_SIZE = 320
EPOCHS_STAGE0 = 1                 # first round epochs
EPOCHS_STAGE_T = 1                # subsequent round epochs
BATCH = 16
OPTIMIZER = "AdamW"
LR0 = 2.5e-3
LRF = 0.1
MOMENTUM = 0.93
WEIGHT_DECAY = 5e-4

AUG_BASE = dict(
    mosaic=0.2, mixup=0.05, fliplr=0.2,
    hsv_h=0.003, hsv_s=0.15, hsv_v=0.15,
    degrees=5.0, translate=0.05, scale=0.2, shear=2.5,
    perspective=0.0005, copy_paste=0.05, multi_scale=False
)

# --------- Hyperparameter tuning (random search) ----------
DO_TUNE = True
TUNE_TRIALS = 6                    # number of random search trials per family
TUNE_EPOCHS = 1                    # epochs per tuning trial
TUNE_SEED = 0

# Calibration
CALIBRATE_TEMPERATURE = True

# Final fusion on test
RUN_FINAL_ENSEMBLE = True
ENSEMBLE_CONF = 0.001
ENSEMBLE_GROUP_IOU = 0.5
ENSEMBLE_NMS_IOU = 0.5
MAX_TEST_IMAGES = 0                # 0 = all

# Smoke test controls (set SMOKE_TEST_ONLY=1 to run only the smoke test and exit)
SMOKE_TEST = os.getenv("SMOKE_TEST", "1") == "1"
SMOKE_TEST_ONLY = os.getenv("SMOKE_TEST_ONLY", "0") == "1"

# Paths
PROJECT_NAME = "per_family_boost_then_vote"
TMP_DIR = Path.cwd() / ".tmp_boost_vote"
MODELS_DIR = TMP_DIR / "models"
LISTS_DIR = TMP_DIR / "lists"
PROJECT_DIR = Path.cwd() / "runs" / PROJECT_NAME
DATA_YAML = TMP_DIR / "dataset.yaml"
ROUND_DATA_YAML = TMP_DIR / "dataset.round.yaml"

for p in (TMP_DIR, MODELS_DIR, LISTS_DIR, PROJECT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# =========================
# Model YAMLs (shared head)
# =========================
XCEPTION_YAML = (
    "task: detect\n"
    f"nc: {NUM_CLASSES}\n"
    "ch: 1\n"
    "backbone:\n"
    "  - [-1, 1, XceptionFPN, []]\n"
    "  - [0, 1, Index, [256, 0]]\n"
    "  - [0, 1, Index, [256, 1]]\n"
    "  - [0, 1, Index, [256, 2]]\n"
    "head:\n"
    "  - [[1, 2, 3], 1, Detect, [nc]]\n"
)
RESNEXT_YAML = (
    "task: detect\n"
    f"nc: {NUM_CLASSES}\n"
    "ch: 1\n"
    "backbone:\n"
    "  - [-1, 1, ResNeXtFPN, []]\n"
    "  - [0, 1, Index, [256, 0]]\n"
    "  - [0, 1, Index, [256, 1]]\n"
    "  - [0, 1, Index, [256, 2]]\n"
    "head:\n"
    "  - [[1, 2, 3], 1, Detect, [nc]]\n"
)
DENSENET_YAML = (
    "task: detect\n"
    f"nc: {NUM_CLASSES}\n"
    "ch: 1\n"
    "backbone:\n"
    "  - [-1, 1, DenseNetFPN, []]\n"
    "  - [0, 1, Index, [256, 0]]\n"
    "  - [0, 1, Index, [256, 1]]\n"
    "  - [0, 1, Index, [256, 2]]\n"
    "head:\n"
    "  - [[1, 2, 3], 1, Detect, [nc]]\n"
)
EFFICIENTNET_YAML = (
    "task: detect\n"
    f"nc: {NUM_CLASSES}\n"
    "ch: 1\n"
    "backbone:\n"
    "  - [-1, 1, EfficientNetFPN, []]\n"
    "  - [0, 1, Index, [256, 0]]\n"
    "  - [0, 1, Index, [256, 1]]\n"
    "  - [0, 1, Index, [256, 2]]\n"
    "head:\n"
    "  - [[1, 2, 3], 1, Detect, [nc]]\n"
)

# =========================
# Custom Blocks
# =========================
class Take(nn.Module):
    def __init__(self, c1=None, c2=None, i=0, *args, **kwargs):
        """
        Ultralytics parses custom modules as (c1, c2, *args). This Take block selects
        one tensor from a list produced by a previous layer. Expose correct output
        channels via self.c2 so downstream layers (e.g., Detect) are built with
        the right in_channels.
        """
        super().__init__()
        if 'i' in kwargs:
            self.i = int(kwargs['i'])
        else:
            self.i = int(i) if i is not None else 0

        out_ch = None
        if isinstance(c1, (list, tuple)) and len(c1) > 0:
            idx = min(max(0, self.i), len(c1) - 1)
            try:
                out_ch = int(c1[idx])
            except Exception:
                out_ch = None
        elif isinstance(c1, (int, float)) and c1:
            out_ch = int(c1)

        self.c2 = out_ch if (out_ch and out_ch > 0) else 256
        self._warned = False

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"Take[{self.i}] expects list/tuple")
        if self.i >= len(x):
            if not self._warned:
                print(f"[Take] idx {self.i} > {len(x)-1}; clamping")
                self._warned = True
            return x[-1]
        return x[self.i]

# Alias Index -> Take so YAML "Index" works
class Index(Take):
    pass

ytasks.Take = Take
ytasks.Index = Index

class _ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, act=True):
        if p is None: p = (k - 1)//2
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False), nn.BatchNorm2d(out_ch)]
        if act: layers.append(nn.SiLU(inplace=False))
        super().__init__(*layers)

def _ensure_no_inplace(m):
    if isinstance(m, (nn.SiLU, nn.ReLU)): m.inplace = False

class _FPN(nn.Module):
    def __init__(self, c3, c4, c5, out=256):
        super().__init__()
        self.l3 = _ConvBNAct(c3, out, k=1, act=False)
        self.l4 = _ConvBNAct(c4, out, k=1, act=False)
        self.l5 = _ConvBNAct(c5, out, k=1, act=False)
        self.o3 = _ConvBNAct(out, out, k=3)
        self.o4 = _ConvBNAct(out, out, k=3)
        self.o5 = _ConvBNAct(out, out, k=3)

    def forward(self, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.o3(p3), self.o4(p4), self.o5(p5)]

def _timm_feats(name, in_chans=1, pretrained=True):
    m = timm.create_model(name, features_only=True, in_chans=in_chans,
                          pretrained=pretrained, out_indices=(2,3,4), act_layer=nn.SiLU)
    for mod in m.modules(): _ensure_no_inplace(mod)
    return m

class XceptionFPN(nn.Module):
    def __init__(self, name="xception41", pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = _timm_feats(name, in_chans=1, pretrained=pretrained)
        c3,c4,c5 = self.backbone.feature_info.channels()
        self.neck = _FPN(c3,c4,c5,out_channels); self.c2 = [out_channels]*3
    def forward(self,x):
        if x.shape[1]!=1: x=x.mean(1,keepdim=True)
        c3,c4,c5 = self.backbone(x)
        p3,p4,p5 = self.neck(c3.contiguous(), c4.contiguous(), c5.contiguous())
        return [p3,p4,p5]

class DenseNetFPN(nn.Module):
    def __init__(self, name="densenet121", pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = _timm_feats(name, in_chans=1, pretrained=pretrained)
        c3,c4,c5 = self.backbone.feature_info.channels()
        self.neck = _FPN(c3,c4,c5,out_channels); self.c2 = [out_channels]*3
    def forward(self,x):
        if x.shape[1]!=1: x=x.mean(1,keepdim=True)
        c3,c4,c5 = self.backbone(x)
        p3,p4,p5 = self.neck(c3.contiguous(), c4.contiguous(), c5.contiguous())
        return [p3,p4,p5]

class ResNeXtFPN(nn.Module):
    def __init__(self, name="resnext50_32x4d", pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = _timm_feats(name, in_chans=1, pretrained=pretrained)
        c3,c4,c5 = self.backbone.feature_info.channels()
        self.neck = _FPN(c3,c4,c5,out_channels); self.c2 = [out_channels]*3
    def forward(self,x):
        if x.shape[1]!=1: x=x.mean(1,keepdim=True)
        c3,c4,c5 = self.backbone(x)
        p3,p4,p5 = self.neck(c3.contiguous(), c4.contiguous(), c5.contiguous())
        return [p3,p4,p5]

class EfficientNetFPN(nn.Module):
    def __init__(self, name="efficientnet_b0", pretrained=True, out_channels=256):
        super().__init__()
        self.backbone = _timm_feats(name, in_chans=1, pretrained=pretrained)
        c3,c4,c5 = self.backbone.feature_info.channels()
        self.neck = _FPN(c3,c4,c5,out_channels); self.c2 = [out_channels]*3
    def forward(self,x):
        if x.shape[1]!=1: x=x.mean(1,keepdim=True)
        c3,c4,c5 = self.backbone(x)
        p3,p4,p5 = self.neck(c3.contiguous(), c4.contiguous(), c5.contiguous())
        return [p3,p4,p5]

ytasks.XceptionFPN = XceptionFPN
ytasks.DenseNetFPN = DenseNetFPN
ytasks.ResNeXtFPN = ResNeXtFPN
ytasks.EfficientNetFPN = EfficientNetFPN

# =========================
# Helpers
# =========================
def device_str():
    return 0 if torch.cuda.is_available() else "cpu"

def clear_dataset_caches(labels_root: Path) -> None:
    for split in ("train", "val", "test"):
        cf = labels_root / f"{split}.cache"
        if cf.exists():
            try:
                cf.unlink(); print(f"[DATA] Removed cache: {cf}")
            except Exception:
                pass

def ensure_dataset_yaml(images_dir: Path, labels_dir: Path, out_yaml: Path, nc: int) -> Path:
    for s in ("train","val","test"):
        if not (images_dir/s).exists(): raise FileNotFoundError(f"Missing images {s}")
        if not (labels_dir/s).exists(): raise FileNotFoundError(f"Missing labels {s}")
    root_posix = images_dir.parent.as_posix()
    data = f"""
path: "{root_posix}"
train: images/train
val: images/val
test: images/test
names: {json.dumps(CLASS_NAMES)}
nc: {nc}
""".strip()
    out_yaml.write_text(data+"\n"); return out_yaml

def write_data_yaml_copy(base_yaml: Path, out_yaml: Path, train_override: Optional[str]=None) -> Path:
    lines = base_yaml.read_text().splitlines()
    if train_override is not None:
        rep=[]; found=False
        q=f'"{Path(train_override).as_posix()}"'
        for ln in lines:
            if ln.strip().startswith("train:"):
                rep.append(f"train: {q}"); found=True
            else: rep.append(ln)
        if not found: rep.append(f"train: {q}")
        lines=rep
    out_yaml.write_text("\n".join(lines)+"\n")
    return out_yaml

def build_list(dir_path: Path) -> List[str]:
    imgs = sorted([*dir_path.glob("*.png"), *dir_path.glob("*.jpg"), *dir_path.glob("*.jpeg")])
    return [str(p) for p in imgs]

def save_list(paths: List[str], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as f:
        for p in paths: f.write(p+"\n")

def build_weighted_list(all_paths: List[str], weights: Dict[str,float], out_txt: Path, dup_floor=1) -> Path:
    N=len(all_paths); wsum=sum(weights.get(p,0.0) for p in all_paths) or 1.0
    counts={p:max(dup_floor, int(round(weights.get(p,0.0)/wsum*N))) for p in all_paths}
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as f:
        for p in all_paths:
            for _ in range(counts[p]): f.write(p+"\n")
    return out_txt

# ---------- Detect-only transfer (robust) ----------
def _find_detect_module(yolo_obj: YOLO):
    for m in yolo_obj.model.modules():
        if m.__class__.__name__.lower().endswith("detect"):
            return m
    return None

def load_detect_only(dst: YOLO, src_ckpt: Path) -> int:
    """
    Copy only the Detect head weights from a trained checkpoint into an
    untrained/other-architecture model, as long as the head shapes match.
    This is resilient to Ultralytics' internal key naming.
    """
    try:
        ysrc = YOLO(str(src_ckpt))
    except Exception as e:
        print(f"[TRANSFER] Could not open src ckpt: {e}")
        return 0

    src_det = _find_detect_module(ysrc)
    dst_det = _find_detect_module(dst)
    if src_det is None or dst_det is None:
        print("[TRANSFER] Could not locate Detect module in src and/or dst model.")
        return 0

    src_sd = {k: v.cpu() for k, v in src_det.state_dict().items()}
    dst_sd = dst_det.state_dict()

    to_load = {}
    for k, v in src_sd.items():
        if k in dst_sd and tuple(dst_sd[k].shape) == tuple(v.shape):
            to_load[k] = v

    if not to_load:
        def shape_map(sd): return {k: tuple(v.shape) for k, v in sd.items()}
        print("[TRANSFER] No matching Detect tensors by shape.")
        print("[TRANSFER] src Detect shapes:", shape_map(src_sd))
        print("[TRANSFER] dst Detect shapes:", shape_map(dst_sd))
        return 0

    missing, unexpected = dst_det.load_state_dict(to_load, strict=False)
    n = len(to_load)
    print(f"[TRANSFER] Detect-only tensors loaded: {n} (missing={len(missing)}, unexpected={len(unexpected)})")
    return n

# ---------- Labels / IOU ----------
def yolo_label_path_for_image(img_path: Path, labels_root_split: Path) -> Path:
    return labels_root_split / (img_path.stem + ".txt")

def read_yolo_labels(label_path: Path) -> List[Tuple[int,float,float,float,float]]:
    if not label_path.exists(): return []
    out=[]
    for ln in label_path.read_text().splitlines():
        ps=ln.split()
        if len(ps)>=5:
            try:
                cl=int(float(ps[0])); cx,cy,w,h=map(float,ps[1:5])
                out.append((cl,cx,cy,w,h))
            except: pass
    return out

def xywhn_to_xyxy(cx,cy,w,h):
    x1=cx - w/2.0; y1=cy - h/2.0
    x2=cx + w/2.0; y2=cy + h/2.0
    return x1,y1,x2,y2

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    xi1=max(ax1,bx1); yi1=max(ay1,by1)
    xi2=min(ax2,bx2); yi2=min(ay2,by2)
    iw=max(0.0, xi2-xi1); ih=max(0.0, yi2-yi1)
    inter=iw*ih
    ua=max(0.0, (ax2-ax1))*max(0.0,(ay2-ay1))
    ub=max(0.0, (bx2-bx1))*max(0.0,(by2-by1))
    union=ua+ub-inter
    return inter/union if union>0 else 0.0

def per_image_correct(model: YOLO, img_path: str, labels_dir_train: Path,
                      imgsz=320, iou_thr=0.5, score_thr=0.0) -> bool:
    r = model.predict(source=[img_path], imgsz=imgsz, conf=0.001, iou=0.5,
                      device=device_str(), verbose=False)[0]
    try:
        boxes = r.boxes.xyxy.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
    except Exception:
        boxes=[]; clses=[]; confs=[]
    H,W = r.orig_shape[:2]
    gt = read_yolo_labels(yolo_label_path_for_image(Path(img_path), labels_dir_train))
    gt_xyxy=[]
    for cl,cx,cy,w,h in gt:
        x1n,y1n,x2n,y2n=xywhn_to_xyxy(cx,cy,w,h)
        gt_xyxy.append((cl,x1n*W,y1n*H,x2n*W,y2n*H))
    if len(gt_xyxy)==0:
        return sum(1 for c in confs if c>=score_thr)==0
    for (bx1,by1,bx2,by2), pcl, cf in zip(boxes, clses, confs):
        if cf < score_thr: continue
        for gcl,gx1,gy1,gx2,gy2 in gt_xyxy:
            if gcl==int(pcl) and iou_xyxy((bx1,by1,bx2,by2),(gx1,gy1,gx2,gy2))>=iou_thr:
                return True
    return False

# ---------- KD ----------
def _write_yolo_txt(out_path: Path, cls: int, cx: float, cy: float, w: float, h: float):
    with out_path.open("a") as f:
        f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def _xyxy_to_xywhn(x1,y1,x2,y2,W,H):
    w=max(0.0,x2-x1); h=max(0.0,y2-y1)
    cx=x1 + w/2.0; cy=y1 + h/2.0
    return (cx/W, cy/H, w/W, h/W)

def build_pseudo_labels_from_teacher(teacher_ckpt: Path, train_list_txt: Path, out_labels_dir: Path,
                                     imgsz: int=320, conf: float=0.05, iou: float=0.6,
                                     max_det: int=300, overwrite_empty: bool=True):
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    teacher = YOLO(str(teacher_ckpt))
    imgs = [Path(ln.strip()) for ln in train_list_txt.read_text().splitlines() if ln.strip()]
    if not imgs: return
    B=16
    for i in range(0,len(imgs),B):
        batch=imgs[i:i+B]
        res = teacher.predict(source=[str(p) for p in batch], imgsz=imgsz, conf=conf, iou=iou,
                              device=device_str(), verbose=False, max_det=max_det)
        for img_path, r in zip(batch, res):
            H,W = r.orig_shape[:2]
            out = out_labels_dir / (img_path.stem + ".txt")
            if out.exists(): out.unlink()
            wrote=False
            try:
                boxes=r.boxes.xyxy.cpu().numpy()
                clses=r.boxes.cls.cpu().numpy().astype(int)
                for (x1,y1,x2,y2), cl in zip(boxes, clses):
                    cx,cy,w,h=_xyxy_to_xywhn(x1,y1,x2,y2,W,H)
                    if w<=0 or h<=0: continue
                    _write_yolo_txt(out, int(cl), cx, cy, w, h)
                    wrote=True
            except Exception:
                pass
            if overwrite_empty and not wrote: out.touch()

# ---------- Train / Val ----------
def _merge_train_args(hp: Dict) -> Tuple[Dict, Dict]:
    """Split hp dict into trainer args and augmentation args."""
    aug_keys = set(AUG_BASE.keys())
    trainer, aug = {}, AUG_BASE.copy()
    for k, v in (hp or {}).items():
        if k in aug_keys:
            aug[k] = v
        else:
            trainer[k] = v
    return trainer, aug

def _merge_trainer_defaults(trainer_over: Dict, *, patience: int | None = None) -> Dict:
    """Merge default trainer hyperparameters with overrides safely.

    Ensures we pass each key exactly once to Ultralytics' train(), preventing
    duplicate arguments like 'optimizer'.
    """
    base = {
        "lr0": LR0,
        "lrf": LRF,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER,
        "patience": TRAIN_PATIENCE_EPOCHS if patience is None else patience,
        "seed": 0,
        "pretrained": False,
        "deterministic": False,
    }
    # trainer_overrides take precedence
    return {**base, **(trainer_over or {})}

def train_yolo(model_yaml: Path, data_yaml: Path, epochs: int, name: str, hp: Optional[Dict]=None) -> Tuple[YOLO, Path]:
    y = YOLO(str(model_yaml))
    trainer_over, aug_over = _merge_train_args(hp or {})
    trainer_kwargs = _merge_trainer_defaults(trainer_over, patience=TRAIN_PATIENCE_EPOCHS)
    res = y.train(
        data=str(data_yaml), imgsz=IMG_SIZE, epochs=epochs, batch=BATCH,
        workers=min(8, os.cpu_count() or 8), device=device_str(),
        project=str(PROJECT_DIR), name=name, exist_ok=True, plots=False,
        **aug_over, **trainer_kwargs
    )
    return y, Path(res.save_dir)

def train_existing(y: YOLO, data_yaml: Path, epochs: int, name: str, hp: Optional[Dict]=None) -> Path:
    trainer_over, aug_over = _merge_train_args(hp or {})
    trainer_kwargs = _merge_trainer_defaults(trainer_over, patience=TRAIN_PATIENCE_EPOCHS)
    res = y.train(
        data=str(data_yaml), imgsz=IMG_SIZE, epochs=epochs, batch=BATCH,
        workers=min(8, os.cpu_count() or 8), device=device_str(),
        project=str(PROJECT_DIR), name=name, exist_ok=True, plots=False,
        **aug_over, **trainer_kwargs
    )
    return Path(res.save_dir)

def evaluate_map(y: YOLO, split="val") -> Dict[str,float]:
    r = y.val(data=str(DATA_YAML), split=split, imgsz=IMG_SIZE, device=device_str(),
              plots=False, save_json=False, verbose=False)
    out={}
    try:
        out = {"map50-95": float(r.box.map), "map50": float(r.box.map50), "map75": float(r.box.map75)}
    except Exception:
        pass
    return out

# ---------- Residuals ----------
def compute_residuals_pool(model_yaml: Path, ckpt_prev: Optional[Path], pool_paths: List[str],
                           labels_dir: Path, hp: Optional[Dict]=None) -> Dict[str,float]:
    y = YOLO(str(model_yaml))
    if ckpt_prev and ckpt_prev.exists():
        try: load_detect_only(y, ckpt_prev)
        except Exception: pass
    trainer_over, aug_over = _merge_train_args(hp or {})
    trainer_kwargs = _merge_trainer_defaults(trainer_over, patience=max(1, TRAIN_PATIENCE_EPOCHS//2))
    _ = y.train(data=str(DATA_YAML), imgsz=IMG_SIZE, epochs=1, batch=BATCH,
                workers=min(8, os.cpu_count() or 8), device=device_str(),
                project=str(PROJECT_DIR), name="pool_fit", exist_ok=True, plots=False,
                **aug_over, **trainer_kwargs)
    residuals={}
    for p in pool_paths:
        ok = per_image_correct(y, p, labels_dir, imgsz=IMG_SIZE, iou_thr=0.5, score_thr=0.0)
        residuals[p] = 1.0 - float(ok) + 1e-3
    return residuals

def residuals_to_weights(residuals: Dict[str,float], clip=(0.2,5.0), shrink=0.4) -> Dict[str,float]:
    vals=list(residuals.values()); m=sum(vals)/max(1,len(vals))
    w={}
    for p,r in residuals.items():
        base = r/max(m,1e-8)
        base = min(clip[1], max(clip[0], base))
        w[p] = shrink*base + (1.0 - shrink)*1.0
    s=sum(w.values()) or 1.0
    return {k:v/s for k,v in w.items()}

# ---------- Calibration ----------
def fit_temperature(y: YOLO, val_paths: List[str]) -> float:
    if not CALIBRATE_TEMPERATURE: return 1.0
    temps=[0.75,1.0,1.25,1.5]
    best_t, best=1.0, float("inf")
    import numpy as np
    for t in temps:
        score=0.0; n=0
        for p in val_paths[:min(128,len(val_paths))]:
            r=y.predict(source=[p], imgsz=IMG_SIZE, conf=0.001, iou=0.5, device=device_str(), verbose=False)[0]
            try:
                confs = r.boxes.conf.cpu().numpy()
            except Exception:
                confs = []
            if len(confs)==0: continue
            confs=np.clip(np.array(confs,dtype=float),1e-6,1-1e-6)
            logits=np.log(confs/(1-confs))/t
            cal=1/(1+np.exp(-logits))
            score += float(cal.mean()); n+=1
        score = score/max(1,n)
        if score < best: best, best_t = score, t
    return best_t

def apply_T(conf: float, T: float) -> float:
    c = min(1-1e-6, max(1e-6, conf))
    logit = math.log(c/(1-c))/T
    return 1/(1+math.exp(-logit))

# ---------- Final WBF ----------
def final_weighted_wbf(family_heads: List[Tuple[str, Path, float, float]],
                       images: List[Path], out_dir: Path,
                       conf=0.001, iou_group=0.5, nms_iou=0.5, imgsz=320):
    """
    family_heads: list of (name, ckpt_path, weight, temperature)
    weight is typically validation mAP (or function of it).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    models=[(name, YOLO(str(ckpt)), w, T) for (name, ckpt, w, T) in family_heads]

    def iou_fast(a,b):
        xa1,ya1,xa2,ya2=a; xb1,yb1,xb2,yb2=b
        xi1=max(xa1,xb1); yi1=max(ya1,yb1)
        xi2=min(xa2,xb2); yi2=min(ya2,yb2)
        inter=max(0,xi2-xi1)*max(0,yi2-yi1)
        area_a=max(0,xa2-xa1)*max(0,ya2-ya1)
        area_b=max(0,xb2-xb1)*max(0,yb2-yb1)
        union=area_a+area_b-inter
        return inter/union if union>0 else 0.0

    for img in images:
        per_class: Dict[int, List[Tuple[Tuple[float,float,float,float], float]]] = {}
        for (fname, model, w, T) in models:
            r = model.predict(source=[str(img)], imgsz=imgsz, conf=conf, iou=0.5,
                              device=device_str(), verbose=False)[0]
            try:
                boxes=r.boxes.xyxy.cpu().numpy()
                confs=r.boxes.conf.cpu().numpy()
                clses=r.boxes.cls.cpu().numpy().astype(int)
                for b,c,cl in zip(boxes, confs, clses):
                    cT = apply_T(float(c), T)
                    per_class.setdefault(int(cl), []).append((tuple(b.tolist()), float(cT)*float(w)))
            except Exception:
                continue

        # group by IoU, weighted average
        ensembled=[]
        for cl, items in per_class.items():
            used=[False]*len(items)
            for i,(bi, wi) in enumerate(items):
                if used[i]: continue
                group=[(bi,wi)]; used[i]=True
                for j in range(i+1,len(items)):
                    if used[j]: continue
                    bj, wj = items[j]
                    if iou_fast(bi, bj) > iou_group:
                        used[j]=True; group.append((bj,wj))
                ws=sum(w for _,w in group) or 1.0
                bx=sum(b[0]*w for b,w in group)/ws
                by=sum(b[1]*w for b,w in group)/ws
                ex=sum(b[2]*w for b,w in group)/ws
                ey=sum(b[3]*w for b,w in group)/ws
                cf=min(1.0, ws/len(models))  # normalized weight as score
                ensembled.append({"class": int(cl), "box": [bx,by,ex,ey], "conf": cf})

        # class-wise simple NMS
        final=[]
        for cl in set(e["class"] for e in ensembled):
            cand=[e for e in ensembled if e["class"]==cl]
            cand=sorted(cand, key=lambda x:x["conf"], reverse=True)
            kept=[]
            for e in cand:
                b=e["box"]
                if all(iou_fast(b,k["box"])<=nms_iou for k in kept):
                    kept.append(e)
            final.extend(kept)

        (out_dir / f"{img.stem}.json").write_text(json.dumps({"image": str(img), "predictions": final}, indent=2))

# =========================
# Smoke test
# =========================
def run_smoke_test() -> bool:
    """Run a minimal end-to-end check: tiny train -> val -> predict.

    Returns True if succeeds without exception.
    """
    try:
        print("[SMOKE] Starting smoke test...")
        if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
            print("[SMOKE] Dataset folders missing; skipping.")
            return False
        ensure_dataset_yaml(IMAGES_DIR, LABELS_DIR, DATA_YAML, nc=NUM_CLASSES)
        clear_dataset_caches(LABELS_DIR)

        # Use Xception family for quick check
        yml = MODELS_DIR / "xception.smoke.yaml"
        yml.write_text(XCEPTION_YAML)

        # Tiny train list
        train_paths = build_list(IMAGES_DIR/"train")[: min(8, max(0, len(build_list(IMAGES_DIR/"train"))))]
        if not train_paths:
            print("[SMOKE] No training images found; skipping.")
            return False
        smoke_txt = LISTS_DIR / "smoke_train.txt"
        save_list(train_paths, smoke_txt)
        write_data_yaml_copy(DATA_YAML, ROUND_DATA_YAML, train_override=str(smoke_txt))

        # Quick train (1 epoch)
        y, _ = train_yolo(yml, ROUND_DATA_YAML, epochs=1, name="smoke", hp={
            "mosaic": 0.0, "mixup": 0.0, "fliplr": 0.1
        })

        # Validation
        metrics = evaluate_map(y, split="val")
        print(f"[SMOKE] val metrics: {metrics}")

        # Predict on one test image if available
        test_list = build_list(IMAGES_DIR/"test")
        if test_list:
            r = y.predict(source=[test_list[0]], imgsz=IMG_SIZE, conf=0.001, iou=0.5,
                          device=device_str(), verbose=False)[0]
            n_preds = int(getattr(getattr(r, 'boxes', []), 'shape', [0])[0]) if hasattr(r, 'boxes') else 0
            print(f"[SMOKE] predictions on one test image: {n_preds}")

        print("[SMOKE] Completed successfully.")
        return True
    except Exception as e:
        print(f"[SMOKE] Failed: {e}")
        return False

# =========================
# Hyperparameter tuning
# =========================
def _sample_hp(rng: random.Random) -> Dict:
    """Randomly sample a training+aug hyperparameter set."""
    hp = {
        # trainer-level
        "optimizer": rng.choice(["AdamW", "SGD"]),
        "lr0": rng.choice([1e-3, 2.5e-3, 5e-3]),
        "lrf": rng.choice([0.1, 0.2, 0.4, 0.5]),
        "momentum": rng.choice([0.9, 0.93, 0.95]),
        "weight_decay": rng.choice([5e-5, 1e-4, 5e-4]),
        # aug-level
        "mosaic": rng.choice([0.0, 0.1, 0.2, 0.3]),
        "mixup": rng.choice([0.0, 0.05, 0.15]),
        "fliplr": rng.choice([0.1, 0.2, 0.3]),
        "hsv_s": rng.choice([0.10, 0.15, 0.2]),
        "hsv_v": rng.choice([0.10, 0.15, 0.2]),
        "degrees": rng.choice([0.0, 5.0, 10.0]),
        "scale": rng.choice([0.1, 0.2, 0.3]),
        "translate": rng.choice([0.05, 0.10]),
        "shear": rng.choice([2.5, 5.0]),
        "copy_paste": rng.choice([0.0, 0.05, 0.1]),
    }
    return hp

def tune_family(family_name: str, yml: Path, train_list_txt: Path, trials: int, epochs: int) -> Dict:
    """Run short random-search tuning and return best hp dict."""
    rng = random.Random(TUNE_SEED)
    best_hp, best_map = None, -1.0

    # Build a tuning data YAML that points to the same full val/test and uses the given train list
    tune_yaml = TMP_DIR / f"tune_{family_name}.yaml"
    root_posix = IMAGES_DIR.parent.as_posix()
    tune_yaml.write_text(
        "path: \"" + root_posix + "\"\n"
        "train: \"" + Path(train_list_txt).as_posix() + "\"\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {json.dumps(CLASS_NAMES)}\n"
        f"nc: {NUM_CLASSES}\n"
    )

    for t in range(trials):
        hp = _sample_hp(rng)
        model = YOLO(str(yml))
        # quick train
        run_name = f"{family_name}_tune_{t+1}"
        ALLOWED_TRAIN_KEYS = {"mosaic","mixup","degrees","translate","scale","shear","flipud","fliplr","erasing","hsv_h","hsv_s","hsv_v","patience","cos_lr","optimizer","lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr","close_mosaic","box","cls","dfl","label_smoothing","val","device","project","name","save","exist_ok","workers","seed"}
        train_kwargs = {k: v for k, v in {**AUG_BASE, **hp}.items() if k in ALLOWED_TRAIN_KEYS}
        _ = model.train(
            data=str(tune_yaml),
            imgsz=IMG_SIZE,
            epochs=epochs,
            batch=BATCH,
            **train_kwargs
        )
        # evaluate
        metrics = evaluate_map(model, split="val")
        m50 = float(metrics.get("map50", 0.0))
        print(f"[TUNE {family_name}] trial {t+1}/{trials}: mAP50={m50:.4f} hp={hp}")
        if m50 > best_map:
            best_map, best_hp = m50, hp

    print(f"[TUNE {family_name}] best mAP50={best_map:.4f} with hp={best_hp}")
    return best_hp or {}

# =========================
# Family boosting runner
# =========================
def run_family(family_name: str, yaml_text: str) -> Tuple[Path, float, float]:
    """
    Returns: (family_head_ckpt, validation_weight, temperature)
    - family_head_ckpt: Path to checkpoint to use for final fusion
    - validation_weight: scalar weight (map50) for voting
    - temperature: fitted T for this family
    """
    # write model yaml
    yml = MODELS_DIR / f"{family_name}.yaml"
    yml.write_text(yaml_text)

    # prepare lists
    all_train = build_list(IMAGES_DIR/"train")
    save_list(all_train, LISTS_DIR / f"train_all_{family_name}.txt")
    val_list = build_list(IMAGES_DIR/"val")

    # initial uniform sampling list for round 1
    cur_txt = LISTS_DIR / f"{family_name}_r1.txt"
    build_weighted_list(all_train, {p: 1.0/len(all_train) for p in all_train}, cur_txt)
    write_data_yaml_copy(DATA_YAML, ROUND_DATA_YAML, train_override=str(cur_txt))

    # ---- Hyperparameter tuning (optional) ----
    best_hp = {}
    if DO_TUNE and TUNE_TRIALS > 0 and TUNE_EPOCHS > 0:
        best_hp = tune_family(family_name, yml, cur_txt, TUNE_TRIALS, TUNE_EPOCHS)

    # stage loop with round-level early stopping
    ckpts: List[Path] = []
    best_round_map = -1.0
    best_round_ckpt: Optional[Path] = None
    rounds_no_improve = 0

    t = 1
    while t <= MAX_ROUNDS_PER_FAMILY and rounds_no_improve < ROUND_ES_PATIENCE:
        # --- build (possibly reweighted) list already stored in ROUND_DATA_YAML
        # student
        y = YOLO(str(yml))
        if ckpts:
            try: load_detect_only(y, ckpts[-1])
            except Exception: pass

        # KD pretrain (optional)
        if KD_PRE_EPOCHS>0 and ckpts:
            kd_labels = TMP_DIR / f"kd_{family_name}_r{t}"
            kd_yaml   = TMP_DIR / f"kd_{family_name}_r{t}.yaml"
            build_pseudo_labels_from_teacher(ckpts[-1], cur_txt, kd_labels,
                                             imgsz=IMG_SIZE, conf=KD_CONF, iou=KD_IOU,
                                             max_det=KD_MAX_DETS, overwrite_empty=KD_OVERWRITE_EMPTY)
            # swap labels
            orig = LABELS_DIR / "train"; backup = LABELS_DIR / "train.gt.bak"
            if not backup.exists():
                print(f"[KD] Backup GT -> {backup}")
                orig.rename(backup); orig.mkdir(parents=True, exist_ok=True)
            try:
                for p in kd_labels.glob("*.txt"):
                    (orig / p.name).write_text(p.read_text())
                # KD YAML (Windows-safe)
                root_posix = IMAGES_DIR.parent.as_posix()
                train_list_posix = Path(cur_txt).as_posix()
                kd_yaml.write_text(
                    "path: \"" + root_posix + "\"\n"
                    "train: \"" + train_list_posix + "\"\n"
                    "val: images/val\n"
                    "test: images/test\n"
                    f"names: {json.dumps(CLASS_NAMES)}\n"
                    f"nc: {NUM_CLASSES}\n"
                )
                kd_name = f"{family_name}_r{t}_KD"
                _ = train_existing(y, kd_yaml, KD_PRE_EPOCHS, kd_name, hp=best_hp)
            finally:
                for p in (LABELS_DIR/"train").glob("*.txt"): p.unlink(missing_ok=True)
                (LABELS_DIR/"train").rmdir()
                backup.rename(LABELS_DIR/"train")

        # normal train
        epochs = EPOCHS_STAGE0 if t==1 else EPOCHS_STAGE_T
        run_dir = train_existing(y, ROUND_DATA_YAML, epochs, f"{family_name}_r{t}", hp=best_hp)
        ckpt = (Path(run_dir)/"weights"/"best.pt")
        if not ckpt.exists(): ckpt = (Path(run_dir)/"weights"/"last.pt")
        ckpts.append(ckpt)

        # evaluate round
        round_metrics = evaluate_map(y, split="val")
        round_map50 = float(round_metrics.get("map50", 0.0))
        print(f"[{family_name}] Round {t} mAP50={round_map50:.4f}")

        # early stopping on rounds
        if round_map50 > best_round_map + ROUND_MIN_DELTA:
            best_round_map = round_map50
            best_round_ckpt = ckpt
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        # compute residuals for next stage (anti-leakage) and build next list
        if t >= MAX_ROUNDS_PER_FAMILY or rounds_no_improve >= ROUND_ES_PATIENCE:
            break
        if RESIDUAL_MODE=="oof" and _HAS_SK:
            from sklearn.model_selection import KFold  # local import to avoid top failure
            kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=0)
            residuals = {}
            idxs = list(range(len(all_train)))
            # use current trained y to score heldout folds (approx OOF)
            for _, val_idx in kf.split(idxs):
                for i in val_idx:
                    pth = all_train[i]
                    ok = per_image_correct(y, pth, LABELS_DIR/"train", imgsz=IMG_SIZE, iou_thr=0.5, score_thr=0.0)
                    residuals[pth] = 1.0 - float(ok) + 1e-3
            weights = residuals_to_weights(residuals, clip=RESIDUAL_CLIP, shrink=RESIDUAL_SHRINK)
        else:
            n_pool = max(1, int(len(all_train)*RESIDUAL_POOL_FRAC))
            pool = random.sample(all_train, n_pool)
            residuals = compute_residuals_pool(yml, ckpts[-1], pool, LABELS_DIR/"train", hp=best_hp)
            base = {p:1.0 for p in all_train}
            for p, r in residuals.items(): base[p]=r
            weights = residuals_to_weights(base, clip=RESIDUAL_CLIP, shrink=RESIDUAL_SHRINK)

        # build next round list and override in ROUND_DATA_YAML
        t += 1
        cur_txt = LISTS_DIR / f"{family_name}_r{t}.txt"
        build_weighted_list(all_train, weights, cur_txt)
        write_data_yaml_copy(DATA_YAML, ROUND_DATA_YAML, train_override=str(cur_txt))

    # choose family head = best across rounds
    assert best_round_ckpt is not None, "No round produced a checkpoint."
    yhead = YOLO(str(best_round_ckpt))
    val_metrics = evaluate_map(yhead, split="val")
    w = float(val_metrics.get("map50", 0.0))
    T = fit_temperature(yhead, val_list) if CALIBRATE_TEMPERATURE else 1.0
    print(f"[{family_name}] BEST map50={w:.4f}, T={T:.2f}, ckpt={best_round_ckpt.name}")
    return best_round_ckpt, w, T

# =========================
# Main
# =========================
def main():
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise RuntimeError("IMAGES_DIR and LABELS_DIR must exist with train/val/test subfolders.")
    ensure_dataset_yaml(IMAGES_DIR, LABELS_DIR, DATA_YAML, nc=NUM_CLASSES)
    clear_dataset_caches(LABELS_DIR)

    # Always run an initial smoke test (can be disabled via SMOKE_TEST=0)
    if SMOKE_TEST:
        ok = run_smoke_test()
        if SMOKE_TEST_ONLY:
            print("[MAIN] SMOKE_TEST_ONLY=1 set; exiting after smoke test.")
            return
        if not ok:
            print("[MAIN] Smoke test did not complete; continuing, but errors may occur.")

    # write model yamls once
    (MODELS_DIR/"xception.yaml").write_text(XCEPTION_YAML)
    (MODELS_DIR/"resnext.yaml").write_text(RESNEXT_YAML)
    (MODELS_DIR/"densenet.yaml").write_text(DENSENET_YAML)
    (MODELS_DIR/"efficientnet.yaml").write_text(EFFICIENTNET_YAML)

    # Run each family
    families = [
        ("Xception", XCEPTION_YAML),
        ("ResNeXt", RESNEXT_YAML),
        ("DenseNet", DENSENET_YAML),
        ("EfficientNet", EFFICIENTNET_YAML),
    ]
    family_heads: List[Tuple[str, Path, float, float]] = []
    for name, yml in families:
        ckpt, weight, T = run_family(name.lower(), yml)
        family_heads.append((name, ckpt, weight, T))

    # Final ensemble on test
    if RUN_FINAL_ENSEMBLE:
        test_imgs = [Path(p) for p in build_list(IMAGES_DIR/"test")]
        if MAX_TEST_IMAGES>0: test_imgs = test_imgs[:MAX_TEST_IMAGES]
        out_dir = PROJECT_DIR / "final_ensemble"
        s = sum(w for _,_,w,_ in family_heads) or 1.0
        norm_heads = [(n, c, w/s, T) for (n,c,w,T) in family_heads]
        final_weighted_wbf(norm_heads, test_imgs, out_dir,
                           conf=ENSEMBLE_CONF, iou_group=ENSEMBLE_GROUP_IOU,
                           nms_iou=ENSEMBLE_NMS_IOU, imgsz=IMG_SIZE)
        print(f"[FINAL] Ensemble JSONs -> {out_dir}")

if __name__ == "__main__":
    main()