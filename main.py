#!/usr/bin/env python3
"""
Traditional Train + GA Tuning + Stacking Ensemble (Ultralytics YOLO, single file)

What it does
------------
1) For each detector family (Xception / ResNeXt / DenseNet / EfficientNet):
   - GA (DEAP) hyperparameter search (3 epochs per eval)
   - Final train with early stopping, up to 50 epochs

2) Stacking meta-learner:
   - Build validation-time IoU groups across the 4 models’ predictions
   - For each group, features = [conf_model1, conf_model2, conf_model3, conf_model4]
     (missing model → 0)
   - Label a group positive if it IoU-matches any GT (≥0.5), else negative
   - Train LogisticRegression to map features → fused confidence
   - At test-time, predict fused confidences for groups and write JSONs

Requirements
------------
- ultralytics
- timm
- deap
- scikit-learn
- torch
- numpy

Dataset layout
--------------
datasets/yolo/
  images/{train,val,test}/*.(png|jpg|jpeg)
  labels/{train,val,test}/*.txt  # YOLO format

Notes
-----
- Keeps your custom FPN backbones via timm; Detect head is the standard Ultralytics head.
- Early stopping uses Ultralytics 'patience' parameter inside the 50-epoch cap.
- GA config is small by default for practicality; tweak GA_POP/GA_GEN if you want.
"""

from __future__ import annotations
import os, sys, json, random, math, pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------- Optional installs (quiet) ----------
def _pip_install(pkg: str):
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
    except Exception as e:
        print(f"[WARN] Failed to pip install {pkg}: {e}")

try:
    import timm  # type: ignore
except Exception:
    _pip_install("timm")
    import timm  # type: ignore

try:
    from deap import base, creator, tools, algorithms  # type: ignore
except Exception:
    _pip_install("deap")
    from deap import base, creator, tools, algorithms  # type: ignore

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:
    _pip_install("scikit-learn")
    from sklearn.linear_model import LogisticRegression  # type: ignore

import numpy as np
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

# Train/val/test image size and batch
IMG_SIZE = 320
BATCH = 16

# Final training (per family)
FINAL_EPOCHS = 50
FINAL_PATIENCE = 10   # early stopping patience (epochs with no val improvement)

# GA hyperparameter tuning (per family)
GA_ENABLE = True
GA_EVAL_EPOCHS = 3              # each GA evaluation trains this many epochs
GA_POP = 10                     # population size
GA_GEN = 6                      # number of generations
GA_SEED = 0
GA_TUNE_ON_VAL = True           # if True, GA trains on the validation split (train=val, val=val)

# IoU thresholds
IOU_MATCH = 0.5                 # for positive/negative group labels
GROUP_IOU = 0.5                 # to group boxes across models during stacking
NMS_IOU = 0.5                   # simple final NMS per class in stacking

# Save / Project
PROJECT_NAME = "classic_train_stack"
ROOT_DIR = Path.cwd()
TMP_DIR = ROOT_DIR / ".tmp_classic"
MODELS_DIR = TMP_DIR / "models"
PROJECT_DIR = ROOT_DIR / "runs" / PROJECT_NAME
DATA_YAML = TMP_DIR / "dataset.yaml"

META_MODEL_PATH = PROJECT_DIR / "meta_stack.pkl"    # stores LogisticRegression
STACK_JSON_DIR  = PROJECT_DIR / "stacked_test_json" # outputs

for p in (TMP_DIR, MODELS_DIR, PROJECT_DIR, STACK_JSON_DIR):
    p.mkdir(parents=True, exist_ok=True)


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
        if not (images_dir/s).exists(): raise FileNotFoundError(f"Missing images/{s}")
        if not (labels_dir/s).exists(): raise FileNotFoundError(f"Missing labels/{s}")
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

def ensure_val_as_train_dataset_yaml(images_dir: Path, labels_dir: Path, out_yaml: Path, nc: int) -> Path:
    """Create a dataset YAML where both train and val point to the validation split.
    Useful for GA evaluation when you want HP search to use only the validation data.
    """
    for s in ("val","test"):
        if not (images_dir/s).exists(): raise FileNotFoundError(f"Missing images/{s}")
        if s == "val" and not (labels_dir/s).exists(): raise FileNotFoundError(f"Missing labels/{s}")
    root_posix = images_dir.parent.as_posix()
    data = f"""
path: "{root_posix}"
train: images/val
val: images/val
test: images/test
names: {json.dumps(CLASS_NAMES)}
nc: {nc}
""".strip()
    out_yaml.write_text(data+"\n"); return out_yaml

def list_images(dir_path: Path) -> List[str]:
    imgs = sorted([*dir_path.glob("*.png"), *dir_path.glob("*.jpg"), *dir_path.glob("*.jpeg")])
    return [str(p) for p in imgs]


# =========================
# Custom Blocks (FPN + Timm backbones)
# =========================
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

# Index module: select particular output from list (Ultralytics expects channel metadata)
class Take(nn.Module):
    def __init__(self, c1=None, c2=None, i=0, *args, **kwargs):
        super().__init__()
        # Accept both YAML arg styles:
        #   - Index, [256, idx]  (legacy where an extra channel hint precedes idx)
        #   - Index, [idx]
        if 'i' in kwargs:
            self.i = int(kwargs['i'])
        elif len(args) > 0:
            # use last positional arg as index when provided via list
            try:
                self.i = int(args[-1])
            except Exception:
                self.i = int(i) if i is not None else 0
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
class Index(Take): pass
ytasks.Take = Take
ytasks.Index = Index


# =========================
# Training / Evaluation
# =========================
BASE_AUG = dict(
    mosaic=0.2, mixup=0.05, fliplr=0.2,
    hsv_h=0.003, hsv_s=0.15, hsv_v=0.15,
    degrees=5.0, translate=0.05, scale=0.2, shear=2.5,
    perspective=0.0005, copy_paste=0.05, multi_scale=False
)

def _merge_train_args(hp: Dict) -> Tuple[Dict, Dict]:
    aug_keys = set(BASE_AUG.keys())
    trainer, aug = {}, BASE_AUG.copy()
    for k, v in (hp or {}).items():
        if k in aug_keys:
            aug[k] = v
        else:
            trainer[k] = v
    return trainer, aug

def _trainer_defaults(over: Dict, *, patience: int) -> Dict:
    base = {
        "optimizer": "AdamW",
        "lr0": 2.5e-3,
        "lrf": 0.1,
        "momentum": 0.93,
        "weight_decay": 5e-4,
        "patience": patience,
        "seed": 0,
        "pretrained": True,
        "deterministic": False,
        "val": True
    }
    return {**base, **(over or {})}

def train_model(model_yaml: Path, data_yaml: Path, epochs: int, patience: int,
                run_name: str, hp: Optional[Dict]=None) -> Tuple[YOLO, Path]:
    y = YOLO(str(model_yaml))
    t_over, aug_over = _merge_train_args(hp or {})
    kwargs = _trainer_defaults(t_over, patience=patience)
    res = y.train(
        data=str(data_yaml), imgsz=IMG_SIZE, epochs=epochs, batch=BATCH,
        workers=min(8, os.cpu_count() or 8), device=device_str(),
        project=str(PROJECT_DIR), name=run_name, exist_ok=True, plots=False,
        **aug_over, **kwargs
    )
    return y, Path(res.save_dir)

def evaluate_map(y: YOLO, split="val", data_yaml: Optional[Path]=None) -> Dict[str,float]:
    r = y.val(data=str(data_yaml or DATA_YAML), split=split, imgsz=IMG_SIZE, device=device_str(),
              plots=False, save_json=False, verbose=False)
    out={}
    try:
        out = {"map50-95": float(r.box.map), "map50": float(r.box.map50), "map75": float(r.box.map75)}
    except Exception:
        pass
    return out


# =========================
# GA Hyperparameter Tuning (DEAP)
# =========================
# Discrete choices for simplicity/robustness
HP_SPACE = {
    "optimizer": ["AdamW", "SGD"],
    "lr0":       [1e-3, 2.5e-3, 5e-3],
    "lrf":       [0.1, 0.2, 0.4],
    "momentum":  [0.90, 0.93, 0.95],
    "weight_decay": [5e-5, 1e-4, 5e-4],
    "mosaic":    [0.0, 0.1, 0.2, 0.3],
    "mixup":     [0.0, 0.05, 0.10, 0.15],
    "fliplr":    [0.1, 0.2, 0.3],
    "hsv_s":     [0.10, 0.15, 0.20],
    "hsv_v":     [0.10, 0.15, 0.20],
    "degrees":   [0.0, 5.0, 10.0],
    "scale":     [0.1, 0.2, 0.3],
    "translate": [0.05, 0.10],
    "shear":     [2.5, 5.0],
    "copy_paste":[0.0, 0.05, 0.10],
}

HP_KEYS = list(HP_SPACE.keys())
HP_SIZES = [len(HP_SPACE[k]) for k in HP_KEYS]

def _gene_to_hp(gene: List[int]) -> Dict:
    hp = {}
    for idx, k in enumerate(HP_KEYS):
        hp[k] = HP_SPACE[k][gene[idx] % len(HP_SPACE[k])]
    return hp

def run_ga_for_family(family: str, yml_path: Path) -> Dict:
    if not GA_ENABLE:
        print(f"[GA] Disabled, using BASE_AUG defaults for {family}.")
        return {}

    random.seed(GA_SEED)
    np.random.seed(GA_SEED)

    # Choose dataset YAML for GA
    ga_data_yaml = DATA_YAML
    if GA_TUNE_ON_VAL:
        ga_yaml_path = TMP_DIR / f"ga_val_{family}.yaml"
        ensure_val_as_train_dataset_yaml(IMAGES_DIR, LABELS_DIR, ga_yaml_path, nc=NUM_CLASSES)
        ga_data_yaml = ga_yaml_path
        print(f"[GA] {family}: training and validating on the validation split for HP search -> {ga_data_yaml}")

    # Create DEAP structures
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    for i, size in enumerate(HP_SIZES):
        toolbox.register(f"gene_{i}", random.randrange, size)

    def _make_individual():
        return creator.Individual([toolbox.__getattribute__(f"gene_{i}")() for i in range(len(HP_SIZES))])

    toolbox.register("individual", _make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        # Map gene -> hp
        hp = _gene_to_hp(individual)
        # Quick train for GA_EVAL_EPOCHS
        run_name = f"{family}_GAeval"
        y, _ = train_model(yml_path, ga_data_yaml, epochs=GA_EVAL_EPOCHS, patience=max(1, GA_EVAL_EPOCHS-1),
                           run_name=run_name, hp=hp)
        metrics = evaluate_map(y, split="val", data_yaml=ga_data_yaml)
        # Diagnostics: small prediction summary on a tiny val sample
        try:
            val_dir = IMAGES_DIR / "val"
            sample_imgs = list_images(val_dir)[:24]
            pred_map = predict_boxes(y, sample_imgs, conf=0.001, iou=0.5)
            total_preds = sum(len(v) for v in pred_map.values())
            per_img = (total_preds / max(1, len(sample_imgs))) if sample_imgs else 0.0
            # class histogram
            hist = {c: 0 for c in range(NUM_CLASSES)}
            for vs in pred_map.values():
                for cl, _, _ in vs:
                    if cl in hist: hist[cl] += 1
            print(f"[GA][{family}] preds/img={per_img:.2f} total={total_preds} class_hist={hist} map50={metrics.get('map50',0.0):.4f}")
        except Exception as _diag_e:
            pass
        score = float(metrics.get("map50", 0.0))
        # We maximize mAP50
        return (score,)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[0]*len(HP_SIZES),
                     up=[s-1 for s in HP_SIZES], indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=GA_POP)
    hof = tools.HallOfFame(1)

    # Simple EA
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=GA_GEN, halloffame=hof, verbose=True)

    best_gene = list(hof[0])
    best_hp = _gene_to_hp(best_gene)
    print(f"[GA] {family} best hp: {best_hp}")
    return best_hp


# =========================
# Prediction / Stacking utils
# =========================
def yolo_label_path_for_image(img_path: Path, labels_root_split: Path) -> Path:
    return labels_root_split / (img_path.stem + ".txt")

def read_yolo_labels(label_path: Path) -> List[Tuple[int,float,float,float,float]]:
    if not label_path.exists(): return []
    out=[]
    for ln in label_path.read_text().splitlines():
        ps=ln.strip().split()
        if len(ps)>=5:
            try:
                cl=int(float(ps[0])); cx,cy,w,h = map(float, ps[1:5])
                out.append((cl,cx,cy,w,h))
            except: pass
    return out

def xywhn_to_xyxy(cx,cy,w,h,W,H):
    x1=(cx - w/2.0)*W; y1=(cy - h/2.0)*H
    x2=(cx + w/2.0)*W; y2=(cy + h/2.0)*H
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

def predict_boxes(model: YOLO, image_paths: List[str], conf=0.001, iou=0.5) -> Dict[str, List[Tuple[int, float, Tuple[float,float,float,float]]]]:
    """
    Returns dict: image_path -> list of (cls, conf, (x1,y1,x2,y2))
    """
    out = {}
    B = 16
    for i in range(0, len(image_paths), B):
        batch = image_paths[i:i+B]
        res = model.predict(source=batch, imgsz=IMG_SIZE, conf=conf, iou=iou,
                            device=device_str(), verbose=False)
        for img, r in zip(batch, res):
            H,W = r.orig_shape[:2]
            preds=[]
            try:
                boxes=r.boxes.xyxy.cpu().numpy()
                confs=r.boxes.conf.cpu().numpy()
                clses=r.boxes.cls.cpu().numpy().astype(int)
                for (x1,y1,x2,y2), cf, cl in zip(boxes, confs, clses):
                    preds.append((int(cl), float(cf), (float(x1),float(y1),float(x2),float(y2))))
            except Exception:
                pass
            out[img] = preds
    return out

def group_boxes_across_models(per_model: List[Dict[str, List[Tuple[int,float,Tuple[float,float,float,float]]]]],
                              image_paths: List[str]) -> Dict[str, List[Dict]]:
    """
    For each image, cluster predictions per class by IoU > GROUP_IOU across models.
    Returns:
      image -> list of groups, each group: {
        'cls': int,
        'members': [(model_idx, conf, (x1,y1,x2,y2)) ...],
        'box': average box over members (simple mean),
        'feat': 4-dim vector (max conf per model; 0 if absent)
      }
    """
    M = len(per_model)  # number of models
    out = {}
    for img in image_paths:
        groups=[]
        # Flatten all predictions with model index
        flat=[]
        for m_idx, pdict in enumerate(per_model):
            for (cl, cf, b) in pdict.get(img, []):
                flat.append((m_idx, cl, cf, b))

        # Per-class greedy grouping
        for cl in sorted(set([f[1] for f in flat])):
            items = [(m,cf,b) for (m, ccl, cf, b) in flat if ccl==cl]
            used=[False]*len(items)
            for i,(m_i,cf_i,b_i) in enumerate(items):
                if used[i]: continue
                group=[(m_i,cf_i,b_i)]; used[i]=True
                # grow
                for j,(m_j,cf_j,b_j) in enumerate(items):
                    if used[j]: continue
                    if iou_xyxy(b_i, b_j) > GROUP_IOU:
                        used[j]=True; group.append((m_j,cf_j,b_j))
                # aggregate
                xs=[g[2][0] for g in group]; ys=[g[2][1] for g in group]
                xe=[g[2][2] for g in group]; ye=[g[2][3] for g in group]
                avg_box = (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(xe)), float(np.mean(ye)))
                feat = [0.0]*M
                for (m,cf,_) in group:
                    feat[m] = max(feat[m], float(cf))
                groups.append({"cls":cl, "members":group, "box":avg_box, "feat":feat})
        out[img] = groups
    return out

def build_meta_labels(groups: Dict[str, List[Dict]], labels_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make (X, y) for meta-learner:
      - y=1 if group box IoU-matches any GT of same class at IOU_MATCH, else 0
      - X is the 4-dim feature vector (max conf per model in group)
    """
    X=[]; y=[]
    for img, gs in groups.items():
        H,W = None, None  # not needed for GT; GT stored in normalized
        # Load GT boxes
        gts_n = read_yolo_labels(yolo_label_path_for_image(Path(img), labels_root))
        # convert to XYXY with image size… but we don't have H,W here.
        # Workaround: we’ll compute IoU in normalized coordinates by normalizing group box as well.
        # To do so, we need image dim. We can get it with cv2 or PIL, but to avoid new deps, use torch via Ultralytics?
        # Simpler: read size via PIL (available in base Python installs).
        try:
            from PIL import Image
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            # if cannot read, skip labeling for this image
            continue

        gts=[]
        for (cl,cx,cy,w,h) in gts_n:
            gts.append((cl, *xywhn_to_xyxy(cx,cy,w,h,W,H)))

        for g in gs:
            cl = g["cls"]
            bx = g["box"]
            # match?
            is_pos = False
            for (gcl, gx1,gy1,gx2,gy2) in gts:
                if gcl != cl: continue
                if iou_xyxy(bx, (gx1,gy1,gx2,gy2)) >= IOU_MATCH:
                    is_pos = True; break
            X.append(g["feat"])
            y.append(1 if is_pos else 0)
    if len(X)==0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def apply_meta_on_groups(groups: Dict[str, List[Dict]], meta: LogisticRegression) -> Dict[str, List[Dict]]:
    out={}
    for img, gs in groups.items():
        preds=[]
        if len(gs)==0:
            out[img]=preds; continue
        X = np.array([g["feat"] for g in gs], dtype=np.float32)
        # Predict probability of positive (class 1)
        ps = meta.predict_proba(X)[:,1] if hasattr(meta, "predict_proba") else meta.decision_function(X)
        for g, p in zip(gs, ps):
            preds.append({"cls": g["cls"], "box": g["box"], "conf": float(p)})
        out[img] = preds
    return out

def nms_by_class(preds: List[Dict], iou_thr: float) -> List[Dict]:
    out=[]
    for cl in sorted(set(p["cls"] for p in preds)):
        items=[p for p in preds if p["cls"]==cl]
        items=sorted(items, key=lambda x:x["conf"], reverse=True)
        kept=[]
        for e in items:
            b=e["box"]
            if all(iou_xyxy(b,k["box"])<=iou_thr for k in kept):
                kept.append(e)
        out.extend(kept)
    return out


# =========================
# Pipeline per family
# =========================
def write_yaml_text(name: str, text: str) -> Path:
    p = MODELS_DIR / f"{name}.yaml"
    p.write_text(text)
    return p

def train_family(family: str, yaml_text: str) -> Tuple[str, Path, Dict]:
    yml = write_yaml_text(family, yaml_text)
    # GA tuning
    best_hp = run_ga_for_family(family, yml)
    # Final train with ES (50 epochs max)
    run_name = f"{family}_final"
    y, run_dir = train_model(yml, DATA_YAML, epochs=FINAL_EPOCHS, patience=FINAL_PATIENCE,
                             run_name=run_name, hp=best_hp)
    ckpt = (run_dir / "weights" / "best.pt")
    if not ckpt.exists():
        ckpt = (run_dir / "weights" / "last.pt")
    print(f"[{family}] Final checkpoint -> {ckpt}")
    return family, ckpt, best_hp


# =========================
# Main
# =========================
def main():
    # Dataset sanity
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        raise RuntimeError("IMAGES_DIR and LABELS_DIR must exist with train/val/test subfolders.")
    ensure_dataset_yaml(IMAGES_DIR, LABELS_DIR, DATA_YAML, nc=NUM_CLASSES)
    clear_dataset_caches(LABELS_DIR)

    # Families
    families = [
        ("xception",    XCEPTION_YAML),
        ("resnext",     RESNEXT_YAML),
        ("densenet",    DENSENET_YAML),
        ("efficientnet",EFFICIENTNET_YAML),
    ]

    # ---- Train all families (GA -> Final) ----
    trained = []
    for name, yml in families:
        fam, ckpt, hp = train_family(name, yml)
        trained.append((fam, ckpt, hp))

    # ---- Load final YOLO models for predictions ----
    yolos = []
    for fam, ckpt, _ in trained:
        yolos.append((fam, YOLO(str(ckpt))))

    # ---- Build validation meta-dataset ----
    val_imgs = list_images(IMAGES_DIR / "val")
    per_model_preds_val = []
    for fam, model in yolos:
        print(f"[STACK] Predicting val with {fam}...")
        per_model_preds_val.append(predict_boxes(model, val_imgs, conf=0.001, iou=0.5))

    val_groups = group_boxes_across_models(per_model_preds_val, val_imgs)
    X_val, y_val = build_meta_labels(val_groups, LABELS_DIR / "val")
    if X_val.shape[0] == 0:
        print("[STACK] No validation groups found; meta-learner will default to uniform averaging.")
        meta = None
    else:
        print(f"[STACK] Meta training set: X={X_val.shape}, positives={int(y_val.sum())}, negatives={int((y_val==0).sum())}")
        meta = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")
        meta.fit(X_val, y_val)
        META_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(META_MODEL_PATH, "wb") as f:
            pickle.dump(meta, f)
        print(f"[STACK] Saved meta model -> {META_MODEL_PATH}")

    # ---- Apply stacking on TEST ----
    test_imgs = list_images(IMAGES_DIR / "test")
    per_model_preds_test = []
    for fam, model in yolos:
        print(f"[STACK] Predicting test with {fam}...")
        per_model_preds_test.append(predict_boxes(model, test_imgs, conf=0.001, iou=0.5))

    test_groups = group_boxes_across_models(per_model_preds_test, test_imgs)

    if meta is None:
        # Fallback: uniform score = average of per-model max confs in group
        fused = {}
        for img, gs in test_groups.items():
            preds=[]
            for g in gs:
                score = float(sum(g["feat"])) / max(1, len(g["feat"]))
                preds.append({"cls": g["cls"], "box": g["box"], "conf": score})
            fused[img] = preds
    else:
        fused = apply_meta_on_groups(test_groups, meta)

    # Simple class-wise NMS and write JSONs
    for img in test_imgs:
        preds = fused.get(img, [])
        final = nms_by_class(preds, iou_thr=NMS_IOU)
        (STACK_JSON_DIR / f"{Path(img).stem}.json").write_text(json.dumps({
            "image": img,
            "predictions": final
        }, indent=2))
    print(f"[STACK] Wrote stacked predictions -> {STACK_JSON_DIR}")


if __name__ == "__main__":
    main()
