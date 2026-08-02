# ================================================================
# RimGraph-DG V4 — scientifically staged, reproducible Colab runner
# Repository: https://github.com/AzizulHakim00/Glaucomma
# This runner is assembled and executed from ONE Colab cell.
# ================================================================

import os, sys, subprocess, warnings, json, math, random, time, shutil, hashlib, platform, re, gc
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


def _ensure_packages():
    checks = {
        "timm": "timm>=1.0.9",
        "kagglehub": "kagglehub>=0.3.4",
        "optuna": "optuna>=4.0.0",
        "openpyxl": "openpyxl>=3.1.5",
    }
    missing = []
    for module, package in checks.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython.display import display, clear_output, Markdown

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import timm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss, log_loss, jaccard_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

CFG = {
    # Data
    "kaggle_dataset": "arnavjain1/glaucoma-datasets",
    "manual_data_dir": "",
    "sources": ["ORIGA", "REFUGE", "G1020"],
    "fold_targets": ["ORIGA", "REFUGE", "G1020"],
    "image_size": 320,
    "num_workers": 0,
    "canonicalize_laterality": True,
    "exclude_cross_source_duplicates": True,

    # Experiment plan
    "project_name": "RimGraph_DG_V4",
    "run_name": "paper_run_v4",
    "code_revision": "rimgraph-dg-v4-20260802",
    "seed": 2029,
    "seeds": [2029],
    "run_global_baseline": True,
    "run_full_model": True,
    "fast_dev_run": False,
    "resume": True,

    # Training
    "val_fraction": 0.20,
    "baseline_epochs": 12,
    "full_epochs": 30,
    "seg_warmup_epochs": 5,
    "anatomy_warmup_epochs": 10,
    "patience": 8,
    "batch_size": 4,
    "grad_accum": 1,
    "mixed_precision": True,
    "backbone": "convnext_tiny.fb_in22k_ft_in1k",
    "backbone_fallback": "convnext_tiny",
    "pretrained": True,
    "fpn_dim": 128,
    "feature_dim": 256,
    "graph_hidden": 128,
    "graph_heads": 4,
    "dropout": 0.25,
    "num_sectors": 12,
    "roi_output_size": 20,

    # Optimisation
    "lr": 2e-4,
    "backbone_lr_scale": 0.10,
    "weight_decay": 1e-4,
    "focal_gamma": 1.5,
    "lambda_seg": 1.0,
    "lambda_vcdr": 0.20,
    "lambda_cons": 0.10,
    "lambda_domain": 0.04,
    "lambda_proto": 0.04,
    "max_grad_norm": 1.0,

    # Optional source-only tuning after pipeline validation
    "run_optuna": False,
    "optuna_trials": 8,
    "tune_epochs": 4,

    # Evaluation
    "threshold_policy": "youden",
    "bootstrap_samples": 500,
    "n_visual_examples": 4,
    "drive_folder": "Glaucomma_RimGraphDG",
    "save_every_epoch": True,
}
CFG.update(globals().get("GLAUCOMMA_OVERRIDES", {}))
if CFG["fast_dev_run"]:
    CFG.update({
        "baseline_epochs": 1,
        "full_epochs": 2,
        "seg_warmup_epochs": 1,
        "anatomy_warmup_epochs": 1,
        "patience": 2,
        "run_optuna": False,
        "bootstrap_samples": 20,
        "n_visual_examples": 2,
    })

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if IN_COLAB and DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU runtime is required. In Colab choose Runtime > Change runtime type > T4 GPU, then rerun the single cell."
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(int(CFG["seed"]))

COLAB_ROOT = Path("/content/Glaucomma_runs") if IN_COLAB else Path.cwd() / "Glaucomma_runs"
DRIVE_ROOT = (Path("/content/drive/MyDrive") / CFG["drive_folder"]) if IN_COLAB else (Path.cwd() / "DriveMirror" / CFG["drive_folder"])
LOCAL_RUN = COLAB_ROOT / CFG["run_name"]
DRIVE_RUN = DRIVE_ROOT / CFG["run_name"]
for _p in [LOCAL_RUN, DRIVE_RUN]:
    _p.mkdir(parents=True, exist_ok=True)


class ArtifactStore:
    def __init__(self, local_root, drive_root):
        self.local_root = Path(local_root)
        self.drive_root = Path(drive_root)

    def dirs(self, rel=""):
        a, b = self.local_root / rel, self.drive_root / rel
        a.mkdir(parents=True, exist_ok=True)
        b.mkdir(parents=True, exist_ok=True)
        return a, b

    def mirror(self, local_path, rel=None):
        local_path = Path(local_path)
        rel = Path(rel) if rel is not None else local_path.relative_to(self.local_root)
        target = self.drive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(local_path, target)
        else:
            shutil.copy2(local_path, target)
        return target

    def save_json(self, obj, rel):
        lp = self.local_root / rel
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        self.mirror(lp, rel)
        return lp

    def save_df(self, df, rel):
        lp = self.local_root / rel
        lp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(lp, index=False)
        self.mirror(lp, rel)
        return lp

    def save_text(self, text, rel):
        lp = self.local_root / rel
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(text, encoding="utf-8")
        self.mirror(lp, rel)
        return lp

    def save_figure(self, fig, rel, dpi=180):
        lp = self.local_root / rel
        lp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(lp, dpi=dpi, bbox_inches="tight")
        self.mirror(lp, rel)
        return lp


STORE = ArtifactStore(LOCAL_RUN, DRIVE_RUN)


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)


def section(title):
    display(Markdown(f"## {title}"))


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_checkpoint(obj, rel):
    lp = LOCAL_RUN / rel
    atomic_torch_save(obj, lp)
    STORE.mirror(lp, rel)
    return lp


def find_existing(rel):
    lp, dp = LOCAL_RUN / rel, DRIVE_RUN / rel
    if lp.exists(): return lp
    if dp.exists():
        lp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dp, lp)
        return lp
    return None

ENV = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "timm": getattr(timm, "__version__", "unknown"),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "opencv": cv2.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "device": str(DEVICE),
}
STORE.save_json(CFG, "config.json")
STORE.save_json(ENV, "environment.json")

section("RimGraph-DG V4 configuration")
display(pd.DataFrame({"setting": list(CFG.keys()), "value": [str(v) for v in CFG.values()]}).head(30))
