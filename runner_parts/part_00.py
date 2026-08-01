# ================================================================
# RimGraph-DG — Reproducible Single-Cell Colab Experiment Runner
# Repository: https://github.com/AzizulHakim00/Glaucomma
# Run this ONE cell from top to bottom.
# ================================================================

# ---------------------------- 0. INSTALL ----------------------------
import os, sys, subprocess, warnings, json, math, random, time, shutil, hashlib, platform, re, gc
from pathlib import Path

def _pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

_pip_install([
    "timm>=1.0.9", "albumentations>=1.4.14", "kagglehub>=0.3.4",
    "optuna>=4.0.0", "openpyxl>=3.1.5"
])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from IPython.display import display, clear_output, Markdown

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss, log_loss
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ---------------------------- 1. CONFIG ----------------------------
CFG = {
    # Data
    "kaggle_dataset": "arnavjain1/glaucoma-datasets",
    "manual_data_dir": "",  # Optional: e.g. /content/drive/MyDrive/GlaucomaData
    "sources": ["ORIGA", "REFUGE", "G1020"],
    "image_size": 384,
    "num_workers": 2,

    # Experiment
    "project_name": "RimGraph_DG",
    "run_name": time.strftime("run_%Y%m%d_%H%M%S"),
    "seed": 2029,
    "fold_targets": ["ORIGA", "REFUGE", "G1020"],
    "val_fraction": 0.20,
    "epochs": 25,
    "warmup_epochs": 2,
    "patience": 7,
    "batch_size": 4,
    "grad_accum": 1,
    "mixed_precision": True,
    "resume": True,
    "fast_dev_run": False,  # True = tiny smoke test, not paper results

    # Model
    "backbone": "convnext_tiny.fb_in22k_ft_in1k",
    "pretrained": True,
    "feature_dim": 256,
    "graph_hidden": 128,
    "graph_heads": 4,
    "dropout": 0.25,
    "num_sectors": 12,

    # Optimisation
    "lr": 2e-4,
    "backbone_lr_scale": 0.15,
    "weight_decay": 1e-4,
    "label_smoothing": 0.02,
    "focal_gamma": 2.0,
    "lambda_seg": 1.0,
    "lambda_vcdr": 0.25,
    "lambda_cons": 0.20,
    "lambda_domain": 0.08,
    "lambda_proto": 0.08,
    "max_grad_norm": 1.0,

    # Optional nested source-only tuning (never touches held-out test source)
    "run_optuna": False,
    "optuna_trials": 8,
    "tune_epochs": 3,

    # Output
    "drive_folder": "Glaucomma_RimGraphDG",
    "save_every_epoch": True,
    "n_visual_examples": 6,
}

# The one-cell notebook may define GLAUCOMMA_OVERRIDES before executing this runner.
CFG.update(globals().get("GLAUCOMMA_OVERRIDES", {}))

if CFG["fast_dev_run"]:
    CFG.update({"epochs": 2, "patience": 2, "batch_size": 4, "run_optuna": False})

# ---------------------------- 2. DRIVE + PATHS ----------------------------
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

COLAB_ROOT = Path("/content/Glaucomma_runs") if IN_COLAB else Path.cwd() / "Glaucomma_runs"
DRIVE_ROOT = (Path("/content/drive/MyDrive") / CFG["drive_folder"]) if IN_COLAB else (Path.cwd() / "DriveMirror" / CFG["drive_folder"])
RUN_ID = CFG["run_name"]
LOCAL_RUN = COLAB_ROOT / RUN_ID
DRIVE_RUN = DRIVE_ROOT / RUN_ID
for p in [LOCAL_RUN, DRIVE_RUN]:
    p.mkdir(parents=True, exist_ok=True)

class ArtifactStore:
    def __init__(self, local_root, drive_root):
        self.local_root = Path(local_root)
        self.drive_root = Path(drive_root)

    def dirs(self, rel=""):
        a, b = self.local_root / rel, self.drive_root / rel
        a.mkdir(parents=True, exist_ok=True); b.mkdir(parents=True, exist_ok=True)
        return a, b

    def mirror(self, local_path, rel=None):
        local_path = Path(local_path)
        rel = Path(rel) if rel else local_path.relative_to(self.local_root)
        target = self.drive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_dir():
            if target.exists(): shutil.rmtree(target)
            shutil.copytree(local_path, target)
        else:
            shutil.copy2(local_path, target)
        return target

    def save_json(self, obj, rel):
