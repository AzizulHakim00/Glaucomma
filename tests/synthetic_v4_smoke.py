import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="rimgraph_v44_ci_"))
os.chdir(TMP)

# Build a small, fully discoverable dataset with deliberately different
# source-domain geometry so duplicate protection is exercised without
# deleting the complete training pool.
for source_i, source in enumerate(["ORIGA", "REFUGE", "G1020"]):
    mask_dir = TMP / source / "Masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    disc_x = 48 + (source_i - 1) * 10
    disc_y = 47 + source_i * 2
    for label, class_name in [(0, "Normal"), (1, "Glaucoma")]:
        image_dir = TMP / source / class_name / "Images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            image = np.zeros((96, 96, 3), np.uint8)
            image[:] = (12 + source_i * 13, 18 + source_i * 9, 24 + source_i * 7)
            cv2.circle(image, (48, 48), 39, (65 + source_i * 9 + label * 20, 78 + label * 18, 90 + source_i * 5), -1)
            cv2.circle(image, (disc_x, disc_y), 13 + 2 * label, (185, 160 - source_i * 12, 125 + source_i * 8), -1)
            cv2.line(image, (8, 18 + source_i * 15), (88, 26 + source_i * 11), (30 + source_i * 35, 55, 75), 2)
            cv2.circle(image, (15 + i * 7, 78 - source_i * 6), 2, (100 + i * 5, 45, 35), -1)
            stem = f"{source}_{class_name}_{i:03d}"
            cv2.imwrite(str(image_dir / f"{stem}.png"), image)

            mask = np.full((96, 96), 255, np.uint8)
            cv2.circle(mask, (disc_x, disc_y), 18, 128, -1)
            cv2.circle(mask, (disc_x, disc_y), 7 + 2 * label, 0, -1)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

GLAUCOMMA_OVERRIDES = {
    "manual_data_dir": str(TMP),
    "sources": ["ORIGA", "REFUGE", "G1020"],
    "fold_targets": ["G1020"],
    "run_name": "ci_v44",
    "code_revision": "rimgraph-dg-v4.4-ci",
    "seeds": [7],
    "fast_dev_run": False,
    "resume": False,
    "pretrained": False,
    "backbone": "resnet18",
    "backbone_fallback": "resnet18",
    "image_size": 64,
    "fpn_dim": 32,
    "feature_dim": 32,
    "graph_hidden": 32,
    "graph_heads": 4,
    "roi_output_size": 8,
    "batch_size": 2,
    "grad_accum": 2,
    "mixed_precision": False,
    "gradient_checkpointing": True,
    "baseline_epochs": 1,
    "full_epochs": 3,
    "seg_warmup_epochs": 1,
    "anatomy_warmup_epochs": 2,
    "patience": 3,
    "bootstrap_samples": 10,
    "n_visual_examples": 2,
    "num_workers": 0,
    "run_optuna": False,
}

raw = "\n".join((ROOT / f"v4_parts/part_{i:02d}.py").read_text() for i in range(7))
code = raw
for patch_name, fn_name in [
    ("runner_patch_v41.py", "apply_v41"),
    ("runner_patch_v42.py", "apply_v42"),
    ("runner_patch_v43.py", "apply_v43"),
    ("runner_patch_v43_autograd.py", "apply_v43_autograd"),
    ("runner_patch_v44_runtime.py", "apply_v44_runtime"),
]:
    namespace = {}
    source = (ROOT / patch_name).read_text()
    exec(compile(source, patch_name, "exec"), namespace, namespace)
    code = namespace[fn_name](code)

compile(code, "rimgraph_dg_v44_ci.py", "exec")
exec(code, globals(), globals())

metrics_path = LOCAL_RUN / "folds/G1020/seed_7/rimgraph_v4/metrics.json"
history_path = LOCAL_RUN / "folds/G1020/seed_7/rimgraph_v4/history.csv"
assert metrics_path.exists() and history_path.exists()
metrics = json.loads(metrics_path.read_text())
assert int(metrics["best_epoch"]) >= 3, metrics
assert metrics["model_type"] == "rimgraph_v4"
assert (LOCAL_RUN / "folds/G1020/seed_7/global_baseline/best_model.pt").exists()
assert (LOCAL_RUN / "folds/G1020/seed_7/rimgraph_v4/best_model.pt").exists()
assert (DRIVE_RUN / "folds/G1020/seed_7/rimgraph_v4/test_predictions.csv").exists()
assert (LOCAL_RUN / "RUN_COMPLETED.json").exists()
print("RIMGRAPH_V44_FULL_STAGE_END_TO_END_PASSED")
