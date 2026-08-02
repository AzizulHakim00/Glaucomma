import os, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="rimgraph_v4_ci_"))
os.chdir(TMP)

GLAUCOMMA_OVERRIDES = {
    "manual_data_dir": str(TMP),
    "fast_dev_run": True,
    "run_name": "ci_v4",
    "code_revision": "rimgraph-dg-v4-ci",
    "pretrained": False,
    "backbone": "resnet18",
    "backbone_fallback": "resnet18",
    "image_size": 64,
    "fpn_dim": 32,
    "feature_dim": 32,
    "graph_hidden": 32,
    "graph_heads": 4,
    "roi_output_size": 8,
    "batch_size": 4,
    "grad_accum": 1,
    "mixed_precision": False,
    "bootstrap_samples": 10,
    "num_workers": 0,
    "resume": False,
}

ns = globals()
for part in ["part_00.py", "part_02.py", "part_03.py", "part_04.py", "part_05.py"]:
    code = (ROOT / "v4_parts" / part).read_text()
    exec(compile(code, part, "exec"), ns, ns)

rows = []
for source_i, source in enumerate(["ORIGA", "REFUGE", "G1020"]):
    source_dir = TMP / source / "Images"
    mask_dir = TMP / source / "Masks"
    source_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i in range(12):
        label = i % 2
        image = np.zeros((96, 96, 3), np.uint8)
        image[:] = (18 + source_i * 8, 20 + source_i * 6, 22 + source_i * 5)
        cv2.circle(image, (48, 48), 39, (70 + label * 20, 80 + label * 20, 90 + label * 20), -1)
        cv2.circle(image, (60, 48), 12 + 3 * label, (180, 160, 130), -1)
        image_path = source_dir / f"{source}_{i:03d}.png"
        cv2.imwrite(str(image_path), image)

        mask = np.full((96, 96), 255, np.uint8)
        cv2.circle(mask, (60, 48), 18, 128, -1)
        cv2.circle(mask, (60, 48), 7 + 2 * label, 0, -1)
        mask_path = mask_dir / f"{source}_{i:03d}.png"
        cv2.imwrite(str(mask_path), mask)
        rows.append({
            "image_path": str(image_path), "source": source, "dataset_split": "unspecified",
            "label": label, "label_origin": "synthetic", "laterality": "L" if i % 3 else "R",
            "combined_mask_path": str(mask_path), "disc_mask_path": None, "cup_mask_path": None,
            "representation_rank": 0, "image_key": f"{source}_{i:03d}",
            "fingerprint": f"{source}_{i:03d}",
        })

META = pd.DataFrame(rows)
EXCLUDED_UNLABELLED = pd.DataFrame()

train_df, val_df, test_df = prepare_fold_data("G1020", 7, "folds/G1020/seed_7")
base_metrics, base_pred = train_baseline("G1020", 7, train_df, val_df, test_df, "folds/G1020/seed_7")
full_metrics, full_pred, model = train_full("G1020", 7, train_df, val_df, test_df, "folds/G1020/seed_7")

assert len(base_pred) > 0 and len(full_pred) > 0
assert {"auroc", "auprc", "ece"}.issubset(base_metrics)
assert {"auroc", "auprc", "dice_disc", "dice_cup", "ece"}.issubset(full_metrics)
assert np.isfinite(full_metrics["auroc"])
assert (LOCAL_RUN / "folds/G1020/seed_7/global_baseline/best_model.pt").exists()
assert (LOCAL_RUN / "folds/G1020/seed_7/rimgraph_v4/best_model.pt").exists()
assert (DRIVE_RUN / "folds/G1020/seed_7/rimgraph_v4/test_predictions.csv").exists()
print("RIMGRAPH_V4_END_TO_END_SMOKE_PASSED")
