from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="glaucomma_smoke_"))
DATA = TMP / "data"


def make_source(source: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image_dir = DATA / source / "Images"
    mask_dir = DATA / source / "Masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = []

    yy, xx = np.mgrid[:64, :64]
    disc = (xx - 32) ** 2 + (yy - 32) ** 2 <= 15**2
    cup = (xx - 32) ** 2 + (yy - 32) ** 2 <= 7**2

    for label in (0, 1):
        for index in range(10):
            name = f"{source.lower()}_{label}_{index:02d}"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[..., 1] = np.clip(70 + rng.normal(0, 8, (64, 64)), 0, 255)
            image[..., 0] = np.clip(105 + 45 * disc + 30 * cup * label + rng.normal(0, 8, (64, 64)), 0, 255)
            image[..., 2] = np.clip(35 + 20 * disc + rng.normal(0, 6, (64, 64)), 0, 255)
            Image.fromarray(image).save(image_dir / f"{name}.jpg")

            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[disc] = 128
            mask[cup] = 255
            Image.fromarray(mask).save(mask_dir / f"{name}.png")
            records.append({"image": f"{name}.jpg", "label": label})

    pd.DataFrame(records).to_csv(DATA / source / "labels.csv", index=False)


for i, source in enumerate(("ORIGA", "REFUGE", "G1020"), start=1):
    make_source(source, i)

# Overrides are read by the single-cell runner before execution.
GLAUCOMMA_OVERRIDES = {
    "manual_data_dir": str(DATA),
    "sources": ["ORIGA", "REFUGE", "G1020"],
    "fold_targets": ["ORIGA"],
    "run_name": "ci_runtime_smoke_v3",
    "code_revision": "rimgrah-dg-runtime-v3-20260802",
    "drive_folder": "ci_drive_mirror",
    "image_size": 64,
    "num_workers": 0,
    "epochs": 1,
    "patience": 1,
    "batch_size": 4,
    "grad_accum": 1,
    "mixed_precision": False,
    "resume": False,
    "fast_dev_run": False,
    "backbone": "resnet18",
    "pretrained": False,
    "feature_dim": 32,
    "graph_hidden": 32,
    "graph_heads": 4,
    "dropout": 0.1,
    "run_optuna": False,
    "save_every_epoch": True,
    "n_visual_examples": 1,
}

original = "\n".join((REPO / "runner_parts" / f"part_{i:02d}.py").read_text() for i in range(8))
ns_v2: dict[str, object] = {}
ns_v3: dict[str, object] = {}
exec(compile((REPO / "runner_patch_v2.py").read_text(), "runner_patch_v2.py", "exec"), ns_v2, ns_v2)
exec(compile((REPO / "runner_patch_v3.py").read_text(), "runner_patch_v3.py", "exec"), ns_v3, ns_v3)
code = ns_v2["apply_patches"](original)
code = ns_v3["apply_patches_v3"](code)
compile(code, "rimgraph_dg_runtime_smoke.py", "exec")
exec(code, globals(), globals())

run_root = REPO / "Glaucomma_runs" / GLAUCOMMA_OVERRIDES["run_name"]
required = [
    run_root / "metadata.csv",
    run_root / "folds" / "ORIGA" / "best_model.pt",
    run_root / "folds" / "ORIGA" / "last_checkpoint.pt",
    run_root / "folds" / "ORIGA" / "history.csv",
    run_root / "folds" / "ORIGA" / "test_predictions.csv",
    run_root / "folds" / "ORIGA" / "metrics.json",
    run_root / "folds" / "ORIGA" / "COMPLETED.json",
    run_root / "fold_summary.csv",
    run_root / "aggregate_metrics.csv",
    run_root / "artifact_manifest.csv",
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, f"Missing runtime artifacts: {missing}"

completed = json.loads((run_root / "folds" / "ORIGA" / "COMPLETED.json").read_text())
assert completed["code_revision"] == GLAUCOMMA_OVERRIDES["code_revision"]
predictions = pd.read_csv(run_root / "folds" / "ORIGA" / "test_predictions.csv")
assert len(predictions) == 20
assert {"prob_raw", "prob_calibrated", "vcdr_pred", "sector_01"}.issubset(predictions.columns)
print("END_TO_END_RUNTIME_SMOKE_PASSED")
