import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="rimgraph_v45_audit_ci_"))
os.chdir(TMP)

# Tiny three-source dataset with canonical/derived mask encodings.
for source in ["ORIGA", "REFUGE", "G1020"]:
    mask_dir = TMP / source / "Masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for label, cls in [(0, "Normal"), (1, "Glaucoma")]:
        image_dir = TMP / source / cls / "Images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            stem = f"{source}_{cls}_{i:03d}"
            image = np.full((64, 64, 3), 40 + label * 30, np.uint8)
            cv2.circle(image, (32, 32), 20, (130, 100, 80), -1)
            cv2.imwrite(str(image_dir / f"{stem}.png"), image)

            if source == "REFUGE":
                mask = np.full((64, 64), 255, np.uint8)
                cv2.circle(mask, (32, 32), 15, 128, -1)
                cv2.circle(mask, (32, 32), 6 + label, 0, -1)
            elif source == "G1020":
                mask = np.zeros((64, 64), np.uint8)
                cv2.circle(mask, (32, 32), 15, 1, -1)
                cv2.circle(mask, (32, 32), 6 + label, 2, -1)
            else:
                mask = np.zeros((64, 64), np.uint8)
                cv2.circle(mask, (32, 32), 15, 128, -1)
                cv2.circle(mask, (32, 32), 6 + label, 255, -1)
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)

GLAUCOMMA_OVERRIDES = {
    "manual_data_dir": str(TMP),
    "sources": ["ORIGA", "REFUGE", "G1020"],
    "fold_targets": ["ORIGA"],
    "run_name": "ci_v45_cpu_audit",
    "code_revision": "rimgraph-dg-v4.5-cpu-audit-ci",
    "run_global_baseline": False,
    "run_full_model": False,
    "run_optuna": False,
    "resume": False,
    "pretrained": False,
    "num_workers": 0,
}

raw = "\n".join((ROOT / f"v4_parts/part_{i:02d}.py").read_text() for i in range(7))
code = raw
for patch_name, fn_name in [
    ("runner_patch_v41.py", "apply_v41"),
    ("runner_patch_v42.py", "apply_v42"),
    ("runner_patch_v43.py", "apply_v43"),
    ("runner_patch_v43_autograd.py", "apply_v43_autograd"),
    ("runner_patch_v44_runtime.py", "apply_v44_runtime"),
    ("runner_patch_v45_masks.py", "apply_v45_masks"),
    ("runner_patch_v45_lowlabels.py", "apply_v45_lowlabels"),
    ("runner_patch_v45_audit_only.py", "apply_v45_audit_only"),
]:
    ns = {}
    source = (ROOT / patch_name).read_text()
    exec(compile(source, patch_name, "exec"), ns, ns)
    code = ns[fn_name](code)

assert "_v44_gpu_model_preflight()" not in code
assert "for seed in CFG[\"seeds\"]" not in code
compile(code, "rimgraph_v45_cpu_audit_ci.py", "exec")
exec(code, globals(), globals())

run = Path.cwd() / "Glaucomma_runs" / "ci_v45_cpu_audit"
summary_path = run / "mask_validity_by_source.csv"
marker_path = run / "MASK_AUDIT_COMPLETED.json"
assert summary_path.exists(), summary_path
assert marker_path.exists(), marker_path
summary = pd.read_csv(summary_path).set_index("source")
for source in ["ORIGA", "REFUGE", "G1020"]:
    assert summary.loc[source, "disc_valid_rate"] == 1.0
    assert summary.loc[source, "cup_valid_rate"] == 1.0
marker = json.loads(marker_path.read_text())
assert marker["mode"] == "cpu_mask_audit_only"
assert not list(run.rglob("best_model.pt")), "Audit-only mode unexpectedly trained a model"
print("RIMGRAPH_V45_CPU_MASK_AUDIT_ONLY_PASSED")
