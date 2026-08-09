import ast
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

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
]:
    ns = {}
    source = (ROOT / patch_name).read_text()
    exec(compile(source, patch_name, "exec"), ns, ns)
    code = ns[fn_name](code)

module = ast.parse(code)
wanted = {
    "_read_mask_gray",
    "_border_mode",
    "_mask_levels",
    "read_binary_mask",
    "decode_combined_mask",
}
selected = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
mini = ast.Module(body=selected, type_ignores=[])
ast.fix_missing_locations(mini)
ns = {"cv2": cv2, "np": np, "pd": pd, "Path": Path}
exec(compile(mini, "v45_mask_decoder_only.py", "exec"), ns, ns)

with tempfile.TemporaryDirectory(prefix="rimgraph_v45_masks_") as td:
    td = Path(td)

    compact = np.zeros((128, 128), np.uint8)
    cv2.circle(compact, (64, 64), 30, 1, -1)
    cv2.circle(compact, (64, 64), 12, 2, -1)
    p = td / "compact_012.png"
    cv2.imwrite(str(p), compact)
    disc, cup, dv, cv = ns["decode_combined_mask"](str(p), compact.shape)
    assert dv == 1.0 and cv == 1.0
    assert disc.sum() > cup.sum() > 20
    assert np.all(cup <= disc)

    binary = np.zeros((128, 128), np.uint8)
    cv2.circle(binary, (64, 64), 25, 1, -1)
    p2 = td / "binary_01.png"
    cv2.imwrite(str(p2), binary)
    obj, valid = ns["read_binary_mask"](str(p2), binary.shape)
    assert valid == 1.0 and obj.sum() > 20

print("RIMGRAPH_V45_LOW_LABEL_MASKS_PASSED")
