from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

ROOT = Path(__file__).resolve().parents[1]
raw = "\n".join((ROOT / f"v4_parts/part_{i:02d}.py").read_text() for i in range(7))
code = raw
for patch_name, fn_name in [
    ("runner_patch_v41.py", "apply_v41"),
    ("runner_patch_v42.py", "apply_v42"),
    ("runner_patch_v43.py", "apply_v43"),
]:
    ns = {}
    src = (ROOT / patch_name).read_text()
    exec(compile(src, patch_name, "exec"), ns, ns)
    code = ns[fn_name](code)

compile(code, "rimgraph_dg_v43.py", "exec")
required = [
    "Prototype-aware source/class balancing",
    "prototype_pair_count",
    "eligible_for_selection = va[\"stage\"] == \"full\"",
    "vertical_cup_extent / vertical_disc_extent",
    "torch.fft.rfft(rim_vector",
    "cup_prob = raw[:, 1:2] * disc_prob",
    "gate_context",
]
for marker in required:
    assert marker in code, marker
assert "improved = score > best_score" not in code[code.index("def train_full("):]

sampler_start = code.index("class BalancedSourceClassBatchSampler")
sampler_end = code.index("\n\ndef make_loader", sampler_start)
sampler_ns = {"Sampler": Sampler, "np": np, "pd": pd, "math": math, "CFG": {"grad_accum": 2}}
exec(code[sampler_start:sampler_end], sampler_ns, sampler_ns)
SamplerClass = sampler_ns["BalancedSourceClassBatchSampler"]

df = pd.DataFrame([
    {"source": source, "label": label}
    for source in ["ORIGA", "REFUGE"]
    for label in [0, 1]
    for _ in range(4)
])
sampler = SamplerClass(df, batch_size=2, seed=7)
batches = list(iter(sampler))[:2]
seen = set()
for batch in batches:
    rows = df.iloc[batch]
    assert rows.source.nunique() == 2
    assert rows.label.nunique() == 1
    seen.update(zip(rows.source, rows.label))
assert seen == {("ORIGA", 0), ("ORIGA", 1), ("REFUGE", 0), ("REFUGE", 1)}

helper_start = code.index("def prototype_pair_count")
helper_end = code.index("\n\ndef expected_calibration_error", helper_start)
helper_ns = {"torch": torch}
exec(code[helper_start:helper_end], helper_ns, helper_ns)
for batch in batches:
    rows = df.iloc[batch]
    labels = torch.tensor(rows.label.to_numpy())
    domains = torch.tensor(pd.Categorical(rows.source).codes)
    assert helper_ns["prototype_pair_count"](labels, domains) == 1

print("RIMGRAPH_V43_SCIENTIFIC_GUARDS_PASSED")
