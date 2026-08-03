import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

ROOT = Path(__file__).resolve().parents[1]
raw_part = (ROOT / "v4_parts" / "part_02.py").read_text(encoding="utf-8")
patch_ns = {}
exec(
    compile((ROOT / "runner_patch_v42.py").read_text(encoding="utf-8"), "runner_patch_v42.py", "exec"),
    patch_ns,
    patch_ns,
)
patched = patch_ns["apply_v42"](raw_part)
compile(patched, "patched_part_02.py", "exec")

# Execute only the sampler class from the patched source.
tree = ast.parse(patched)
class_node = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "BalancedSourceClassBatchSampler"
)
module = ast.Module(body=[class_node], type_ignores=[])
env = {
    "Sampler": Sampler,
    "np": np,
    "pd": pd,
    "math": math,
    "CFG": {"grad_accum": 2},
}
exec(compile(module, "balanced_sampler_only.py", "exec"), env, env)
BalancedSampler = env["BalancedSourceClassBatchSampler"]

rows = []
for source in ["ORIGA", "REFUGE"]:
    for label in [0, 1]:
        for repeat in range(5):
            rows.append({"source": source, "label": label, "repeat": repeat})
df = pd.DataFrame(rows)
all_groups = set(zip(df.source, df.label))

# Exact Colab/T4 profile that previously crashed.
sampler = BalancedSampler(df, batch_size=2, seed=2029)
batches = list(iter(sampler))
assert len(batches) == len(sampler)
assert all(len(batch) == 2 for batch in batches)

# One optimizer update uses two micro-batches (grad_accum=2): all four groups
# must be represented, and each micro-batch should contain both sources/classes.
first_window = batches[:2]
window_groups = {
    (df.iloc[index].source, int(df.iloc[index].label))
    for batch in first_window
    for index in batch
}
assert window_groups == all_groups, (window_groups, all_groups)
for batch in first_window:
    selected = df.iloc[batch]
    assert selected.source.nunique() == 2, selected
    assert selected.label.nunique() == 2, selected

# Smaller micro-batches must rotate safely instead of raising an exception.
sampler_one = BalancedSampler(df, batch_size=1, seed=2029)
one_batches = list(iter(sampler_one))
first_cycle_groups = {
    (df.iloc[batch[0]].source, int(df.iloc[batch[0]].label))
    for batch in one_batches[:4]
}
assert first_cycle_groups == all_groups

# Deterministic for the same epoch and different after an epoch change.
sampler_a = BalancedSampler(df, batch_size=2, seed=7)
sampler_b = BalancedSampler(df, batch_size=2, seed=7)
assert list(iter(sampler_a)) == list(iter(sampler_b))
sampler_b.set_epoch(1)
assert list(iter(sampler_a)) != list(iter(sampler_b))

print("RIMGRAPH_V42_BALANCED_MICROBATCH_SAMPLER_PASSED")
