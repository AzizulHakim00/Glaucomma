"""V4.1 source patch for Colab Drive persistence and T4 memory safety."""


def _replace_once(code: str, old: str, new: str, label: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"V4.1 patch expected one {label} block, found {count}")
    return code.replace(old, new, 1)


def apply_v41(code: str) -> str:
    code = _replace_once(
        code,
        "import os, sys, subprocess, warnings, json, math, random, time, shutil, hashlib, platform, re, gc\n",
        "import os, sys, subprocess, warnings, json, math, random, time, shutil, hashlib, platform, re, gc, traceback\n",
        "import",
    )

    old_paths = '''COLAB_ROOT = Path("/content/Glaucomma_runs") if IN_COLAB else Path.cwd() / "Glaucomma_runs"
DRIVE_ROOT = (Path("/content/drive/MyDrive") / CFG["drive_folder"]) if IN_COLAB else (Path.cwd() / "DriveMirror" / CFG["drive_folder"])
LOCAL_RUN = COLAB_ROOT / CFG["run_name"]
DRIVE_RUN = DRIVE_ROOT / CFG["run_name"]
for _p in [LOCAL_RUN, DRIVE_RUN]:
    _p.mkdir(parents=True, exist_ok=True)
'''
    new_paths = '''def _resolve_my_drive_root():
    if not IN_COLAB:
        return Path.cwd() / "DriveMirror"
    candidates = [Path("/content/drive/MyDrive"), Path("/content/drive/My Drive")]
    root = next((p for p in candidates if p.exists()), None)
    if root is None:
        listing = []
        drive_mount = Path("/content/drive")
        if drive_mount.exists():
            listing = [str(p) for p in drive_mount.iterdir()]
        raise RuntimeError(
            "Google Drive is not accessible after mounting. "
            f"Observed /content/drive entries: {listing}. Remount Drive and rerun."
        )
    probe_dir = root / CFG["drive_folder"]
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / ".rimgraph_write_probe"
    probe.write_text("ok", encoding="utf-8")
    if probe.read_text(encoding="utf-8") != "ok":
        raise RuntimeError(f"Google Drive write verification failed at {probe_dir}")
    probe.unlink(missing_ok=True)
    return root


COLAB_ROOT = Path("/content/Glaucomma_runs") if IN_COLAB else Path.cwd() / "Glaucomma_runs"
MYDRIVE_ROOT = _resolve_my_drive_root()
DRIVE_ROOT = MYDRIVE_ROOT / CFG["drive_folder"]
LOCAL_RUN = COLAB_ROOT / CFG["run_name"]
DRIVE_RUN = DRIVE_ROOT / CFG["run_name"]
for _p in [LOCAL_RUN, DRIVE_RUN]:
    _p.mkdir(parents=True, exist_ok=True)

_startup = {
    "status": "running",
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_name": CFG["run_name"],
    "code_revision": CFG["code_revision"],
    "local_run": str(LOCAL_RUN),
    "drive_run": str(DRIVE_RUN),
}
for _marker in [LOCAL_RUN / "RUNNING.json", DRIVE_RUN / "RUNNING.json"]:
    _marker.write_text(json.dumps(_startup, indent=2), encoding="utf-8")
print(f"Resolved Colab output: {LOCAL_RUN}")
print(f"Resolved Drive output: {DRIVE_RUN}")
print("Drive write verification: PASSED")
'''
    code = _replace_once(code, old_paths, new_paths, "output path")

    old_json = '''def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    if torch.is_tensor(x): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)
'''
    new_json = old_json + '''\n\ndef write_heartbeat(stage, **extra):
    payload = {
        "status": "running",
        "stage": str(stage),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": CFG["run_name"],
        "code_revision": CFG["code_revision"],
        **extra,
    }
    text = json.dumps(payload, indent=2, default=_json_default)
    for marker in [LOCAL_RUN / "RUNNING.json", DRIVE_RUN / "RUNNING.json"]:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(text, encoding="utf-8")
\n\nwrite_heartbeat("startup_complete", device=str(DEVICE))
'''
    code = _replace_once(code, old_json, new_json, "JSON helper")

    old_backbone = '''            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
            return model, name
'''
    new_backbone = '''            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
            if CFG.get("gradient_checkpointing", True) and hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(True)
            return model, name
'''
    code = _replace_once(code, old_backbone, new_backbone, "backbone creation")

    old_fold = '''        section(f"Seed {seed} — held-out {target}")
        root_rel = f"folds/{target}/seed_{seed}"
'''
    new_fold = '''        write_heartbeat("fold_start", seed=int(seed), held_out_source=str(target))
        section(f"Seed {seed} — held-out {target}")
        root_rel = f"folds/{target}/seed_{seed}"
'''
    code = _replace_once(code, old_fold, new_fold, "fold start")

    old_cleanup = '''        fold_rows = [m for m in [baseline_metrics, full_metrics] if m is not None]
        display(pd.DataFrame(fold_rows)[["model_type", "held_out_source", "auroc", "auprc", "sensitivity", "specificity", "f1", "mcc", "ece"] + (["dice_disc", "dice_cup"] if full_metrics is not None else [])].fillna(""))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
'''
    new_cleanup = '''        if "model" in locals():
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        fold_rows = [m for m in [baseline_metrics, full_metrics] if m is not None]
        display(pd.DataFrame(fold_rows)[["model_type", "held_out_source", "auroc", "auprc", "sensitivity", "specificity", "f1", "mcc", "ece"] + (["dice_disc", "dice_cup"] if full_metrics is not None else [])].fillna(""))
        write_heartbeat("fold_complete", seed=int(seed), held_out_source=str(target), completed_models=len(fold_rows))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
'''
    code = _replace_once(code, old_cleanup, new_cleanup, "fold cleanup")

    old_finish = '''clear_output(wait=True)
display(Markdown("# ✅ RimGraph-DG V4 run completed"))
'''
    new_finish = '''for marker in [LOCAL_RUN / "RUNNING.json", DRIVE_RUN / "RUNNING.json"]:
    marker.unlink(missing_ok=True)
STORE.save_json({
    "status": "completed",
    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_name": CFG["run_name"],
    "code_revision": CFG["code_revision"],
}, "RUN_COMPLETED.json")
clear_output(wait=True)
display(Markdown("# ✅ RimGraph-DG V4.1 run completed"))
'''
    code = _replace_once(code, old_finish, new_finish, "completion marker")

    return code
