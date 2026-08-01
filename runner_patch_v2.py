"""Deterministic source patches for the immutable RimGraph-DG single-cell runner."""


def _replace_once(code: str, old: str, new: str, name: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"Patch {name!r} expected exactly one match, found {count}.")
    return code.replace(old, new, 1)


def apply_patches(code: str) -> str:
    # 1) Record the versions that materially affect reproducibility.
    old = '''ENV = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "device": str(DEVICE),
}'''
    new = '''import importlib.metadata as _ilm
ENV = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "timm": _ilm.version("timm"),
    "albumentations": _ilm.version("albumentations"),
    "kagglehub": _ilm.version("kagglehub"),
    "optuna": _ilm.version("optuna"),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "device": str(DEVICE),
}
if IN_COLAB and DEVICE.type != "cuda":
    raise RuntimeError("GPU runtime is required. In Colab select Runtime > Change runtime type > T4 GPU, then run the single cell again.")'''
    code = _replace_once(code, old, new, "environment-and-gpu-check")

    # 2) Prefer real combined OD/OC masks over separate disc-only/cup-only files.
    old = '''def choose_mask(src, image_path):
    key = norm_key(image_path)
    candidates = MASK_INDEX.get((src, key), [])
    if not candidates:
        for (s, k), vals in MASK_INDEX.items():
            if s == src and (k == key or k in key or key in k):
                candidates.extend(vals)
                if len(candidates) >= 3: break
    if not candidates: return None
    # Prefer combined/cropped mask close to image path characteristics.
    cropped = "cropped" in str(image_path).lower()
    candidates = sorted(set(candidates), key=lambda p: ("cropped" in str(p).lower()) != cropped)
    return str(candidates[0])'''
    new = '''def choose_mask(src, image_path):
    key = norm_key(image_path)
    candidates = list(MASK_INDEX.get((src, key), []))
    if not candidates:
        for (s, k), vals in MASK_INDEX.items():
            if s == src and (k == key or k in key or key in k):
                candidates.extend(vals)
                if len(candidates) >= 8:
                    break
    if not candidates:
        return None

    image_text = str(image_path).lower()
    want_cropped = "cropped" in image_text
    want_square = "square" in image_text

    def mask_rank(p):
        text = str(p).lower().replace("\\\\", "/")
        stem = Path(p).stem.lower()
        separate = bool(re.search(r"(?:^|[_-])(disc|cup)(?:$|[_-])", stem)) or stem.endswith(("disc", "cup"))
        # The Kaggle package supplies combined masks in Masks/Masks_Cropped/Masks_Square.
        combined_dir = any(token in text for token in ["/masks/", "/masks_cropped/", "/masks_square/"])
        representation_penalty = int(("cropped" in text) != want_cropped) + int(("square" in text) != want_square)
        return (separate, not combined_dir, representation_penalty, len(text), text)

    return str(sorted(set(candidates), key=mask_rank)[0])'''
    code = _replace_once(code, old, new, "combined-mask-selection")

    # 3) Missing masks are valid classification samples, not Path(float('nan')) crashes.
    old = '''def decode_combined_mask(path, target_hw):
    h, w = target_hw
    if path is None or not Path(path).exists():
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), 0.0'''
    new = '''def decode_combined_mask(path, target_hw):
    h, w = target_hw
    if path is None or (isinstance(path, float) and np.isnan(path)) or pd.isna(path) or not Path(str(path)).exists():
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), 0.0'''
    code = _replace_once(code, old, new, "missing-mask-guard")

    # 4) Modern PyTorch defaults to weights_only loading; our own trusted checkpoints contain metadata.
    code = _replace_once(
        code,
        'ck=torch.load(last,map_location=DEVICE);',
        'ck=torch.load(last,map_location=DEVICE,weights_only=False);',
        "resume-checkpoint-load",
    )
    code = _replace_once(
        code,
        'best=torch.load(find_existing(f"{fold_rel}/best_model.pt"),map_location=DEVICE);model.load_state_dict(best["model"])',
        'best=torch.load(find_existing(f"{fold_rel}/best_model.pt"),map_location=DEVICE,weights_only=False);model.load_state_dict(best["model"])',
        "best-checkpoint-load",
    )

    # 5) Save the updated best score/epoch in last_checkpoint so resume cannot replace a better model.
    old = '''        ck={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"history":history,"best_score":best_score,"best_epoch":best_epoch,"cfg":CFG}
        if CFG["save_every_epoch"]:save_checkpoint(ck,f"{fold_rel}/last_checkpoint.pt")
        if score>best_score:
            best_score=score;best_epoch=epoch;save_checkpoint({**ck,"best_score":best_score,"best_epoch":best_epoch},f"{fold_rel}/best_model.pt")'''
    new = '''        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
        ck={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"history":history,"best_score":best_score,"best_epoch":best_epoch,"cfg":CFG}
        if CFG["save_every_epoch"]:
            save_checkpoint(ck,f"{fold_rel}/last_checkpoint.pt")
        if improved:
            save_checkpoint(ck,f"{fold_rel}/best_model.pt")'''
    code = _replace_once(code, old, new, "resume-best-state")

    # 6) A completed smoke-test folder must never be mistaken for a paper run.
    old = '''    if CFG["resume"] and completed:
        saved=json.loads(Path(completed).read_text());pred=pd.read_csv(find_existing(f"{fold_rel}/test_predictions.csv"))
        return saved["metrics"],pred,None'''
    new = '''    if CFG["resume"] and completed:
        saved=json.loads(Path(completed).read_text())
        completed_mode = bool(saved.get("fast_dev_run", False))
        if completed_mode == bool(CFG["fast_dev_run"]):
            pred_path = find_existing(f"{fold_rel}/test_predictions.csv")
            if pred_path is not None:
                pred=pd.read_csv(pred_path)
                return saved["metrics"],pred,None'''
    code = _replace_once(code, old, new, "completed-mode-check")

    old = '''    STORE.save_json({"metrics":metrics,"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")},f"{fold_rel}/COMPLETED.json")'''
    new = '''    STORE.save_json({"metrics":metrics,"fast_dev_run":bool(CFG["fast_dev_run"]),"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")},f"{fold_rel}/COMPLETED.json")'''
    code = _replace_once(code, old, new, "completed-mode-write")

    # 7) Avoid all-NaN segmentation summaries when a fold contains no verified masks.
    old = '''    m=classification_metrics(np.array(all_y),np.array(all_p)); dd=np.nanmean(np.stack(dices),axis=0)
    m.update({"loss":np.mean(losses),"dice_disc":float(dd[0]),"dice_cup":float(dd[1])})'''
    new = '''    m=classification_metrics(np.array(all_y),np.array(all_p))
    dd=np.nanmean(np.stack(dices),axis=0) if dices else np.array([np.nan,np.nan])
    m.update({"loss":np.mean(losses),"dice_disc":float(dd[0]),"dice_cup":float(dd[1])})'''
    code = _replace_once(code, old, new, "epoch-dice-guard")

    old = '''    dd=np.nanmean(np.stack(dices),axis=0)
    return pd.DataFrame(rows),{"dice_disc":float(dd[0]),"dice_cup":float(dd[1])}'''
    new = '''    dd=np.nanmean(np.stack(dices),axis=0) if dices else np.array([np.nan,np.nan])
    return pd.DataFrame(rows),{"dice_disc":float(dd[0]),"dice_cup":float(dd[1])}'''
    code = _replace_once(code, old, new, "predict-dice-guard")

    return code
