"""V4.5 mask-validity patch.

Goals:
- decode the common OD/OC label conventions used by ORIGA/REFUGE/G1020;
- prefer valid separate masks when a combined mask is unusable;
- audit *decoded* supervision, not only mask-path presence;
- fail before any long GPU work when disc/cup supervision is implausible;
- allow exact reuse of a completed V4.4 baseline when the split and all
  baseline-sensitive settings are identical;
- force RimGraph itself to start fresh under the new V4.5 run/revision.
"""


def _once(code: str, old: str, new: str, label: str) -> str:
    n = code.count(old)
    if n != 1:
        raise RuntimeError(f"V4.5 expected one {label}, found {n}")
    return code.replace(old, new, 1)


def _block(code: str, start: str, end: str, replacement: str, label: str) -> str:
    a = code.find(start)
    b = code.find(end, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        raise RuntimeError(f"V4.5 could not locate {label}")
    return code[:a] + replacement.rstrip() + code[b:]


def apply_v45_masks(code: str) -> str:
    # ------------------------------------------------------------------
    # 1) Robust mask decoding.
    # ------------------------------------------------------------------
    decoder = r'''def _read_mask_gray(path, shape=None):
    if path is None or pd.isna(path) or not Path(str(path)).exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if shape is not None and m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def _border_mode(mask):
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_levels(mask):
    values, counts = np.unique(mask, return_counts=True)
    return [(int(v), int(c)) for v, c in zip(values, counts)]


def read_binary_mask(path, shape):
    m = _read_mask_gray(path, shape)
    if m is None:
        return np.zeros(shape, np.float32), 0.0
    bg = _border_mode(m)
    out = (np.abs(m.astype(np.int16) - bg) > 3).astype(np.float32)
    return out, float(out.sum() > 20)


def decode_combined_mask(path, shape):
    """Decode a nested optic-disc/optic-cup label mask.

    Supported common conventions include REFUGE (255 background, 128 disc,
    0 cup) and the black-background grey/white masks commonly distributed in
    derived ORIGA/REFUGE/G1020 bundles.  The disc target includes the cup.
    """
    m = _read_mask_gray(path, shape)
    if m is None:
        z = np.zeros(shape, np.float32)
        return z, z.copy(), 0.0, 0.0

    bg = _border_mode(m)
    levels = _mask_levels(m)
    # Ignore tiny compression/edge speckles; semantic labels must occupy a
    # meaningful region. Keep at least 10 pixels or 0.002% of the mask.
    min_region = max(10, int(round(m.size * 0.00002)))
    candidates = [(v, c) for v, c in levels if abs(v - bg) > 3 and c >= min_region]

    if not candidates:
        z = np.zeros(shape, np.float32)
        return z, z.copy(), 0.0, 0.0

    # REFUGE canonical labels: 255 background, 128 disc, 0 cup.
    values = {v for v, _ in levels}
    if bg >= 250 and 128 in values and 0 in values:
        disc = (m < 250).astype(np.float32)
        cup = (m < 64).astype(np.float32)
        return disc, cup * disc, float(disc.sum() > 20), float(cup.sum() > 10)

    # Common converted convention: black background, grey disc, white cup.
    if bg <= 5 and 128 in values and max(values) >= 250:
        disc = (m > 5).astype(np.float32)
        cup = (m >= 250).astype(np.float32)
        return disc, cup * disc, float(disc.sum() > 20), float(cup.sum() > 10)

    # Generic labelled mask: all non-background semantic regions form the
    # disc; the smallest meaningful nested class is the cup.
    semantic_values = [v for v, _ in candidates]
    disc = np.isin(m, semantic_values).astype(np.float32)
    if len(candidates) >= 2:
        cup_value = min(candidates, key=lambda x: x[1])[0]
        cup = (m == cup_value).astype(np.float32)
    else:
        cup = np.zeros_like(disc)
    return disc, cup * disc, float(disc.sum() > 20), float(cup.sum() > 10)


def load_masks(row, shape):
    """Load the best available supervision without trusting one path blindly."""
    z = np.zeros(shape, np.float32)
    disc, cup, dv, cv = z, z.copy(), 0.0, 0.0

    if row.combined_mask_path is not None and not pd.isna(row.combined_mask_path):
        disc, cup, dv, cv = decode_combined_mask(row.combined_mask_path, shape)

    # A dataset bundle can expose both a combined mask and explicit OD/OC
    # masks. Prefer explicit masks when they recover supervision that the
    # combined decoder could not validate.
    if row.disc_mask_path is not None and not pd.isna(row.disc_mask_path):
        d2, dv2 = read_binary_mask(row.disc_mask_path, shape)
        if dv2 > dv:
            disc, dv = d2, dv2
    if row.cup_mask_path is not None and not pd.isna(row.cup_mask_path):
        c2, cv2 = read_binary_mask(row.cup_mask_path, shape)
        if cv2 > cv:
            cup, cv = c2, cv2

    if dv:
        cup = cup * disc
    if cv and float(cup.sum()) <= 10:
        cv = 0.0
    return disc, cup, float(dv), float(cv)
'''
    code = _block(code, 'def read_binary_mask(path, shape):', '\n\ndef fov_bbox', decoder, 'mask decoder')

    # ------------------------------------------------------------------
    # 2) Decode-validity audit before the V4.4 model/GPU preflight.
    # ------------------------------------------------------------------
    audit_code = r'''def _decode_mask_for_audit(row):
    # Read only annotation files here; do not decode the large RGB fundus image.
    paths = [row.combined_mask_path, row.disc_mask_path, row.cup_mask_path]
    existing = [Path(str(p)) for p in paths if p is not None and not pd.isna(p) and Path(str(p)).exists()]
    if not existing:
        return {"disc_valid": 0, "cup_valid": 0, "vcdr_valid": 0, "disc_pixels": 0, "cup_pixels": 0, "mask_shape": "", "mask_levels": ""}

    probe = _read_mask_gray(existing[0], None)
    if probe is None:
        return {"disc_valid": 0, "cup_valid": 0, "vcdr_valid": 0, "disc_pixels": 0, "cup_pixels": 0, "mask_shape": "", "mask_levels": "unreadable"}
    shape = probe.shape
    disc, cup, dv, cv = load_masks(row, shape)
    levels = _mask_levels(probe)
    compact_levels = ";".join(f"{v}:{c}" for v, c in levels[:12])
    return {
        "disc_valid": int(dv > 0),
        "cup_valid": int(cv > 0),
        "vcdr_valid": int(dv > 0 and cv > 0),
        "disc_pixels": int(disc.sum()),
        "cup_pixels": int(cup.sum()),
        "mask_shape": f"{shape[0]}x{shape[1]}",
        "mask_levels": compact_levels,
    }


def validate_decoded_masks():
    print("\\n=== V4.5 DECODED MASK AUDIT ===", flush=True)
    write_heartbeat("decoded_mask_audit_start")
    rows = []
    for i, row in META.iterrows():
        info = _decode_mask_for_audit(row)
        rows.append({
            "source": row.source,
            "image_key": row.image_key,
            "label": int(row.label),
            "image_path": row.image_path,
            "combined_mask_path": row.combined_mask_path,
            "disc_mask_path": row.disc_mask_path,
            "cup_mask_path": row.cup_mask_path,
            **info,
        })
        if (i + 1) % 500 == 0:
            print(f"[MASK AUDIT] decoded {i + 1}/{len(META)} annotations", flush=True)

    detail = pd.DataFrame(rows)
    STORE.save_df(detail, "mask_validity_audit.csv")
    summary = detail.groupby("source").agg(
        total=("image_key", "size"),
        valid_disc=("disc_valid", "sum"),
        valid_cup=("cup_valid", "sum"),
        valid_vcdr=("vcdr_valid", "sum"),
    ).reset_index()
    summary["disc_valid_rate"] = summary.valid_disc / summary.total
    summary["cup_valid_rate"] = summary.valid_cup / summary.total
    summary["vcdr_valid_rate"] = summary.valid_vcdr / summary.total
    STORE.save_df(summary, "mask_validity_by_source.csv")
    display(summary.round(4))

    failed = detail[(detail.disc_valid == 0) | (detail.cup_valid == 0)].copy()
    STORE.save_df(failed, "mask_decode_failures_or_missing_cup.csv")

    # Disc annotations should be essentially complete in all three sources.
    # G1020 legitimately contains eyes without a visible optic cup; its source
    # paper reports 60 glaucoma + 170 healthy images without visible OC.
    min_disc = {"ORIGA": 0.95, "REFUGE": 0.95, "G1020": 0.95}
    min_cup = {"ORIGA": 0.90, "REFUGE": 0.90, "G1020": 0.70}
    problems = []
    for rec in summary.to_dict("records"):
        src = rec["source"]
        if rec["disc_valid_rate"] < min_disc.get(src, 0.90):
            problems.append(f"{src}: valid disc rate {rec['disc_valid_rate']:.3f} < {min_disc.get(src, 0.90):.2f}")
        if rec["cup_valid_rate"] < min_cup.get(src, 0.70):
            problems.append(f"{src}: valid cup rate {rec['cup_valid_rate']:.3f} < {min_cup.get(src, 0.70):.2f}")
    if problems:
        raise RuntimeError(
            "V4.5 decoded-mask audit FAILED before GPU training. "
            + " | ".join(problems)
            + f". Inspect {DRIVE_RUN / 'mask_validity_audit.csv'}"
        )
    print("V4.5 DECODED MASK AUDIT: PASSED", flush=True)
    write_heartbeat("decoded_mask_audit_passed")
    return summary


MASK_VALIDITY = validate_decoded_masks()
'''
    preflight_call = '_v44_gpu_model_preflight()\n\nALL_METRICS, ALL_PREDS = [], []\n'
    code = _once(code, preflight_call, audit_code + '\n\n_v44_gpu_model_preflight()\n\nALL_METRICS, ALL_PREDS = [], []\n', 'preflight call')

    # ------------------------------------------------------------------
    # 3) Never allow silent zero-mask training metrics again.
    # ------------------------------------------------------------------
    marker = '    metrics["prototype_active_batch_ratio"] = float(prototype_active_batches / full_stage_batches) if full_stage_batches else np.nan\n'
    guard = marker + '''    if train and int(metrics.get("n_mask_disc", 0)) == 0:\n        raise RuntimeError("V4.5 guard: training epoch contained zero valid optic-disc masks")\n    if train and stage in {"anatomy", "full"} and int(metrics.get("n_mask_cup", 0)) == 0:\n        raise RuntimeError("V4.5 guard: anatomy/full epoch contained zero valid optic-cup masks")\n'''
    code = _once(code, marker, guard, 'segmentation count guard')

    # Make progress logs include the actual supervision counts.
    old_log = 'print(f"[RIMGRAPH] epoch {epoch} done | stage={stage} loss={metrics[\'loss\']:.4f} auroc={metrics[\'auroc\']:.4f} auprc={metrics[\'auprc\']:.4f} disc_dice={metrics[\'dice_disc\']:.4f} cup_dice={metrics[\'dice_cup\']:.4f} proto_active={proto_text}", flush=True)'
    new_log = 'print(f"[RIMGRAPH] epoch {epoch} done | stage={stage} loss={metrics[\'loss\']:.4f} auroc={metrics[\'auroc\']:.4f} auprc={metrics[\'auprc\']:.4f} disc_dice={metrics[\'dice_disc\']:.4f} cup_dice={metrics[\'dice_cup\']:.4f} n_disc={metrics[\'n_mask_disc\']} n_cup={metrics[\'n_mask_cup\']} proto_active={proto_text}", flush=True)'
    code = _once(code, old_log, new_log, 'RimGraph progress log')

    # ------------------------------------------------------------------
    # 4) Reuse only a scientifically identical completed V4.4 baseline.
    # ------------------------------------------------------------------
    baseline_helper = r'''def _split_signature(df):
    cols = [c for c in ["source", "image_key", "label"] if c in df.columns]
    if len(cols) != 3:
        return None
    stable = df[cols].copy()
    stable["label"] = stable["label"].astype(int)
    stable = stable.sort_values(cols).reset_index(drop=True)
    return hashlib.sha256(stable.to_csv(index=False).encode("utf-8")).hexdigest()


def _try_reuse_prior_baseline(target_source, seed, train_df, val_df, test_df, root_rel, rel):
    prior_run = str(CFG.get("baseline_reuse_run", "") or "").strip()
    if not prior_run or prior_run == CFG["run_name"]:
        return None
    prior_root = DRIVE_ROOT / prior_run
    prior_fold = prior_root / root_rel
    prior_rel = prior_fold / "global_baseline"
    needed = [
        prior_rel / "COMPLETED.json", prior_rel / "test_predictions.csv", prior_rel / "best_model.pt",
        prior_fold / "train_split.csv", prior_fold / "val_split.csv", prior_fold / "test_split.csv",
    ]
    if not all(p.exists() for p in needed):
        print(f"[BASELINE REUSE] prior artifacts incomplete in {prior_rel}; training baseline normally", flush=True)
        return None

    prior_train = pd.read_csv(prior_fold / "train_split.csv")
    prior_val = pd.read_csv(prior_fold / "val_split.csv")
    prior_test = pd.read_csv(prior_fold / "test_split.csv")
    current_sigs = [_split_signature(x) for x in [train_df, val_df, test_df]]
    prior_sigs = [_split_signature(x) for x in [prior_train, prior_val, prior_test]]
    if None in current_sigs or current_sigs != prior_sigs:
        print("[BASELINE REUSE] split signature mismatch; training baseline normally", flush=True)
        return None

    ck = torch.load(prior_rel / "best_model.pt", map_location="cpu", weights_only=False)
    old_cfg = ck.get("cfg", {})
    sensitive = [
        "image_size", "backbone", "backbone_fallback", "pretrained", "baseline_epochs",
        "val_fraction", "lr", "backbone_lr_scale", "weight_decay", "focal_gamma",
        "batch_size", "grad_accum",
    ]
    mismatch = [k for k in sensitive if old_cfg.get(k) != CFG.get(k)]
    if mismatch or ck.get("model_type") != "global_baseline" or int(ck.get("seed", -1)) != int(seed):
        print(f"[BASELINE REUSE] config mismatch {mismatch}; training baseline normally", flush=True)
        return None

    current_local = LOCAL_RUN / rel
    current_drive = DRIVE_RUN / rel
    if current_local.exists(): shutil.rmtree(current_local)
    if current_drive.exists(): shutil.rmtree(current_drive)
    shutil.copytree(prior_rel, current_local)
    shutil.copytree(prior_rel, current_drive)

    saved = json.loads((prior_rel / "COMPLETED.json").read_text(encoding="utf-8"))
    metrics = saved["metrics"]
    pred = pd.read_csv(prior_rel / "test_predictions.csv")
    provenance = {
        "status": "reused",
        "source_run": prior_run,
        "source_code_revision": saved.get("code_revision"),
        "current_code_revision": CFG["code_revision"],
        "held_out_source": target_source,
        "seed": int(seed),
        "split_signatures": current_sigs,
        "validated_config_fields": sensitive,
    }
    STORE.save_json(provenance, f"{rel}/BASELINE_REUSE.json")
    STORE.save_json({"metrics": metrics, "code_revision": CFG["code_revision"], "reused_from": provenance}, f"{rel}/COMPLETED.json")
    print(f"[BASELINE REUSE] VERIFIED reuse from {prior_run} for held-out {target_source}", flush=True)
    return metrics, pred


'''
    train_baseline_marker = 'def train_baseline(target_source, seed, train_df, val_df, test_df, root_rel):\n'
    code = _once(code, train_baseline_marker, baseline_helper + train_baseline_marker, 'baseline helper insertion')
    baseline_start = '''def train_baseline(target_source, seed, train_df, val_df, test_df, root_rel):
    rel = f"{root_rel}/global_baseline"; STORE.dirs(rel)
    completed = find_existing(f"{rel}/COMPLETED.json")
'''
    baseline_new = '''def train_baseline(target_source, seed, train_df, val_df, test_df, root_rel):
    rel = f"{root_rel}/global_baseline"; STORE.dirs(rel)
    reused = _try_reuse_prior_baseline(target_source, seed, train_df, val_df, test_df, root_rel, rel)
    if reused is not None:
        return reused
    completed = find_existing(f"{rel}/COMPLETED.json")
'''
    code = _once(code, baseline_start, baseline_new, 'baseline reuse hook')

    # This revision changes segmentation supervision. A V4.4 RimGraph
    # checkpoint must never be imported/resumed into V4.5. The new run_name and
    # code_revision already enforce this; add an explicit startup statement.
    main_marker = 'ALL_METRICS, ALL_PREDS = [], []\n'
    main_new = 'print("[V4.5] RimGraph checkpoints from earlier revisions are intentionally NOT reused.", flush=True)\n' + main_marker
    code = _once(code, main_marker, main_new, 'fresh RimGraph statement')

    return code
