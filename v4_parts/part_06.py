# ---------------------------- 6. OPTIONAL TUNING, FIGURES, SCIENTIFIC REPORT ----------------------------
def tune_full_source_only(train_df, val_df, target_source, seed, root_rel):
    if not CFG["run_optuna"]: return {}
    import optuna
    domain_map = {s: i for i, s in enumerate(sorted(train_df.source.unique()))}
    train_loader, sampler = make_loader(train_df, True, seed, domain_map)
    val_loader, _ = make_loader(val_df, False, seed, domain_map)
    base = {k: CFG[k] for k in ["lr", "weight_decay", "dropout", "lambda_seg", "lambda_domain", "lambda_proto"]}

    def objective(trial):
        CFG["lr"] = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
        CFG["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 2e-3, log=True)
        CFG["dropout"] = trial.suggest_float("dropout", 0.15, 0.40)
        CFG["lambda_seg"] = trial.suggest_float("lambda_seg", 0.5, 1.5)
        CFG["lambda_domain"] = trial.suggest_float("lambda_domain", 0.01, 0.08, log=True)
        CFG["lambda_proto"] = trial.suggest_float("lambda_proto", 0.01, 0.08, log=True)
        model = RimGraphV4(num_domains=len(domain_map)).to(DEVICE)
        optimizer = make_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=CFG["mixed_precision"] and DEVICE.type == "cuda")
        va = None
        for ep in range(1, int(CFG["tune_epochs"]) + 1):
            run_full_epoch(model, train_loader, optimizer, scaler, ep, sampler)
            va = run_full_epoch(model, val_loader, None, None, ep)
            trial.report(float(np.nan_to_num(va["auroc"])), ep)
            if trial.should_prune(): raise optuna.TrialPruned()
        score = 0.65 * np.nan_to_num(va["auroc"]) + 0.20 * np.nan_to_num(va["auprc"]) + 0.075 * va["dice_disc"] + 0.075 * va["dice_cup"]
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return float(score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=int(CFG["optuna_trials"]), show_progress_bar=False)
    STORE.save_df(study.trials_dataframe(), f"{root_rel}/optuna_trials.csv")
    STORE.save_json(study.best_params, f"{root_rel}/best_hyperparameters.json")
    for k, v in base.items(): CFG[k] = v
    return study.best_params


def save_training_curves(history_path, rel, title):
    history = pd.read_csv(history_path)
    if history.empty: return
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(1, 3, 1); ax.plot(history.epoch, history.train_loss, label="train"); ax.plot(history.epoch, history.val_loss, label="validation"); ax.set_title("Loss"); ax.legend(); ax.grid(alpha=.2)
    ax = fig.add_subplot(1, 3, 2); ax.plot(history.epoch, history.val_auroc, label="AUROC"); ax.plot(history.epoch, history.val_auprc, label="AUPRC"); ax.set_ylim(0, 1.02); ax.set_title("Source-validation discrimination"); ax.legend(); ax.grid(alpha=.2)
    ax = fig.add_subplot(1, 3, 3)
    if "val_dice_disc" in history.columns:
        ax.plot(history.epoch, history.val_dice_disc, label="Disc Dice"); ax.plot(history.epoch, history.val_dice_cup, label="Cup Dice")
    else:
        ax.plot(history.epoch, history.val_f1, label="F1")
    ax.set_ylim(0, 1.02); ax.set_title("Anatomy / threshold metric"); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(title); plt.tight_layout(); STORE.save_figure(fig, f"{rel}/training_curves.png"); plt.close(fig)


def save_discrimination_figure(pred, metrics, rel, title):
    y = pred.label.values; p = pred.prob_calibrated.values; threshold = metrics["threshold"]
    fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(1, 3, 1); ax.plot(fpr, tpr, label=f"AUC={metrics['auroc']:.3f}"); ax.plot([0, 1], [0, 1], '--'); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC"); ax.legend(); ax.grid(alpha=.2)
    ax = fig.add_subplot(1, 3, 2); ax.plot(recall, precision, label=f"AP={metrics['auprc']:.3f}"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall"); ax.legend(); ax.grid(alpha=.2)
    ax = fig.add_subplot(1, 3, 3); ax.imshow(cm); ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2): ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.suptitle(title); plt.tight_layout(); STORE.save_figure(fig, f"{rel}/discrimination.png"); plt.close(fig)

    frac, mean = calibration_curve(y, p, n_bins=min(10, max(3, len(y) // 20)), strategy="quantile")
    fig = plt.figure(figsize=(6, 5)); plt.plot(mean, frac, "o-", label=f"ECE={metrics['ece']:.3f}"); plt.plot([0, 1], [0, 1], '--'); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction"); plt.title(title + " — calibration"); plt.legend(); plt.grid(alpha=.2); STORE.save_figure(fig, f"{rel}/calibration.png"); plt.close(fig)


def denormalize_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    return (x * std + mean).clamp(0, 1)


def save_full_xai(model, test_df, seed, rel):
    if model is None or len(test_df) == 0: return
    sample = test_df.groupby("label", group_keys=False).head(max(1, CFG["n_visual_examples"] // 2)).head(CFG["n_visual_examples"])
    ds = GlaucomaDataset(sample, train=False, seed=seed, domain_map={s: 0 for s in sample.source.unique()})
    fig = plt.figure(figsize=(18, 4 * len(ds)))
    model.eval()
    for row_idx in range(len(ds)):
        batch = ds[row_idx]
        x = batch["image"][None].to(DEVICE).requires_grad_(True)
        out = model(x, stage="full", grl_lambda=0.0, return_features=True)
        model.zero_grad(set_to_none=True); out["logit"].sum().backward()
        p1 = out["p1"]; grad = p1.grad
        weights = grad.mean((2, 3), keepdim=True)
        cam = F.relu((weights * p1).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        image = denormalize_image(x[0]).detach().cpu().permute(1, 2, 0).numpy()
        seg = torch.sigmoid(out["seg"])[0].detach().cpu().numpy()
        overlay = np.zeros_like(image); overlay[..., 1] = seg[0]; overlay[..., 0] = seg[1]
        att = out["graph_attention"][0, :CFG["num_sectors"], :CFG["num_sectors"]].mean(0).detach().cpu().numpy()
        gates = out["gates"][0].detach().cpu().numpy()
        prob = float(torch.sigmoid(out["logit"])[0].detach().cpu())
        ax = fig.add_subplot(len(ds), 5, row_idx * 5 + 1); ax.imshow(image); ax.set_title(f"True={int(batch['label'])}, P={prob:.3f}"); ax.axis("off")
        ax = fig.add_subplot(len(ds), 5, row_idx * 5 + 2); ax.imshow(image); ax.imshow(overlay, alpha=.45); ax.set_title("Predicted disc/cup"); ax.axis("off")
        ax = fig.add_subplot(len(ds), 5, row_idx * 5 + 3); ax.imshow(image); ax.imshow(cam.detach().cpu(), cmap="jet", alpha=.45); ax.set_title("FPN Grad-CAM"); ax.axis("off")
        ax = fig.add_subplot(len(ds), 5, row_idx * 5 + 4); ax.bar(np.arange(1, len(att) + 1), att); ax.set_title("Polar-sector attention"); ax.set_xlabel("Sector")
        ax = fig.add_subplot(len(ds), 5, row_idx * 5 + 5); ax.bar(["Global", "Local", "Graph", "Struct"], gates); ax.set_ylim(0, 1); ax.set_title("Fusion gates"); ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); STORE.save_figure(fig, f"{rel}/qualitative_xai.png", dpi=160); plt.close(fig)


def scientific_diagnostic(summary, comparison):
    lines = ["# RimGraph-DG V4 Scientific Diagnostic Report", ""]
    full = summary[summary.model_type == "rimgraph_v4"]
    baseline = summary[summary.model_type == "global_baseline"]
    if not baseline.empty:
        lines.append(f"- Global baseline mean external AUROC: **{baseline.auroc.mean():.4f}**")
        lines.append(f"- Global baseline worst-domain AUROC: **{baseline.auroc.min():.4f}**")
    if not full.empty:
        lines.append(f"- RimGraph V4 mean external AUROC: **{full.auroc.mean():.4f}**")
        lines.append(f"- RimGraph V4 worst-domain AUROC: **{full.auroc.min():.4f}**")
        lines.append(f"- Mean disc Dice: **{full.dice_disc.mean():.4f}**")
        lines.append(f"- Mean cup Dice: **{full.dice_cup.mean():.4f}**")
        lines.append(f"- Mean ECE: **{full.ece.mean():.4f}**")
    lines += ["", "## Interpretation"]
    if not comparison.empty:
        delta = comparison.delta_auroc.mean()
        if delta > 0.02: lines.append("- Anatomy/graph fusion provides a meaningful average gain over the global baseline.")
        elif delta < -0.02: lines.append("- The full model underperforms the global baseline; graph/anatomy claims are not yet supported and ablation is required.")
        else: lines.append("- Full-model discrimination is close to the baseline; novelty must be justified through calibration, robustness, segmentation, and ablations rather than AUROC alone.")
    if not full.empty:
        if full.dice_cup.mean() < 0.75: lines.append("- Cup segmentation is too weak for reliable structural graph interpretation. Improve masks/decoder before publication claims.")
        elif full.dice_cup.mean() < 0.82: lines.append("- Cup segmentation is usable but still a major limitation for anatomy-derived claims.")
        else: lines.append("- Cup segmentation is sufficiently strong for further structural analysis, subject to visual audit.")
        if full.auroc.min() < 0.80: lines.append("- Severe domain-generalization weakness remains; the current result is not journal-ready.")
        elif full.auroc.min() < 0.85: lines.append("- Worst-domain performance needs substantial improvement before a strong Q2 submission.")
        elif full.auroc.mean() >= 0.88 and full.auroc.min() >= 0.85: lines.append("- Results may support a Q2-oriented manuscript after multi-seed statistics, ablations, and an additional untouched external dataset.")
        if full.ece.mean() > 0.08: lines.append("- Calibration is poor; probabilities should not be presented as clinically reliable.")
        elif full.ece.mean() <= 0.05: lines.append("- Calibration is promising, but reliability diagrams and confidence intervals must still be reported.")
    lines += ["", "## Required publication checks", "- Run at least three seeds for every held-out domain.", "- Report bootstrap confidence intervals and baseline-versus-full deltas.", "- Do not claim clock-hour anatomy for images with unknown laterality.", "- Manually inspect segmentation and XAI failures.", "- Add one completely untouched external dataset for a stronger Q1/Q2 claim."]
    return "\n".join(lines)


ALL_METRICS, ALL_PREDS = [], []
for seed in CFG["seeds"]:
    seed_everything(int(seed))
    for target in CFG["fold_targets"]:
        section(f"Seed {seed} — held-out {target}")
        root_rel = f"folds/{target}/seed_{seed}"
        train_df, val_df, test_df = prepare_fold_data(target, int(seed), root_rel)
        baseline_metrics = None
        if CFG["run_global_baseline"]:
            baseline_metrics, baseline_pred = train_baseline(target, int(seed), train_df, val_df, test_df, root_rel)
            baseline_pred["held_out_source"] = target; baseline_pred["model_type"] = "global_baseline"; baseline_pred["seed"] = int(seed)
            ALL_METRICS.append(baseline_metrics); ALL_PREDS.append(baseline_pred)
            b_rel = f"{root_rel}/global_baseline/figures"; STORE.dirs(b_rel)
            save_training_curves(find_existing(f"{root_rel}/global_baseline/history.csv"), b_rel, f"Global baseline — {target}")
            save_discrimination_figure(baseline_pred, baseline_metrics, b_rel, f"Global baseline — held-out {target}")
        full_metrics = None
        if CFG["run_full_model"]:
            tuned = tune_full_source_only(train_df, val_df, target, int(seed), root_rel)
            original = {k: CFG[k] for k in tuned}
            for k, v in tuned.items(): CFG[k] = v
            full_metrics, full_pred, model = train_full(target, int(seed), train_df, val_df, test_df, root_rel)
            for k, v in original.items(): CFG[k] = v
            full_pred["held_out_source"] = target; full_pred["model_type"] = "rimgraph_v4"; full_pred["seed"] = int(seed)
            ALL_METRICS.append(full_metrics); ALL_PREDS.append(full_pred)
            f_rel = f"{root_rel}/rimgraph_v4/figures"; STORE.dirs(f_rel)
            save_training_curves(find_existing(f"{root_rel}/rimgraph_v4/history.csv"), f_rel, f"RimGraph V4 — {target}")
            save_discrimination_figure(full_pred, full_metrics, f_rel, f"RimGraph V4 — held-out {target}")
            save_full_xai(model, test_df, int(seed), f_rel)
            sector_cols = [f"sector_{j+1:02d}" for j in range(CFG["num_sectors"])]
            if all(c in full_pred.columns for c in sector_cols):
                sector_summary = full_pred.groupby("label")[sector_cols].mean().T.reset_index().rename(columns={"index": "sector", 0: "normal_mean", 1: "glaucoma_mean"})
                STORE.save_df(sector_summary, f"{root_rel}/rimgraph_v4/xai_sector_summary.csv")
        fold_rows = [m for m in [baseline_metrics, full_metrics] if m is not None]
        display(pd.DataFrame(fold_rows)[["model_type", "held_out_source", "auroc", "auprc", "sensitivity", "specificity", "f1", "mcc", "ece"] + (["dice_disc", "dice_cup"] if full_metrics is not None else [])].fillna(""))
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

summary = pd.DataFrame(ALL_METRICS)
predictions = pd.concat(ALL_PREDS, ignore_index=True) if ALL_PREDS else pd.DataFrame()
STORE.save_df(summary, "fold_model_summary.csv"); STORE.save_df(predictions, "all_external_predictions.csv")

aggregate_rows = []
for model_type, group in summary.groupby("model_type"):
    for metric in ["auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "brier", "ece", "dice_disc", "dice_cup"]:
        if metric not in group.columns: continue
        aggregate_rows.append({
            "model_type": model_type, "metric": metric, "mean": group[metric].mean(),
            "std": group[metric].std(ddof=1),
            "worst_domain": group[metric].max() if metric in {"brier", "ece"} else group[metric].min(),
        })
aggregate = pd.DataFrame(aggregate_rows); STORE.save_df(aggregate, "aggregate_metrics.csv")

comparison = pd.DataFrame()
if set(summary.model_type.unique()) >= {"global_baseline", "rimgraph_v4"}:
    key = ["held_out_source", "seed"]
    b = summary[summary.model_type == "global_baseline"].set_index(key)
    f = summary[summary.model_type == "rimgraph_v4"].set_index(key)
    common = b.index.intersection(f.index)
    rows = []
    for idx in common:
        rows.append({"held_out_source": idx[0], "seed": idx[1], "baseline_auroc": b.loc[idx, "auroc"], "full_auroc": f.loc[idx, "auroc"], "delta_auroc": f.loc[idx, "auroc"] - b.loc[idx, "auroc"], "baseline_auprc": b.loc[idx, "auprc"], "full_auprc": f.loc[idx, "auprc"], "delta_auprc": f.loc[idx, "auprc"] - b.loc[idx, "auprc"]})
    comparison = pd.DataFrame(rows); STORE.save_df(comparison, "baseline_vs_full.csv")

report = scientific_diagnostic(summary, comparison)
STORE.save_text(report, "scientific_diagnostic_report.md")

manifest = []
for p in sorted(LOCAL_RUN.rglob("*")):
    if p.is_file(): manifest.append({"file": str(p.relative_to(LOCAL_RUN)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
STORE.save_df(pd.DataFrame(manifest), "artifact_manifest.csv")
zip_path = shutil.make_archive(str(LOCAL_RUN), "zip", root_dir=LOCAL_RUN)
shutil.copy2(zip_path, DRIVE_RUN.parent / f"{CFG['run_name']}.zip")

clear_output(wait=True)
display(Markdown("# ✅ RimGraph-DG V4 run completed"))
display(Markdown(f"**Colab:** `{LOCAL_RUN}`  \n**Drive:** `{DRIVE_RUN}`  \n**ZIP:** `{DRIVE_RUN.parent / (CFG['run_name'] + '.zip')}`"))
display(Markdown("## External results")); display(summary.round(4))
if not comparison.empty:
    display(Markdown("## Baseline versus full model")); display(comparison.round(4))
display(Markdown(report))
