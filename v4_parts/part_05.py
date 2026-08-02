# ---------------------------- 5. TRAINING AND PREDICTION ----------------------------
def batch_to_device(batch):
    tensor_keys = ["image", "disc", "cup", "disc_valid", "cup_valid", "vcdr_valid", "vcdr", "label", "domain"]
    for key in tensor_keys:
        batch[key] = batch[key].to(DEVICE, non_blocking=True)
    return batch


def segmentation_batch_stats(seg, disc, cup, disc_valid, cup_valid):
    prob = torch.sigmoid(seg)
    stats = {}
    for name, pred, target, valid in [
        ("disc", prob[:, 0:1], disc, disc_valid),
        ("cup", prob[:, 1:2], cup, cup_valid),
    ]:
        binary = (pred > 0.5).float()
        inter = (binary * target).sum((1, 2, 3))
        den = (binary + target).sum((1, 2, 3))
        union = (binary + target - binary * target).sum((1, 2, 3))
        dice = (2 * inter + 1) / (den + 1)
        iou = (inter + 1) / (union + 1)
        stats[name] = {
            "dice_sum": float((dice * valid).sum().detach().cpu()),
            "iou_sum": float((iou * valid).sum().detach().cpu()),
            "count": float(valid.sum().detach().cpu()),
        }
    return stats


def merge_seg_stats(total, batch_stats):
    for name in ["disc", "cup"]:
        for key in ["dice_sum", "iou_sum", "count"]:
            total[name][key] += batch_stats[name][key]


def final_seg_stats(total):
    out = {}
    for name in ["disc", "cup"]:
        count = max(1.0, total[name]["count"])
        out[f"dice_{name}"] = total[name]["dice_sum"] / count
        out[f"iou_{name}"] = total[name]["iou_sum"] / count
        out[f"n_mask_{name}"] = int(total[name]["count"])
    return out


def run_baseline_epoch(model, loader, optimizer=None, scaler=None, epoch=0, sampler=None):
    train = optimizer is not None
    model.train(train)
    if train:
        loader.dataset.set_epoch(epoch)
        if sampler is not None: sampler.set_epoch(epoch)
    all_y, all_p, losses = [], [], []
    if train: optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(loader):
        batch = batch_to_device(batch)
        with torch.set_grad_enabled(train), torch.autocast(device_type=DEVICE.type, enabled=CFG["mixed_precision"] and DEVICE.type == "cuda"):
            out = model(batch["image"])
            loss = binary_focal_with_logits(out["logit"], batch["label"], CFG["focal_gamma"])
            scaled = loss / CFG["grad_accum"]
        if train:
            scaler.scale(scaled).backward()
            if (bi + 1) % CFG["grad_accum"] == 0 or bi + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        all_y.extend(batch["label"].detach().cpu().numpy())
        all_p.extend(torch.sigmoid(out["logit"]).detach().cpu().numpy())
        if CFG["fast_dev_run"] and bi >= 2: break
    metrics = classification_metrics(np.asarray(all_y), np.asarray(all_p), 0.5)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def run_full_epoch(model, loader, optimizer=None, scaler=None, epoch=0, sampler=None):
    train = optimizer is not None
    model.train(train)
    if train:
        loader.dataset.set_epoch(epoch)
        if sampler is not None: sampler.set_epoch(epoch)
    stage = full_stage(epoch)
    all_y, all_p, losses = [], [], []
    seg_total = {n: {k: 0.0 for k in ["dice_sum", "iou_sum", "count"]} for n in ["disc", "cup"]}
    if train: optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(loader):
        batch = batch_to_device(batch)
        progress = (epoch + bi / max(1, len(loader))) / max(1, CFG["full_epochs"])
        grl = (2 / (1 + math.exp(-10 * progress)) - 1) if train and stage == "full" else 0.0
        with torch.set_grad_enabled(train), torch.autocast(device_type=DEVICE.type, enabled=CFG["mixed_precision"] and DEVICE.type == "cuda"):
            out = model(batch["image"], stage=stage, grl_lambda=grl)
            l_cls = binary_focal_with_logits(out["logit"], batch["label"], CFG["focal_gamma"])
            l_seg = segmentation_loss(out["seg"], batch["disc"], batch["cup"], batch["disc_valid"], batch["cup_valid"])
            loss = l_cls + CFG["lambda_seg"] * l_seg
            if stage in {"anatomy", "full"}:
                l_vcdr_raw = F.smooth_l1_loss(out["vcdr_reg"], batch["vcdr"], reduction="none")
                l_vcdr = (l_vcdr_raw * batch["vcdr_valid"]).sum() / batch["vcdr_valid"].sum().clamp_min(1)
                l_cons_raw = F.smooth_l1_loss(out["vcdr_reg"], out["vcdr_mask"].detach(), reduction="none")
                l_cons = (l_cons_raw * batch["vcdr_valid"]).sum() / batch["vcdr_valid"].sum().clamp_min(1)
                loss = loss + CFG["lambda_vcdr"] * l_vcdr + CFG["lambda_cons"] * l_cons
            if stage == "full":
                l_domain = F.cross_entropy(out["domain"], batch["domain"])
                l_proto = prototype_loss(out["fused"], batch["label"], batch["domain"])
                loss = loss + CFG["lambda_domain"] * l_domain + CFG["lambda_proto"] * l_proto
            scaled = loss / CFG["grad_accum"]
        if train:
            scaler.scale(scaled).backward()
            if (bi + 1) % CFG["grad_accum"] == 0 or bi + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        all_y.extend(batch["label"].detach().cpu().numpy())
        all_p.extend(torch.sigmoid(out["logit"]).detach().cpu().numpy())
        merge_seg_stats(seg_total, segmentation_batch_stats(out["seg"], batch["disc"], batch["cup"], batch["disc_valid"], batch["cup_valid"]))
        if CFG["fast_dev_run"] and bi >= 2: break
    metrics = classification_metrics(np.asarray(all_y), np.asarray(all_p), 0.5)
    metrics.update(final_seg_stats(seg_total)); metrics["loss"] = float(np.mean(losses)); metrics["stage"] = stage
    return metrics


@torch.no_grad()
def predict_baseline(model, loader):
    model.eval(); rows = []
    for batch in loader:
        batch = batch_to_device(batch); out = model(batch["image"])
        prob = torch.sigmoid(out["logit"]).cpu().numpy()
        for i in range(len(prob)):
            rows.append({"path": batch["path"][i], "source": batch["source_name"][i], "label": int(batch["label"][i].cpu()), "prob_raw": float(prob[i])})
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_full(model, loader):
    model.eval(); rows = []
    seg_total = {n: {k: 0.0 for k in ["dice_sum", "iou_sum", "count"]} for n in ["disc", "cup"]}
    for batch in loader:
        batch = batch_to_device(batch); out = model(batch["image"], stage="full", grl_lambda=0.0)
        prob = torch.sigmoid(out["logit"]).cpu().numpy()
        merge_seg_stats(seg_total, segmentation_batch_stats(out["seg"], batch["disc"], batch["cup"], batch["disc_valid"], batch["cup_valid"]))
        for i in range(len(prob)):
            att = out["graph_attention"][i, :CFG["num_sectors"], :CFG["num_sectors"]].mean(0).cpu().numpy()
            row = {
                "path": batch["path"][i], "source": batch["source_name"][i], "label": int(batch["label"][i].cpu()),
                "prob_raw": float(prob[i]), "vcdr_pred": float(out["vcdr_reg"][i].cpu()),
                "vcdr_mask": float(out["vcdr_mask"][i].cpu()), "quality": float(out["quality"][i].cpu()),
            }
            for j, name in enumerate(["global", "local", "graph", "struct"]): row[f"gate_{name}"] = float(out["gates"][i, j].cpu())
            for j in range(CFG["num_sectors"]): row[f"sector_{j+1:02d}"] = float(att[j])
            rows.append(row)
    return pd.DataFrame(rows), final_seg_stats(seg_total)


def make_optimizer(model):
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": CFG["lr"] * CFG["backbone_lr_scale"]},
        {"params": heads, "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])


def checkpoint_compatible(checkpoint, model_type, seed):
    cfg = checkpoint.get("cfg", {})
    return cfg.get("code_revision") == CFG["code_revision"] and checkpoint.get("model_type") == model_type and int(checkpoint.get("seed", -1)) == int(seed)


def prepare_fold_data(target_source, seed, root_rel):
    test_df = META[META.source == target_source].copy().reset_index(drop=True)
    train_pool = META[META.source != target_source].copy().reset_index(drop=True)
    train_pool, leaked = remove_fold_leakage(train_pool, test_df)
    if not leaked.empty: STORE.save_df(leaked, f"{root_rel}/cross_source_leakage_removed.csv")
    train_df, val_df = source_stratified_split(train_pool, CFG["val_fraction"], seed)
    if CFG["fast_dev_run"]:
        train_df = train_df.groupby(["source", "label"], group_keys=False).head(6).reset_index(drop=True)
        val_df = val_df.groupby(["source", "label"], group_keys=False).head(4).reset_index(drop=True)
        test_df = test_df.groupby(["source", "label"], group_keys=False).head(6).reset_index(drop=True)
    STORE.save_df(train_df, f"{root_rel}/train_split.csv"); STORE.save_df(val_df, f"{root_rel}/val_split.csv"); STORE.save_df(test_df, f"{root_rel}/test_split.csv")
    return train_df, val_df, test_df


def finalize_predictions(val_pred, test_pred, target_source, model_type, seed, rel, seg_metrics=None):
    method, temperature, cal_table = select_calibration(val_pred)
    STORE.save_df(cal_table, f"{rel}/calibration_selection.csv")
    val_cal = apply_temperature(val_pred.prob_raw.values, temperature)
    threshold = choose_threshold(val_pred.label.values, val_cal)
    test_pred = test_pred.copy(); test_pred["prob_calibrated"] = apply_temperature(test_pred.prob_raw.values, temperature)
    metrics = classification_metrics(test_pred.label.values, test_pred.prob_calibrated.values, threshold)
    if seg_metrics: metrics.update(seg_metrics)
    for metric_name in ["auroc", "auprc", "sensitivity", "specificity"]:
        lo, hi = bootstrap_ci(test_pred.label.values, test_pred.prob_calibrated.values, metric_name, threshold, CFG["bootstrap_samples"], seed)
        metrics[f"{metric_name}_ci_low"] = lo; metrics[f"{metric_name}_ci_high"] = hi
    metrics.update({
        "held_out_source": target_source, "model_type": model_type, "seed": int(seed),
        "calibration_method": method, "temperature": temperature, "n_test": len(test_pred),
    })
    STORE.save_df(test_pred, f"{rel}/test_predictions.csv"); STORE.save_json(metrics, f"{rel}/metrics.json")
    return metrics, test_pred


def train_baseline(target_source, seed, train_df, val_df, test_df, root_rel):
    rel = f"{root_rel}/global_baseline"; STORE.dirs(rel)
    completed = find_existing(f"{rel}/COMPLETED.json")
    if CFG["resume"] and completed:
        saved = json.loads(Path(completed).read_text())
        if saved.get("code_revision") == CFG["code_revision"]:
            pred = pd.read_csv(find_existing(f"{rel}/test_predictions.csv")); return saved["metrics"], pred
    domain_map = {s: i for i, s in enumerate(sorted(train_df.source.unique()))}
    train_loader, sampler = make_loader(train_df, True, seed, domain_map)
    val_loader, _ = make_loader(val_df, False, seed, domain_map)
    test_loader, _ = make_loader(test_df, False, seed, {target_source: 0})
    model = GlobalBaseline().to(DEVICE); optimizer = make_optimizer(model); scheduler = make_scheduler(optimizer, CFG["baseline_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=CFG["mixed_precision"] and DEVICE.type == "cuda")
    history, best_score, best_epoch, start = [], -np.inf, 0, 1
    last = find_existing(f"{rel}/last_checkpoint.pt")
    if CFG["resume"] and last:
        ck = torch.load(last, map_location=DEVICE, weights_only=False)
        if checkpoint_compatible(ck, "global_baseline", seed):
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
            history, best_score, best_epoch, start = ck["history"], ck["best_score"], ck["best_epoch"], ck["epoch"] + 1
    for epoch in range(start, CFG["baseline_epochs"] + 1):
        tr = run_baseline_epoch(model, train_loader, optimizer, scaler, epoch, sampler)
        va = run_baseline_epoch(model, val_loader, None, None, epoch)
        scheduler.step(); score = 0.70 * np.nan_to_num(va["auroc"]) + 0.30 * np.nan_to_num(va["auprc"])
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "val_auroc": va["auroc"], "val_auprc": va["auprc"], "val_f1": va["f1"], "selection_score": score, "lr": optimizer.param_groups[-1]["lr"]})
        improved = score > best_score
        if improved: best_score, best_epoch = score, epoch
        ck = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history, "best_score": best_score, "best_epoch": best_epoch, "cfg": CFG, "model_type": "global_baseline", "seed": seed}
        if CFG["save_every_epoch"]: save_checkpoint(ck, f"{rel}/last_checkpoint.pt")
        if improved: save_checkpoint(ck, f"{rel}/best_model.pt")
        STORE.save_df(pd.DataFrame(history), f"{rel}/history.csv")
        if epoch - best_epoch >= CFG["patience"]: break
    best = torch.load(find_existing(f"{rel}/best_model.pt"), map_location=DEVICE, weights_only=False); model.load_state_dict(best["model"])
    val_pred = predict_baseline(model, val_loader); test_pred = predict_baseline(model, test_loader)
    metrics, test_pred = finalize_predictions(val_pred, test_pred, target_source, "global_baseline", seed, rel)
    metrics["best_epoch"] = best_epoch; STORE.save_json(metrics, f"{rel}/metrics.json")
    STORE.save_json({"metrics": metrics, "code_revision": CFG["code_revision"]}, f"{rel}/COMPLETED.json")
    return metrics, test_pred


def train_full(target_source, seed, train_df, val_df, test_df, root_rel):
    rel = f"{root_rel}/rimgraph_v4"; STORE.dirs(rel)
    completed = find_existing(f"{rel}/COMPLETED.json")
    if CFG["resume"] and completed:
        saved = json.loads(Path(completed).read_text())
        if saved.get("code_revision") == CFG["code_revision"]:
            pred = pd.read_csv(find_existing(f"{rel}/test_predictions.csv")); return saved["metrics"], pred, None
    domain_map = {s: i for i, s in enumerate(sorted(train_df.source.unique()))}
    train_loader, sampler = make_loader(train_df, True, seed, domain_map)
    val_loader, _ = make_loader(val_df, False, seed, domain_map)
    test_loader, _ = make_loader(test_df, False, seed, {target_source: 0})
    model = RimGraphV4(num_domains=len(domain_map)).to(DEVICE); optimizer = make_optimizer(model); scheduler = make_scheduler(optimizer, CFG["full_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=CFG["mixed_precision"] and DEVICE.type == "cuda")
    history, best_score, best_epoch, start = [], -np.inf, 0, 1
    last = find_existing(f"{rel}/last_checkpoint.pt")
    if CFG["resume"] and last:
        ck = torch.load(last, map_location=DEVICE, weights_only=False)
        if checkpoint_compatible(ck, "rimgraph_v4", seed):
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
            history, best_score, best_epoch, start = ck["history"], ck["best_score"], ck["best_epoch"], ck["epoch"] + 1
    for epoch in range(start, CFG["full_epochs"] + 1):
        tr = run_full_epoch(model, train_loader, optimizer, scaler, epoch, sampler)
        va = run_full_epoch(model, val_loader, None, None, epoch)
        scheduler.step()
        score = 0.65 * np.nan_to_num(va["auroc"]) + 0.20 * np.nan_to_num(va["auprc"]) + 0.075 * va["dice_disc"] + 0.075 * va["dice_cup"]
        history.append({
            "epoch": epoch, "stage": va["stage"], "train_loss": tr["loss"], "val_loss": va["loss"],
            "val_auroc": va["auroc"], "val_auprc": va["auprc"], "val_f1": va["f1"],
            "val_dice_disc": va["dice_disc"], "val_dice_cup": va["dice_cup"],
            "selection_score": score, "lr": optimizer.param_groups[-1]["lr"],
        })
        improved = score > best_score
        if improved: best_score, best_epoch = score, epoch
        ck = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history, "best_score": best_score, "best_epoch": best_epoch, "cfg": CFG, "model_type": "rimgraph_v4", "seed": seed}
        if CFG["save_every_epoch"]: save_checkpoint(ck, f"{rel}/last_checkpoint.pt")
        if improved: save_checkpoint(ck, f"{rel}/best_model.pt")
        STORE.save_df(pd.DataFrame(history), f"{rel}/history.csv")
        if epoch - best_epoch >= CFG["patience"] and epoch > CFG["anatomy_warmup_epochs"]: break
    best = torch.load(find_existing(f"{rel}/best_model.pt"), map_location=DEVICE, weights_only=False); model.load_state_dict(best["model"])
    val_pred, _ = predict_full(model, val_loader); test_pred, seg_metrics = predict_full(model, test_loader)
    metrics, test_pred = finalize_predictions(val_pred, test_pred, target_source, "rimgraph_v4", seed, rel, seg_metrics)
    metrics["best_epoch"] = best_epoch; STORE.save_json(metrics, f"{rel}/metrics.json")
    STORE.save_json({"metrics": metrics, "code_revision": CFG["code_revision"]}, f"{rel}/COMPLETED.json")
    return metrics, test_pred, model
