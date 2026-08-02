# ---------------------------- 4. LOSSES, METRICS, CALIBRATION ----------------------------
def binary_focal_with_logits(logits, targets, gamma=1.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    return ((1 - pt).pow(gamma) * bce).mean()


def masked_dice_loss(logit, target, valid):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((2, 3)); den = (p + target).sum((2, 3))
    loss = 1 - (2 * inter + 1) / (den + 1)
    return (loss.squeeze(1) * valid).sum() / valid.sum().clamp_min(1)


def masked_bce(logit, target, valid):
    raw = F.binary_cross_entropy_with_logits(logit, target, reduction="none").mean((1, 2, 3))
    return (raw * valid).sum() / valid.sum().clamp_min(1)


def boundary_map(x):
    return (F.max_pool2d(x, 3, 1, 1) - (-F.max_pool2d(-x, 3, 1, 1))).abs()


def segmentation_loss(seg, disc, cup, disc_valid, cup_valid):
    ld = masked_bce(seg[:, 0:1], disc, disc_valid) + masked_dice_loss(seg[:, 0:1], disc, disc_valid)
    lc = masked_bce(seg[:, 1:2], cup, cup_valid) + masked_dice_loss(seg[:, 1:2], cup, cup_valid)
    p = torch.sigmoid(seg)
    bd = F.l1_loss(boundary_map(p[:, 0:1]), boundary_map(disc), reduction="none").mean((1, 2, 3))
    bc = F.l1_loss(boundary_map(p[:, 1:2]), boundary_map(cup), reduction="none").mean((1, 2, 3))
    edge = (bd * disc_valid).sum() / disc_valid.sum().clamp_min(1) + (bc * cup_valid).sum() / cup_valid.sum().clamp_min(1)
    return ld + 1.5 * lc + 0.10 * edge


def prototype_loss(features, labels, domains):
    terms = []
    for cls in [0, 1]:
        protos = []
        for d in domains.unique():
            mask = (labels.long() == cls) & (domains == d)
            if mask.sum() > 0: protos.append(features[mask].mean(0))
        for i in range(len(protos)):
            for j in range(i + 1, len(protos)):
                terms.append(F.mse_loss(protos[i], protos[j]))
    return torch.stack(terms).mean() if terms else features.sum() * 0


def expected_calibration_error(y, p, bins=15):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any(): ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def sensitivity_at_specificity(y, p, target_specificity=0.95):
    fpr, tpr, _ = roc_curve(y, p)
    valid = np.where((1 - fpr) >= target_specificity)[0]
    return float(tpr[valid].max()) if len(valid) else 0.0


def classification_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": safe_auc(y, p),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "sensitivity_at_95_specificity": sensitivity_at_specificity(y, p, 0.95),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.c_[1 - p, p], labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def dice_and_iou(pred, target, valid):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum((2, 3)); union = (pred + target - pred * target).sum((2, 3))
    dice = (2 * inter + 1) / ((pred + target).sum((2, 3)) + 1)
    iou = (inter + 1) / (union + 1)
    v = valid[:, None]
    return (
        ((dice * v).sum(0) / v.sum().clamp_min(1)).detach().cpu().numpy(),
        ((iou * v).sum(0) / v.sum().clamp_min(1)).detach().cpu().numpy(),
    )


def bootstrap_ci(y, p, metric_name="auroc", threshold=0.5, n_boot=500, seed=2029):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    rng = np.random.default_rng(seed); values = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0: return np.nan, np.nan
    for _ in range(int(n_boot)):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        if metric_name == "auroc": value = safe_auc(y[idx], p[idx])
        elif metric_name == "auprc": value = average_precision_score(y[idx], p[idx])
        elif metric_name == "sensitivity": value = classification_metrics(y[idx], p[idx], threshold)["sensitivity"]
        elif metric_name == "specificity": value = classification_metrics(y[idx], p[idx], threshold)["specificity"]
        else: raise ValueError(metric_name)
        values.append(value)
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__(); self.log_t = nn.Parameter(torch.zeros(1))
    @property
    def temperature(self): return self.log_t.exp().clamp(0.05, 10)
    def forward(self, logits): return logits / self.temperature


def fit_temperature(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logits = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    targets = torch.tensor(np.asarray(y), dtype=torch.float32)
    model = TemperatureScaler(); opt = torch.optim.LBFGS(model.parameters(), lr=0.05, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(model(logits), targets); loss.backward(); return loss
    opt.step(closure)
    return float(model.temperature.detach())


def fit_robust_temperature(y, p, domains, steps=200):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logits = torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    targets = torch.tensor(np.asarray(y), dtype=torch.float32)
    domains = np.asarray(domains); log_t = nn.Parameter(torch.zeros(1)); opt = torch.optim.Adam([log_t], lr=0.03)
    for _ in range(steps):
        t = log_t.exp().clamp(0.05, 10); losses = []
        for d in pd.unique(domains):
            idx = torch.tensor(domains == d)
            losses.append(F.binary_cross_entropy_with_logits(logits[idx] / t, targets[idx]))
        loss = 0.10 * torch.logsumexp(torch.stack(losses) / 0.10, dim=0)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(log_t.exp().clamp(0.05, 10).detach())


def apply_temperature(p, temperature):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) / float(temperature)
    return 1 / (1 + np.exp(-z))


def select_calibration(val_pred):
    candidates = [("identity", 1.0)]
    candidates.append(("standard_temperature", fit_temperature(val_pred.label.values, val_pred.prob_raw.values)))
    candidates.append(("robust_temperature", fit_robust_temperature(val_pred.label.values, val_pred.prob_raw.values, val_pred.source.values)))
    rows = []
    for name, t in candidates:
        pc = apply_temperature(val_pred.prob_raw.values, t)
        pooled = classification_metrics(val_pred.label.values, pc, 0.5)
        worst_nll = max(classification_metrics(g.label.values, apply_temperature(g.prob_raw.values, t), 0.5)["nll"] for _, g in val_pred.groupby("source"))
        rows.append({"method": name, "temperature": t, "pooled_nll": pooled["nll"], "pooled_ece": pooled["ece"], "worst_source_nll": worst_nll})
    table = pd.DataFrame(rows).sort_values(["worst_source_nll", "pooled_nll", "pooled_ece"]).reset_index(drop=True)
    best = table.iloc[0]
    return str(best.method), float(best.temperature), table


def choose_threshold(y, p):
    if CFG["threshold_policy"] == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        finite = np.isfinite(thr)
        if not finite.any(): return 0.5
        finite_idx = np.where(finite)[0]
        best = finite_idx[np.argmax((tpr - fpr)[finite_idx])]
        return float(np.clip(thr[best], 0.0, 1.0))
    return 0.5


def full_stage(epoch):
    if epoch <= int(CFG["seg_warmup_epochs"]): return "global_seg"
    if epoch <= int(CFG["anatomy_warmup_epochs"]): return "anatomy"
    return "full"


def make_scheduler(optimizer, total_epochs):
    warmup = max(1, min(3, total_epochs // 5))
    def factor(epoch):
        if epoch < warmup: return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total_epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
