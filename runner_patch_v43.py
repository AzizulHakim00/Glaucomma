"""V4.3 scientific guards: active prototypes, valid full-stage selection, and consistent anatomy metrics."""


def once(code, old, new, label):
    n = code.count(old)
    if n != 1:
        raise RuntimeError(f"V4.3 expected one {label}, found {n}")
    return code.replace(old, new, 1)


def block(code, start, end, replacement, label):
    a = code.find(start)
    b = code.find(end, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        raise RuntimeError(f"V4.3 could not locate {label}")
    return code[:a] + replacement.rstrip() + code[b:]


def extract(code, start, end, label):
    a = code.find(start)
    b = code.find(end, a + 1) if a >= 0 else -1
    if a < 0 or b < 0:
        raise RuntimeError(f"V4.3 could not locate {label}")
    return a, b, code[a:b]


def apply_v43(code):
    checks = '''if int(CFG["batch_size"]) < 1 or int(CFG["grad_accum"]) < 1:
    raise ValueError("batch_size and grad_accum must both be at least 1")
if CFG["run_full_model"] and int(CFG["full_epochs"]) <= int(CFG["anatomy_warmup_epochs"]):
    raise ValueError("full_epochs must exceed anatomy_warmup_epochs so the full graph/domain stage is trained")

'''
    code = once(code, 'IN_COLAB = "google.colab" in sys.modules\n', checks + 'IN_COLAB = "google.colab" in sys.modules\n', "config validation")

    sampler = '''class BalancedSourceClassBatchSampler(Sampler):
    """Prototype-aware balancing across T4-safe accumulation windows.

    With batch_size=2 and grad_accum=2, each micro-batch contains the same
    class from two different source domains; the next micro-batch uses the
    other class. This activates class-conditional cross-domain prototypes and
    covers all four source/class groups before every optimizer update.
    """
    def __init__(self, df, batch_size, seed=2029):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1: raise ValueError("batch_size must be at least 1")
        self.groups = {key: np.asarray(ids, dtype=int) for key, ids in self.df.groupby(["source", "label"]).groups.items()}
        if not self.groups: raise ValueError("empty training dataframe")
        self.classes = sorted({key[1] for key in self.groups})
        self.by_class = {c: sorted([key for key in self.groups if key[1] == c]) for c in self.classes}
        self.grad_accum = max(1, int(CFG.get("grad_accum", 1)))
        base_steps = max(1, math.ceil(len(self.df) / self.batch_size))
        self.steps = int(math.ceil(base_steps / self.grad_accum) * self.grad_accum)

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return self.steps

    def _keys(self, cls, rng):
        candidates = list(self.by_class[cls]); rng.shuffle(candidates)
        chosen, sources = [], set()
        for key in candidates:
            if key[0] not in sources:
                chosen.append(key); sources.add(key[0])
            if len(chosen) == self.batch_size: break
        while len(chosen) < self.batch_size:
            pool = [key for key in candidates if key not in chosen] or candidates
            chosen.append(pool[int(rng.integers(0, len(pool)))])
        return chosen

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        produced = window = 0
        while produced < self.steps:
            order = list(self.classes)
            shift = (self.epoch + window) % len(order)
            order = order[shift:] + order[:shift]
            for slot in range(self.grad_accum):
                keys = self._keys(order[slot % len(order)], rng)
                batch = [int(rng.choice(self.groups[key])) for key in keys]
                rng.shuffle(batch); yield batch
                produced += 1
                if produced >= self.steps: return
            window += 1
'''
    code = block(code, 'class BalancedSourceClassBatchSampler(Sampler):', '\n\ndef make_loader', sampler, "sampler")

    helper = '''def prototype_pair_count(labels, domains):
    pairs = 0
    for cls in labels.long().unique():
        n = int(domains[labels.long() == cls].unique().numel())
        pairs += n * (n - 1) // 2
    return pairs


'''
    code = once(code, 'def expected_calibration_error(y, p, bins=15):\n', helper + 'def expected_calibration_error(y, p, bins=15):\n', "prototype helper")

    seg_loss = '''def segmentation_loss(seg, disc, cup, disc_valid, cup_valid):
    disc_logit, cup_logit = seg[:, 0:1], seg[:, 1:2]
    disc_prob = torch.sigmoid(disc_logit)
    cup_raw = torch.sigmoid(cup_logit)
    cup_prob = cup_raw * disc_prob
    ld = masked_bce(disc_logit, disc, disc_valid) + masked_dice_loss(disc_logit, disc, disc_valid)
    cup_bce = masked_bce(cup_logit, cup, cup_valid)
    inter = (cup_prob * cup).sum((2, 3)); den = (cup_prob + cup).sum((2, 3))
    cup_dice = 1 - (2 * inter + 1) / (den + 1)
    cup_dice = (cup_dice.squeeze(1) * cup_valid).sum() / cup_valid.sum().clamp_min(1)
    bd = F.l1_loss(boundary_map(disc_prob), boundary_map(disc), reduction="none").mean((1, 2, 3))
    bc = F.l1_loss(boundary_map(cup_prob), boundary_map(cup), reduction="none").mean((1, 2, 3))
    edge = (bd * disc_valid).sum() / disc_valid.sum().clamp_min(1)
    edge += (bc * cup_valid).sum() / cup_valid.sum().clamp_min(1)
    outside = (cup_raw * (1 - disc_prob)).mean((1, 2, 3))
    containment = (outside * cup_valid).sum() / cup_valid.sum().clamp_min(1)
    return ld + 1.5 * (cup_bce + cup_dice) + 0.10 * edge + 0.20 * containment
'''
    code = block(code, 'def segmentation_loss(seg, disc, cup, disc_valid, cup_valid):', '\n\ndef prototype_loss', seg_loss, "segmentation loss")

    seg_stats = '''def segmentation_batch_stats(seg, disc, cup, disc_valid, cup_valid):
    raw = torch.sigmoid(seg)
    disc_prob = raw[:, 0:1]
    cup_prob = raw[:, 1:2] * disc_prob
    stats = {}
    for name, pred, target, valid in [("disc", disc_prob, disc, disc_valid), ("cup", cup_prob, cup, cup_valid)]:
        binary = (pred > 0.5).float()
        inter = (binary * target).sum((1, 2, 3)); den = (binary + target).sum((1, 2, 3))
        union = (binary + target - binary * target).sum((1, 2, 3))
        dice = (2 * inter + 1) / (den + 1); iou = (inter + 1) / (union + 1)
        stats[name] = {"dice_sum": float((dice * valid).sum().detach().cpu()), "iou_sum": float((iou * valid).sum().detach().cpu()), "count": float(valid.sum().detach().cpu())}
    return stats
'''
    code = block(code, 'def segmentation_batch_stats(seg, disc, cup, disc_valid, cup_valid):', '\n\ndef merge_seg_stats', seg_stats, "segmentation metrics")

    code = once(code, 'self.struct_proj = nn.Sequential(nn.Linear(self.num_sectors + 5, feat_dim), nn.GELU())', 'self.struct_spectrum_dim = self.num_sectors // 2 + 1\n        self.struct_proj = nn.Sequential(nn.Linear(self.struct_spectrum_dim + 5, feat_dim), nn.GELU())', "struct projection")
    old_vcdr = '''        vertical_disc = disc.sum(3).squeeze(1); vertical_cup = cup.sum(3).squeeze(1)
        yd = torch.linspace(-1, 1, h, device=p1.device)[None]
        md = vertical_disc.sum(1).clamp_min(1e-4); mc = vertical_cup.sum(1).clamp_min(1e-4)
        mean_d = (vertical_disc * yd).sum(1) / md; mean_c = (vertical_cup * yd).sum(1) / mc
        sd = torch.sqrt((vertical_disc * (yd - mean_d[:, None]).pow(2)).sum(1) / md + 1e-6)
        sc = torch.sqrt((vertical_cup * (yd - mean_c[:, None]).pow(2)).sum(1) / mc + 1e-6)
        vcdr = (sc / sd.clamp_min(1e-5)).clamp(0, 1)
'''
    new_vcdr = '''        vertical_disc_extent = disc.amax(dim=3).squeeze(1).sum(1)
        vertical_cup_extent = cup.amax(dim=3).squeeze(1).sum(1)
        vcdr = (vertical_cup_extent / vertical_disc_extent.clamp_min(1e-5)).clamp(0, 1)
'''
    code = once(code, old_vcdr, new_vcdr, "vCDR")
    code = once(code, 'struct = torch.cat([global_geom, torch.stack(rim_ratios, dim=1)], dim=1)', 'rim_vector = torch.stack(rim_ratios, dim=1)\n        rim_spectrum = torch.fft.rfft(rim_vector, dim=1).abs()\n        struct = torch.cat([global_geom, rim_spectrum], dim=1)', "rim spectrum")
    code = once(code, '"fused": fused, "gates": gates, "quality": q.squeeze(1),', '"fused": fused, "gates": gates, "gate_context": q.squeeze(1),', "gate output")
    code = once(code, '"vcdr_mask": float(out["vcdr_mask"][i].cpu()), "quality": float(out["quality"][i].cpu()),', '"vcdr_mask": float(out["vcdr_mask"][i].cpu()), "gate_context": float(out["gate_context"][i].cpu()),', "gate prediction")

    a, b, epoch_fn = extract(code, 'def run_full_epoch(', '\n\n@torch.no_grad()\ndef predict_baseline', "full epoch")
    epoch_fn = epoch_fn.replace('all_y, all_p, losses = [], [], []\n', 'all_y, all_p, losses = [], [], []\n    prototype_active_batches = prototype_pairs = full_stage_batches = 0\n', 1)
    old = '''            if stage == "full":
                l_domain = F.cross_entropy(out["domain"], batch["domain"])
                l_proto = prototype_loss(out["fused"], batch["label"], batch["domain"])
                loss = loss + CFG["lambda_domain"] * l_domain + CFG["lambda_proto"] * l_proto
'''
    new = '''            if stage == "full":
                l_domain = F.cross_entropy(out["domain"], batch["domain"])
                pair_count = prototype_pair_count(batch["label"], batch["domain"])
                l_proto = prototype_loss(out["fused"], batch["label"], batch["domain"])
                full_stage_batches += 1; prototype_pairs += int(pair_count); prototype_active_batches += int(pair_count > 0)
                loss = loss + CFG["lambda_domain"] * l_domain + CFG["lambda_proto"] * l_proto
'''
    if old not in epoch_fn: raise RuntimeError("V4.3 missing domain/prototype block")
    epoch_fn = epoch_fn.replace(old, new, 1)
    old = 'metrics.update(final_seg_stats(seg_total)); metrics["loss"] = float(np.mean(losses)); metrics["stage"] = stage\n'
    new = 'metrics.update(final_seg_stats(seg_total)); metrics["loss"] = float(np.mean(losses)); metrics["stage"] = stage\n    metrics["full_stage_batches"] = int(full_stage_batches)\n    metrics["prototype_pairs"] = int(prototype_pairs)\n    metrics["prototype_active_batch_ratio"] = float(prototype_active_batches / full_stage_batches) if full_stage_batches else np.nan\n'
    if old not in epoch_fn: raise RuntimeError("V4.3 missing full metric block")
    epoch_fn = epoch_fn.replace(old, new, 1)
    code = code[:a] + epoch_fn + code[b:]

    a, b, train_fn = extract(code, 'def train_full(', '\n\n# ---------------------------- 6.', "full training")
    if 'improved = score > best_score' not in train_fn: raise RuntimeError("V4.3 missing checkpoint selection")
    train_fn = train_fn.replace('improved = score > best_score', 'eligible_for_selection = va["stage"] == "full"\n        improved = eligible_for_selection and score > best_score', 1)
    old_stop = 'if epoch - best_epoch >= CFG["patience"] and epoch > CFG["anatomy_warmup_epochs"]: break'
    new_stop = 'if va["stage"] == "full" and epoch - best_epoch >= CFG["patience"]: break'
    if old_stop not in train_fn: raise RuntimeError("V4.3 missing full early stop")
    train_fn = train_fn.replace(old_stop, new_stop, 1)
    code = code[:a] + train_fn + code[b:]

    code = once(code, 'seg = torch.sigmoid(out["seg"])[0].detach().cpu().numpy()\n        overlay = np.zeros_like(image); overlay[..., 1] = seg[0]; overlay[..., 0] = seg[1]', 'seg = torch.sigmoid(out["seg"])[0].detach().cpu().numpy()\n        seg[1] = seg[1] * seg[0]\n        overlay = np.zeros_like(image); overlay[..., 1] = seg[0]; overlay[..., 0] = seg[1]', "XAI containment")
    code = once(code, '"- Do not claim clock-hour anatomy for images with unknown laterality.",', '"- Polar-sector XAI is image-oriented when laterality is unknown; do not label it as clinical clock-hour anatomy.",\n             "- gate_context is a latent fusion variable, not a supervised image-quality score.",', "report guardrails")
    return code
