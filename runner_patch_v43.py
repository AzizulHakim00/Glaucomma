"""V4.3 scientific guards for prototype learning, anatomy metrics, and checkpoint selection."""


def _replace_once(code: str, old: str, new: str, label: str) -> str:
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f"V4.3 patch expected one {label} block, found {count}")
    return code.replace(old, new, 1)


def _replace_block(code: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = code.find(start_marker)
    if start < 0:
        raise RuntimeError(f"V4.3 patch could not find start of {label}")
    end = code.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"V4.3 patch could not find end of {label}")
    return code[:start] + replacement.rstrip() + code[end:]


def _extract_block(code: str, start_marker: str, end_marker: str, label: str):
    start = code.find(start_marker)
    if start < 0:
        raise RuntimeError(f"V4.3 patch could not find start of {label}")
    end = code.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"V4.3 patch could not find end of {label}")
    return start, end, code[start:end]


def apply_v43(code: str) -> str:
    validation = '''if int(CFG["batch_size"]) < 1:
    raise ValueError("batch_size must be at least 1")
if int(CFG["grad_accum"]) < 1:
    raise ValueError("grad_accum must be at least 1")
if CFG["run_full_model"] and int(CFG["full_epochs"]) <= int(CFG["anatomy_warmup_epochs"]):
    raise ValueError(
        "full_epochs must be greater than anatomy_warmup_epochs so the full graph/domain stage is actually trained"
    )

'''
    code = _replace_once(code, 'IN_COLAB = "google.colab" in sys.modules\n', validation + 'IN_COLAB = "google.colab" in sys.modules\n', "configuration validation")

    sampler = '''class BalancedSourceClassBatchSampler(Sampler):
    """Prototype-aware source/class balancing for small micro-batches.

    For the T4 profile (batch_size=2, grad_accum=2), one micro-batch contains
    the same class from two different source domains. The next micro-batch uses
    the other class. Thus every optimizer update sees both classes and all four
    source/class groups, while class-conditional cross-domain prototype loss is
    active instead of silently becoming zero.
    """
    def __init__(self, df, batch_size, seed=2029):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.groups = {
            key: np.asarray(indices, dtype=int)
            for key, indices in self.df.groupby(["source", "label"]).groups.items()
        }
        if not self.groups:
            raise ValueError("BalancedSourceClassBatchSampler received an empty dataframe")
        self.classes = sorted({key[1] for key in self.groups})
        self.class_to_keys = {
            label: sorted([key for key in self.groups if key[1] == label])
            for label in self.classes
        }
        self.grad_accum = max(1, int(CFG.get("grad_accum", 1)))
        base_steps = max(1, math.ceil(len(self.df) / self.batch_size))
        self.steps = int(math.ceil(base_steps / self.grad_accum) * self.grad_accum)

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return self.steps

    def _keys_for_class(self, label, rng):
        candidates = list(self.class_to_keys[label])
        rng.shuffle(candidates)
        selected, used_sources = [], set()
        for key in candidates:
            if key[0] not in used_sources:
                selected.append(key); used_sources.add(key[0])
            if len(selected) == self.batch_size:
                break
        while len(selected) < self.batch_size:
            pool = [key for key in candidates if key not in selected] or candidates
            selected.append(pool[int(rng.integers(0, len(pool)))])
        return selected

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        produced, window = 0, 0
        while produced < self.steps:
            class_order = list(self.classes)
            if class_order:
                shift = (self.epoch + window) % len(class_order)
                class_order = class_order[shift:] + class_order[:shift]
            for slot in range(self.grad_accum):
                label = class_order[slot % len(class_order)]
                keys = self._keys_for_class(label, rng)
                batch = [int(rng.choice(self.groups[key])) for key in keys]
                rng.shuffle(batch)
                yield batch
                produced += 1
                if produced >= self.steps:
                    return
            window += 1
'''
    code = _replace_block(
        code,
        'class BalancedSourceClassBatchSampler(Sampler):',
        '\n\ndef make_loader',
        sampler,
        "balanced sampler",
    )

    proto_helper = '''def prototype_pair_count(labels, domains):
    count = 0
    for cls in labels.long().unique():
        n_domains = int(domains[labels.long() == cls].unique().numel())
        count += n_domains * (n_domains - 1) // 2
    return count


'''
    code = _replace_once(code, 'def expected_calibration_error(y, p, bins=15):\n', proto_helper + 'def expected_calibration_error(y, p, bins=15):\n', "prototype diagnostic")

    segmentation = '''def segmentation_loss(seg, disc, cup, disc_valid, cup_valid):
    disc_logit, cup_logit = seg[:, 0:1], seg[:, 1:2]
    disc_prob = torch.sigmoid(disc_logit)
    cup_raw = torch.sigmoid(cup_logit)
    cup_prob = cup_raw * disc_prob

    ld = masked_bce(disc_logit, disc, disc_valid) + masked_dice_loss(disc_logit, disc, disc_valid)
    cup_bce = masked_bce(cup_logit, cup, cup_valid)
    cup_inter = (cup_prob * cup).sum((2, 3))
    cup_den = (cup_prob + cup).sum((2, 3))
    cup_dice = 1 - (2 * cup_inter + 1) / (cup_den + 1)
    cup_dice = (cup_dice.squeeze(1) * cup_valid).sum() / cup_valid.sum().clamp_min(1)

    bd = F.l1_loss(boundary_map(disc_prob), boundary_map(disc), reduction="none").mean((1, 2, 3))
    bc = F.l1_loss(boundary_map(cup_prob), boundary_map(cup), reduction="none").mean((1, 2, 3))
    edge = (bd * disc_valid).sum() / disc_valid.sum().clamp_min(1)
    edge = edge + (bc * cup_valid).sum() / cup_valid.sum().clamp_min(1)

    containment_raw = (cup_raw * (1 - disc_prob)).mean((1, 2, 3))
    containment = (containment_raw * cup_valid).sum() / cup_valid.sum().clamp_min(1)
    return ld + 1.5 * (cup_bce + cup_dice) + 0.10 * edge + 0.20 * containment
'''
    code = _replace_block(code, 'def segmentation_loss(seg, disc, cup, disc_valid, cup_valid):', '\n\ndef prototype_loss', segmentation, "segmentation loss")

    seg_stats = '''def segmentation_batch_stats(seg, disc, cup, disc_valid, cup_valid):
    raw = torch.sigmoid(seg)
    disc_prob = raw[:, 0:1]
    cup_prob = raw[:, 1:2] * disc_prob
    stats = {}
    for name, pred, target, valid in [
        ("disc", disc_prob, disc, disc_valid),
        ("cup", cup_prob, cup, cup_valid),
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
'''
    code = _replace_block(code, 'def segmentation_batch_stats(seg, disc, cup, disc_valid, cup_valid):', '\n\ndef merge_seg_stats', seg_stats, "segmentation metrics")

    code = _replace_once(
        code,
        'self.struct_proj = nn.Sequential(nn.Linear(self.num_sectors + 5, feat_dim), nn.GELU())',
        'self.struct_spectrum_dim = self.num_sectors // 2 + 1\n        self.struct_proj = nn.Sequential(nn.Linear(self.struct_spectrum_dim + 5, feat_dim), nn.GELU())',
        "reflection-invariant structural projection",
    )

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
    code = _replace_once(code, old_vcdr, new_vcdr, "mask-derived vCDR")
    code = _replace_once(
        code,
        'struct = torch.cat([global_geom, torch.stack(rim_ratios, dim=1)], dim=1)',
        'rim_vector = torch.stack(rim_ratios, dim=1)\n        rim_spectrum = torch.fft.rfft(rim_vector, dim=1).abs()\n        struct = torch.cat([global_geom, rim_spectrum], dim=1)',
        "reflection-invariant rim spectrum",
    )
    code = _replace_once(
        code,
        '"fused": fused, "gates": gates, "quality": q.squeeze(1),',
        '"fused": fused, "gates": gates, "gate_context": q.squeeze(1),',
        "latent gate context output",
    )
    code = _replace_once(
        code,
        '"vcdr_mask": float(out["vcdr_mask"][i].cpu()), "quality": float(out["quality"][i].cpu()),',
        '"vcdr_mask": float(out["vcdr_mask"][i].cpu()), "gate_context": float(out["gate_context"][i].cpu()),',
        "prediction gate context",
    )

    start, end, full_epoch = _extract_block(code, 'def run_full_epoch(', '\n\n@torch.no_grad()\ndef predict_baseline', "full epoch")
    full_epoch = full_epoch.replace(
        'all_y, all_p, losses = [], [], []\n',
        'all_y, all_p, losses = [], [], []\n    prototype_active_batches = 0\n    prototype_pairs = 0\n    full_stage_batches = 0\n',
        1,
    )
    old_domain = '''            if stage == "full":
                l_domain = F.cross_entropy(out["domain"], batch["domain"])
                l_proto = prototype_loss(out["fused"], batch["label"], batch["domain"])
                loss = loss + CFG["lambda_domain"] * l_domain + CFG["lambda_proto"] * l_proto
'''
    new_domain = '''            if stage == "full":
                l_domain = F.cross_entropy(out["domain"], batch["domain"])
                pair_count = prototype_pair_count(batch["label"], batch["domain"])
                l_proto = prototype_loss(out["fused"], batch["label"], batch["domain"])
                full_stage_batches += 1
                prototype_pairs += int(pair_count)
                prototype_active_batches += int(pair_count > 0)
                loss = loss + CFG["lambda_domain"] * l_domain + CFG["lambda_proto"] * l_proto
'''
    if old_domain not in full_epoch:
        raise RuntimeError("V4.3 could not find domain/prototype block")
    full_epoch = full_epoch.replace(old_domain, new_domain, 1)
    old_metrics = 'metrics.update(final_seg_stats(seg_total)); metrics["loss"] = float(np.mean(losses)); metrics["stage"] = stage\n'
    new_metrics = '''metrics.update(final_seg_stats(seg_total)); metrics["loss"] = float(np.mean(losses)); metrics["stage"] = stage
    metrics["full_stage_batches"] = int(full_stage_batches)
    metrics["prototype_pairs"] = int(prototype_pairs)
    metrics["prototype_active_batch_ratio"] = (
        float(prototype_active_batches / full_stage_batches) if full_stage_batches > 0 else np.nan
    )
'''
    if old_metrics not in full_epoch:
        raise RuntimeError("V4.3 could not find full-epoch metric block")
    full_epoch = full_epoch.replace(old_metrics, new_metrics, 1)
    code = code[:start] + full_epoch + code[end:]

    start, end, train_full = _extract_block(code, 'def train_full(', '\n\n# ---------------------------- 6.', "full training")
    if 'improved = score > best_score' not in train_full:
        raise RuntimeError("V4.3 could not find full-model checkpoint selection")
    train_full = train_full.replace(
        'improved = score > best_score',
        'eligible_for_selection = va["stage"] == "full"\n        improved = eligible_for_selection and score > best_score',
        1,
    )
    if 'if epoch - best_epoch >= CFG["patience"]: break' not in train_full:
        raise RuntimeError("V4.3 could not find full-model early stopping")
    train_full = train_full.replace(
        'if epoch - best_epoch >= CFG["patience"]: break',
        'if va["stage"] == "full" and epoch - best_epoch >= CFG["patience"]: break',
        1,
    )
    code = code[:start] + train_full + code[end:]

    code = _replace_once(
        code,
        'seg = torch.sigmoid(out["seg"])[0].detach().cpu().numpy()\n        overlay = np.zeros_like(image); overlay[..., 1] = seg[0]; overlay[..., 0] = seg[1]',
        'seg = torch.sigmoid(out["seg"])[0].detach().cpu().numpy()\n        seg[1] = seg[1] * seg[0]\n        overlay = np.zeros_like(image); overlay[..., 1] = seg[0]; overlay[..., 0] = seg[1]',
        "XAI cup containment",
    )
    code = _replace_once(
        code,
        '"- Do not claim clock-hour anatomy for images with unknown laterality.",',
        '"- Polar-sector XAI is image-oriented when laterality is unknown; do not label it as clinical clock-hour anatomy.",\n             "- The saved gate_context value is a latent fusion variable, not a supervised image-quality score.",',
        "scientific report guardrails",
    )
    return code
