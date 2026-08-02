# ---------------------------- 3. MODELS ----------------------------
def create_backbone():
    tried = []
    for name in [CFG["backbone"], CFG["backbone_fallback"]]:
        try:
            model = timm.create_model(name, pretrained=CFG["pretrained"], features_only=True)
            return model, name
        except Exception as exc:
            tried.append(f"{name}: {exc}")
    raise RuntimeError("Could not create backbone. " + " | ".join(tried))


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        groups = 16 if out_ch % 16 == 0 else 8
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)


class FPNDecoder(nn.Module):
    def __init__(self, channels, out_dim=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_dim, 1) for c in channels])
        self.smooth = nn.ModuleList([ConvGNAct(out_dim, out_dim) for _ in channels])
        self.seg_head = nn.Sequential(
            ConvGNAct(out_dim, out_dim),
            nn.Conv2d(out_dim, 2, 1),
        )

    def forward(self, feats, output_size):
        laterals = [layer(x) for layer, x in zip(self.lateral, feats)]
        pyramids = [None] * len(laterals)
        pyramids[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(pyramids[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False)
            pyramids[i] = laterals[i] + up
        pyramids = [smooth(p) for smooth, p in zip(self.smooth, pyramids)]
        seg_small = self.seg_head(pyramids[0])
        seg_full = F.interpolate(seg_small, size=output_size, mode="bilinear", align_corners=False)
        return pyramids, seg_small, seg_full


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0): return GradReverse.apply(x, lambd)


class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden, heads=4, dropout=0.2):
        super().__init__()
        if hidden % heads != 0: raise ValueError("graph_hidden must be divisible by graph_heads")
        self.heads = heads; self.dk = hidden // heads
        self.q = nn.Linear(in_dim, hidden); self.k = nn.Linear(in_dim, hidden); self.v = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        k = self.k(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        v = self.v(x).view(b, n, self.heads, self.dk).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        score = score.masked_fill(~adjacency[None, None].to(x.device), -1e4)
        attn = self.drop(torch.softmax(score, dim=-1))
        z = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.norm(self.out(z) + self.q(x)), attn


class GlobalBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.backbone_name = create_backbone()
        c = self.backbone.feature_info.channels()[-1]
        self.head = nn.Sequential(
            nn.Linear(c, CFG["feature_dim"]), nn.GELU(), nn.Dropout(CFG["dropout"]),
            nn.LayerNorm(CFG["feature_dim"]), nn.Linear(CFG["feature_dim"], 1),
        )

    def forward(self, x):
        f = self.backbone(x)[-1]
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return {"logit": self.head(pooled).squeeze(1), "global_raw": pooled}


class RimGraphV4(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_sectors = int(CFG["num_sectors"])
        self.backbone, self.backbone_name = create_backbone()
        channels = self.backbone.feature_info.channels()
        c4 = channels[-1]; fpn_dim = int(CFG["fpn_dim"]); feat_dim = int(CFG["feature_dim"])
        self.fpn = FPNDecoder(channels, fpn_dim)
        self.global_proj = nn.Sequential(nn.Linear(c4, feat_dim), nn.GELU(), nn.Dropout(CFG["dropout"]))
        self.local_proj = nn.Sequential(nn.Linear(fpn_dim, feat_dim), nn.GELU(), nn.Dropout(CFG["dropout"]))
        node_dim = fpn_dim + 5
        self.gat1 = SimpleGAT(node_dim, CFG["graph_hidden"], CFG["graph_heads"], CFG["dropout"])
        self.gat2 = SimpleGAT(CFG["graph_hidden"], CFG["graph_hidden"], CFG["graph_heads"], CFG["dropout"])
        self.graph_proj = nn.Sequential(nn.Linear(CFG["graph_hidden"], feat_dim), nn.GELU())
        self.struct_proj = nn.Sequential(nn.Linear(self.num_sectors + 5, feat_dim), nn.GELU())
        self.quality_head = nn.Sequential(nn.Linear(c4, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(feat_dim * 4 + 1, 128), nn.GELU(), nn.Dropout(CFG["dropout"]), nn.Linear(128, 4))
        self.classifier = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(CFG["dropout"]), nn.Linear(feat_dim, 1))
        self.vcdr_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.GELU(), nn.Linear(64, num_domains))
        self.register_buffer("adjacency", self._make_adjacency(self.num_sectors + 1), persistent=False)

    def _make_adjacency(self, n):
        k = n - 1
        a = torch.eye(n, dtype=torch.bool)
        for i in range(k):
            a[i, (i - 1) % k] = True
            a[i, (i + 1) % k] = True
            a[i, (i + k // 2) % k] = True
            a[i, k] = True; a[k, i] = True
        return a

    def _soft_center_and_scale(self, disc):
        b, _, h, w = disc.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=disc.device),
            torch.linspace(-1, 1, w, device=disc.device), indexing="ij"
        )
        xx = xx[None, None]; yy = yy[None, None]
        mass = disc.sum((2, 3), keepdim=True).clamp_min(1e-4)
        cx = (disc * xx).sum((2, 3), keepdim=True) / mass
        cy = (disc * yy).sum((2, 3), keepdim=True) / mass
        vx = (disc * (xx - cx).pow(2)).sum((2, 3), keepdim=True) / mass
        vy = (disc * (yy - cy).pow(2)).sum((2, 3), keepdim=True) / mass
        sx = (4.0 * torch.sqrt(vx.clamp_min(1e-5))).clamp(0.20, 0.80)
        sy = (4.0 * torch.sqrt(vy.clamp_min(1e-5))).clamp(0.20, 0.80)
        return cx, cy, sx, sy, xx, yy

    def _roi_pool(self, p1, disc):
        cx, cy, sx, sy, _, _ = self._soft_center_and_scale(disc)
        b = p1.shape[0]
        theta = torch.zeros(b, 2, 3, device=p1.device, dtype=p1.dtype)
        theta[:, 0, 0] = sx.flatten(); theta[:, 1, 1] = sy.flatten()
        theta[:, 0, 2] = cx.flatten(); theta[:, 1, 2] = cy.flatten()
        out_size = int(CFG["roi_output_size"])
        grid = F.affine_grid(theta, size=(b, p1.shape[1], out_size, out_size), align_corners=False)
        roi = F.grid_sample(p1, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return F.adaptive_avg_pool2d(roi, 1).flatten(1), roi

    def _polar_graph(self, p1, disc, cup):
        b, c, h, w = p1.shape
        cx, cy, _, _, xx, yy = self._soft_center_and_scale(disc)
        angle = (torch.atan2(yy - cy, xx - cx) + 2 * math.pi) % (2 * math.pi)
        sector_id = torch.floor(angle / (2 * math.pi / self.num_sectors)).long().clamp(0, self.num_sectors - 1)
        rim = (disc - cup).clamp(0, 1)
        nodes, rim_ratios = [], []
        for i in range(self.num_sectors):
            sm = (sector_id == i).float()
            wgt = disc * sm
            denom = wgt.sum((2, 3)).clamp_min(1e-4)
            visual = (p1 * wgt).sum((2, 3)) / denom
            d_area = (disc * sm).mean((2, 3))
            c_area = (cup * sm).mean((2, 3))
            r_area = (rim * sm).mean((2, 3))
            local_cdr = (c_area / d_area.clamp_min(1e-5)).clamp(0, 1)
            rim_ratio = (r_area / d_area.clamp_min(1e-5)).clamp(0, 1)
            confidence = (disc * sm).amax((2, 3))
            geom = torch.stack([
                d_area.squeeze(1), c_area.squeeze(1), local_cdr.squeeze(1),
                rim_ratio.squeeze(1), confidence.squeeze(1)
            ], dim=1)
            nodes.append(torch.cat([visual, geom], dim=1))
            rim_ratios.append(rim_ratio.squeeze(1))
        nodes = torch.stack(nodes, dim=1)
        gp = F.adaptive_avg_pool2d(p1, 1).flatten(1)
        d_area = disc.mean((2, 3)); c_area = cup.mean((2, 3)); cdar = (c_area / d_area.clamp_min(1e-5)).clamp(0, 1)
        vertical_disc = disc.sum(3).squeeze(1); vertical_cup = cup.sum(3).squeeze(1)
        yd = torch.linspace(-1, 1, h, device=p1.device)[None]
        md = vertical_disc.sum(1).clamp_min(1e-4); mc = vertical_cup.sum(1).clamp_min(1e-4)
        mean_d = (vertical_disc * yd).sum(1) / md; mean_c = (vertical_cup * yd).sum(1) / mc
        sd = torch.sqrt((vertical_disc * (yd - mean_d[:, None]).pow(2)).sum(1) / md + 1e-6)
        sc = torch.sqrt((vertical_cup * (yd - mean_c[:, None]).pow(2)).sum(1) / mc + 1e-6)
        vcdr = (sc / sd.clamp_min(1e-5)).clamp(0, 1)
        uncertainty = (disc * (1 - disc)).mean((2, 3)) + (cup * (1 - cup)).mean((2, 3))
        global_geom = torch.cat([d_area, c_area, cdar, vcdr[:, None], uncertainty], dim=1)
        global_visual = torch.cat([gp, global_geom], dim=1).unsqueeze(1)
        nodes = torch.cat([nodes, global_visual], dim=1)
        struct = torch.cat([global_geom, torch.stack(rim_ratios, dim=1)], dim=1)
        return nodes, struct, vcdr

    def forward(self, x, stage="full", grl_lambda=0.0, return_features=False):
        feats = self.backbone(x)
        pyramids, seg_small, seg_full = self.fpn(feats, x.shape[-2:])
        p1 = pyramids[0]
        if return_features and p1.requires_grad: p1.retain_grad()
        seg_prob_small = torch.sigmoid(seg_small)
        disc = seg_prob_small[:, 0:1]
        cup = seg_prob_small[:, 1:2] * disc
        global_raw = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        local_raw, roi_feature = self._roi_pool(p1, disc)
        nodes, struct, vcdr_mask = self._polar_graph(p1, disc, cup)
        z, _ = self.gat1(nodes, self.adjacency)
        z, attn = self.gat2(z, self.adjacency)
        graph_raw = z.mean(1)
        fg = self.global_proj(global_raw)
        fl = self.local_proj(local_raw)
        fr = self.graph_proj(graph_raw)
        fs = self.struct_proj(struct)
        q = torch.sigmoid(self.quality_head(global_raw))
        if stage == "global_seg":
            gates = torch.zeros(x.shape[0], 4, device=x.device, dtype=x.dtype); gates[:, 0] = 1.0
        else:
            gate_logits = self.gate(torch.cat([fg, fl, fr, fs, q], dim=1))
            if stage == "anatomy": gate_logits[:, 2] = -1e4
            gates = torch.softmax(gate_logits, dim=1)
        stack = torch.stack([fg, fl, fr, fs], dim=1)
        fused = (stack * gates.unsqueeze(-1)).sum(1)
        out = {
            "logit": self.classifier(fused).squeeze(1),
            "seg": seg_full,
            "seg_small": seg_small,
            "vcdr_reg": self.vcdr_head(fused).squeeze(1),
            "vcdr_mask": vcdr_mask,
            "domain": self.domain_head(grad_reverse(fused, grl_lambda)),
            "fused": fused, "gates": gates, "quality": q.squeeze(1),
            "graph_attention": attn.mean(1), "roi_feature": roi_feature,
        }
        if return_features: out["p1"] = p1
        return out
