def read_rgb(path):
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

def decode_combined_mask(path, target_hw):
    h, w = target_hw
    if path is None or not Path(path).exists():
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), 0.0
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), 0.0
    # Border mode is normally background regardless of encoding convention.
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    bg = int(pd.Series(border).mode().iloc[0])
    values, counts = np.unique(m, return_counts=True)
    fg_values = [int(v) for v in values if abs(int(v) - bg) > 3]
    if len(fg_values) >= 2:
        regions = [(v, int((m == v).sum())) for v in fg_values]
        cup_v = min(regions, key=lambda z: z[1])[0]
        disc = np.isin(m, fg_values).astype(np.float32)
        cup = (m == cup_v).astype(np.float32)
    elif len(fg_values) == 1:
        disc = (m == fg_values[0]).astype(np.float32)
        cup = np.zeros_like(disc)
    else:
        disc = np.zeros_like(m, dtype=np.float32); cup = np.zeros_like(disc)
    disc = cv2.resize(disc, (w, h), interpolation=cv2.INTER_NEAREST)
    cup = cv2.resize(cup, (w, h), interpolation=cv2.INTER_NEAREST)
    cup = cup * disc
    valid = float(disc.sum() > 20)
    return disc, cup, valid

def vcdr_from_numpy(disc, cup):
    ys_d = np.where(disc > .5)[0]; ys_c = np.where(cup > .5)[0]
    if len(ys_d) == 0 or len(ys_c) == 0: return 0.0
    return float((ys_c.max() - ys_c.min() + 1) / max(1, ys_d.max() - ys_d.min() + 1))

train_tf = A.Compose([
    A.Resize(CFG["image_size"], CFG["image_size"]),
    A.HorizontalFlip(p=.5),
    A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=.5),
    A.RandomBrightnessContrast(brightness_limit=.15, contrast_limit=.15, p=.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=8, p=.25),
    A.GaussNoise(std_range=(0.01, 0.04), p=.20),
    A.ImageCompression(quality_range=(70, 100), p=.20),
    A.Normalize(mean=(0.485, .456, .406), std=(.229, .224, .225)),
    ToTensorV2(),
], additional_targets={"disc": "mask", "cup": "mask"})

eval_tf = A.Compose([
    A.Resize(CFG["image_size"], CFG["image_size"]),
    A.Normalize(mean=(0.485, .456, .406), std=(.229, .224, .225)),
    ToTensorV2(),
], additional_targets={"disc": "mask", "cup": "mask"})

class GlaucomaDataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True); self.tf = train_tf if train else eval_tf
        self.domain_map = {s: i for i, s in enumerate(sorted(df["source"].unique()))}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image = read_rgb(r.image_path)
        disc, cup, mask_valid = decode_combined_mask(r.mask_path, image.shape[:2])
        out = self.tf(image=image, disc=disc, cup=cup)
        image_t = out["image"].float()
        disc_t = out["disc"].float().unsqueeze(0)
        cup_t = out["cup"].float().unsqueeze(0)
        return {
            "image": image_t, "disc": disc_t, "cup": cup_t,
            "label": torch.tensor(float(r.label), dtype=torch.float32),
            "domain": torch.tensor(self.domain_map[r.source], dtype=torch.long),
            "source_name": r.source, "path": r.image_path,
            "mask_valid": torch.tensor(mask_valid, dtype=torch.float32),
            "vcdr": torch.tensor(vcdr_from_numpy(disc_t.numpy()[0], cup_t.numpy()[0]), dtype=torch.float32)
        }

# ---------------------------- 8. MODEL ----------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0): return GradReverse.apply(x, lambd)

class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden, heads=4, dropout=.2):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads; self.dk = hidden // heads
        self.q = nn.Linear(in_dim, hidden); self.k = nn.Linear(in_dim, hidden); self.v = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, hidden); self.norm = nn.LayerNorm(hidden); self.drop = nn.Dropout(dropout)
    def forward(self, x, adjacency):
        # x: B,N,D, adjacency: N,N bool
        B,N,_ = x.shape
        q = self.q(x).view(B,N,self.heads,self.dk).transpose(1,2)
        k = self.k(x).view(B,N,self.heads,self.dk).transpose(1,2)
        v = self.v(x).view(B,N,self.heads,self.dk).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(~adjacency[None,None].to(x.device), -1e4)
        attn = self.drop(torch.softmax(scores, dim=-1))
        z = torch.matmul(attn, v).transpose(1,2).reshape(B,N,-1)
        return self.norm(self.out(z) + self.q(x)), attn

class RimGraphDG(nn.Module):
    def __init__(self, num_domains=2):
        super().__init__()
        self.num_sectors = CFG["num_sectors"]
        self.backbone = timm.create_model(CFG["backbone"], pretrained=CFG["pretrained"], features_only=True)
        channels = self.backbone.feature_info.channels()
        c = channels[-1]
        self.seg_head = nn.Sequential(
            nn.Conv2d(c, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 2, 1)
        )
        self.global_proj = nn.Sequential(nn.Linear(c, CFG["feature_dim"]), nn.GELU(), nn.Dropout(CFG["dropout"]))
        self.local_proj = nn.Sequential(nn.Linear(c, CFG["feature_dim"]), nn.GELU(), nn.Dropout(CFG["dropout"]))
        node_dim = c + 4
        self.gat1 = SimpleGAT(node_dim, CFG["graph_hidden"], CFG["graph_heads"], CFG["dropout"])
        self.gat2 = SimpleGAT(CFG["graph_hidden"], CFG["graph_hidden"], CFG["graph_heads"], CFG["dropout"])
        self.graph_proj = nn.Sequential(nn.Linear(CFG["graph_hidden"], CFG["feature_dim"]), nn.GELU())
        self.struct_proj = nn.Sequential(nn.Linear(self.num_sectors + 4, CFG["feature_dim"]), nn.GELU())
        self.quality_head = nn.Sequential(nn.Linear(c, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(CFG["feature_dim"]*4 + 1, 128), nn.GELU(), nn.Linear(128, 4))
        self.classifier = nn.Sequential(nn.LayerNorm(CFG["feature_dim"]), nn.Dropout(CFG["dropout"]), nn.Linear(CFG["feature_dim"], 1))
        self.vcdr_head = nn.Sequential(nn.Linear(CFG["feature_dim"], 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        self.domain_head = nn.Sequential(nn.Linear(CFG["feature_dim"], 64), nn.GELU(), nn.Linear(64, num_domains))
        self.register_buffer("adjacency", self._make_adjacency(self.num_sectors + 1), persistent=False)

    def _make_adjacency(self, n):
        k = n - 1; a = torch.eye(n, dtype=torch.bool)
        for i in range(k):
            a[i, (i-1)%k] = a[i, (i+1)%k] = True
            a[i, (i+k//2)%k] = True
            a[i, k] = a[k, i] = True
        return a

    def _soft_geometry(self, seg_prob, feat):
