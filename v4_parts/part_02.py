# ---------------------------- 2. IMAGE, MASK, DATASET, SAMPLER ----------------------------
def read_rgb(path):
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_binary_mask(path, shape):
    if path is None or pd.isna(path) or not Path(str(path)).exists():
        return np.zeros(shape, np.float32), 0.0
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None: return np.zeros(shape, np.float32), 0.0
    if m.shape != shape: m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    values, counts = np.unique(m, return_counts=True)
    bg = int(values[np.argmax(counts)])
    out = (np.abs(m.astype(np.int16) - bg) > 3).astype(np.float32)
    return out, float(out.sum() > 20)


def decode_combined_mask(path, shape):
    if path is None or pd.isna(path) or not Path(str(path)).exists():
        z = np.zeros(shape, np.float32)
        return z, z.copy(), 0.0, 0.0
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        z = np.zeros(shape, np.float32)
        return z, z.copy(), 0.0, 0.0
    if m.shape != shape: m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    bg = int(pd.Series(border).mode().iloc[0])
    values = [int(v) for v in np.unique(m) if abs(int(v) - bg) > 3]
    if len(values) >= 2:
        areas = [(v, int((m == v).sum())) for v in values]
        cup_value = min(areas, key=lambda x: x[1])[0]
        disc = np.isin(m, values).astype(np.float32)
        cup = (m == cup_value).astype(np.float32)
        return disc, cup * disc, float(disc.sum() > 20), float(cup.sum() > 10)
    if len(values) == 1:
        disc = (m == values[0]).astype(np.float32)
        return disc, np.zeros_like(disc), float(disc.sum() > 20), 0.0
    z = np.zeros(shape, np.float32)
    return z, z.copy(), 0.0, 0.0


def load_masks(row, shape):
    if row.combined_mask_path is not None and not pd.isna(row.combined_mask_path):
        return decode_combined_mask(row.combined_mask_path, shape)
    disc, dv = read_binary_mask(row.disc_mask_path, shape)
    cup, cv = read_binary_mask(row.cup_mask_path, shape)
    cup = cup * disc if dv else cup
    return disc, cup, dv, cv


def fov_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > max(5, int(np.percentile(gray, 10)))
    ys, xs = np.where(mask)
    if len(xs) < 100: return 0, 0, image.shape[1], image.shape[0]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x = int(0.01 * (x1 - x0)); pad_y = int(0.01 * (y1 - y0))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(image.shape[1], x1 + pad_x), min(image.shape[0], y1 + pad_y)


def crop_and_pad_square(image, disc, cup):
    x0, y0, x1, y1 = fov_bbox(image)
    image, disc, cup = image[y0:y1, x0:x1], disc[y0:y1, x0:x1], cup[y0:y1, x0:x1]
    h, w = image.shape[:2]
    side = max(h, w)
    top = (side - h) // 2; bottom = side - h - top
    left = (side - w) // 2; right = side - w - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    disc = cv2.copyMakeBorder(disc, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    cup = cv2.copyMakeBorder(cup, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return image, disc, cup


def random_augment(image, disc, cup, rng):
    h, w = image.shape[:2]
    if rng.random() < 0.60:
        angle = float(rng.uniform(-5.0, 5.0))
        scale = float(rng.uniform(0.97, 1.03))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        disc = cv2.warpAffine(disc, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        cup = cv2.warpAffine(cup, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    img = image.astype(np.float32) / 255.0
    if rng.random() < 0.60:
        contrast = float(rng.uniform(0.88, 1.12)); brightness = float(rng.uniform(-0.08, 0.08))
        img = np.clip(img * contrast + brightness, 0, 1)
    if rng.random() < 0.20:
        sigma = float(rng.uniform(0.0, 0.02))
        img = np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)
    return (img * 255).astype(np.uint8), disc, cup


def vcdr_from_mask(disc, cup):
    yd = np.where(disc > 0.5)[0]; yc = np.where(cup > 0.5)[0]
    if len(yd) == 0 or len(yc) == 0: return 0.0
    return float((yc.max() - yc.min() + 1) / max(1, yd.max() - yd.min() + 1))


class GlaucomaDataset(Dataset):
    def __init__(self, df, train=False, seed=2029, domain_map=None):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.seed = int(seed)
        self.epoch = 0
        self.domain_map = domain_map or {s: i for i, s in enumerate(sorted(df.source.unique()))}

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = read_rgb(row.image_path)
        disc, cup, disc_valid, cup_valid = load_masks(row, image.shape[:2])
        if CFG["canonicalize_laterality"] and row.laterality == "R":
            image = cv2.flip(image, 1); disc = cv2.flip(disc, 1); cup = cv2.flip(cup, 1)
        image, disc, cup = crop_and_pad_square(image, disc, cup)
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + idx)
        if self.train: image, disc, cup = random_augment(image, disc, cup, rng)
        size = int(CFG["image_size"])
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        disc = cv2.resize(disc, (size, size), interpolation=cv2.INTER_NEAREST)
        cup = cv2.resize(cup, (size, size), interpolation=cv2.INTER_NEAREST) * disc
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        disc_t = torch.from_numpy(disc[None]).float(); cup_t = torch.from_numpy(cup[None]).float()
        vcdr_valid = float(disc_valid > 0 and cup_valid > 0)
        return {
            "image": image_t, "disc": disc_t, "cup": cup_t,
            "disc_valid": torch.tensor(disc_valid, dtype=torch.float32),
            "cup_valid": torch.tensor(cup_valid, dtype=torch.float32),
            "vcdr_valid": torch.tensor(vcdr_valid, dtype=torch.float32),
            "vcdr": torch.tensor(vcdr_from_mask(disc, cup), dtype=torch.float32),
            "label": torch.tensor(float(row.label), dtype=torch.float32),
            "domain": torch.tensor(self.domain_map[row.source], dtype=torch.long),
            "source_name": row.source, "path": row.image_path,
            "laterality": row.laterality,
        }


class BalancedSourceClassBatchSampler(Sampler):
    """Every training batch contains all available source/class groups."""
    def __init__(self, df, batch_size, seed=2029):
        self.df = df.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups = {
            key: np.asarray(indices, dtype=int)
            for key, indices in self.df.groupby(["source", "label"]).groups.items()
        }
        self.keys = sorted(self.groups)
        if self.batch_size < len(self.keys):
            raise ValueError(f"batch_size={self.batch_size} is smaller than {len(self.keys)} source/class groups")
        self.steps = max(1, math.ceil(len(self.df) / self.batch_size))

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            batch = []
            while len(batch) < self.batch_size:
                for key in self.keys:
                    if len(batch) >= self.batch_size: break
                    batch.append(int(rng.choice(self.groups[key])))
            rng.shuffle(batch)
            yield batch


def make_loader(df, train, seed, domain_map=None):
    ds = GlaucomaDataset(df, train=train, seed=seed, domain_map=domain_map)
    if train:
        sampler = BalancedSourceClassBatchSampler(df, CFG["batch_size"], seed)
        return DataLoader(ds, batch_sampler=sampler, num_workers=CFG["num_workers"], pin_memory=DEVICE.type == "cuda"), sampler
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=CFG["num_workers"], pin_memory=DEVICE.type == "cuda"), None


def source_stratified_split(df, val_fraction, seed):
    trains, vals = [], []
    for src, group in df.groupby("source"):
        if group.label.nunique() < 2:
            raise RuntimeError(f"Training source {src} has only one class")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        ti, vi = next(splitter.split(group, group.label))
        trains.append(group.iloc[ti]); vals.append(group.iloc[vi])
    return pd.concat(trains).reset_index(drop=True), pd.concat(vals).reset_index(drop=True)


def remove_fold_leakage(train_pool, test_df):
    if not CFG["exclude_cross_source_duplicates"]: return train_pool.copy(), pd.DataFrame()
    test_hashes = set(test_df.fingerprint.dropna())
    leaked = train_pool[train_pool.fingerprint.isin(test_hashes)].copy()
    clean = train_pool[~train_pool.index.isin(leaked.index)].copy()
    return clean.reset_index(drop=True), leaked.reset_index(drop=True)
