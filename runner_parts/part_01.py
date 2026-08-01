        lp = self.local_root / rel; lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        self.mirror(lp, rel); return lp

    def save_df(self, df, rel, index=False):
        lp = self.local_root / rel; lp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(lp, index=index)
        self.mirror(lp, rel); return lp

    def save_figure(self, fig, rel, dpi=180):
        lp = self.local_root / rel; lp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(lp, dpi=dpi, bbox_inches="tight")
        self.mirror(lp, rel); return lp

STORE = ArtifactStore(LOCAL_RUN, DRIVE_RUN)
STORE.save_json(CFG, "config.json")

# ---------------------------- 3. REPRODUCIBILITY ----------------------------
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "device": str(DEVICE),
}
STORE.save_json(ENV, "environment.json")

# ---------------------------- 4. DISPLAY HELPERS ----------------------------
def section(title):
    display(Markdown(f"## {title}"))

def fmt_metrics(m):
    keys = ["auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "brier", "ece"]
    return {k: round(float(m[k]), 4) if k in m and pd.notna(m[k]) else np.nan for k in keys}

class LiveDashboard:
    def __init__(self):
        self.fold_rows = []

    def update(self, fold, epoch, history, note=""):
        clear_output(wait=True)
        display(Markdown(f"# RimGraph-DG Live Run\n**Device:** `{ENV['gpu']}`  \n**Run:** `{RUN_ID}`  \n**Fold:** held-out `{fold}`  \n**Epoch:** `{epoch}/{CFG['epochs']}`  \n{note}"))
        if history:
            h = pd.DataFrame(history)
            cols = [c for c in ["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "val_f1", "val_dice_disc", "val_dice_cup"] if c in h]
            display(h[cols].tail(10).style.format(precision=4).hide(axis="index"))
            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(h["epoch"], h["train_loss"], label="Train")
            ax1.plot(h["epoch"], h["val_loss"], label="Validation")
            ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=.2)
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(h["epoch"], h["val_auroc"], label="AUROC")
            ax2.plot(h["epoch"], h["val_auprc"], label="AUPRC")
            ax2.set_ylim(0, 1.02); ax2.set_title("Validation discrimination"); ax2.legend(); ax2.grid(alpha=.2)
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(h["epoch"], h["val_dice_disc"], label="Disc Dice")
            ax3.plot(h["epoch"], h["val_dice_cup"], label="Cup Dice")
            ax3.set_ylim(0, 1.02); ax3.set_title("Validation segmentation"); ax3.legend(); ax3.grid(alpha=.2)
            plt.tight_layout(); display(fig); plt.close(fig)
        if self.fold_rows:
            display(Markdown("### Completed external folds"))
            display(pd.DataFrame(self.fold_rows).style.format(precision=4).hide(axis="index"))

DASH = LiveDashboard()

# ---------------------------- 5. DATA DOWNLOAD ----------------------------
def resolve_data_root():
    manual = str(CFG.get("manual_data_dir", "")).strip()
    if manual and Path(manual).exists():
        return Path(manual)
    candidates = [
        DRIVE_ROOT / "data" / "glaucoma-datasets",
        Path("/content/glaucoma-datasets"),
        Path("/content/drive/MyDrive/glaucoma-datasets") if IN_COLAB else Path("__missing__"),
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*")):
            return c
    import kagglehub
    section("Downloading/locating Kaggle dataset")
    path = Path(kagglehub.dataset_download(CFG["kaggle_dataset"]))
    return path

DATA_ROOT = resolve_data_root()
display(Markdown(f"**Dataset root:** `{DATA_ROOT}`"))

# ---------------------------- 6. METADATA DISCOVERY ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def norm_key(x):
    return re.sub(r"[^a-z0-9]", "", Path(str(x)).stem.lower())

def infer_source(path):
    s = str(path).lower()
    if "g1020" in s: return "G1020"
    if "origa" in s: return "ORIGA"
    if "refuge" in s: return "REFUGE"
    return None

def is_mask_path(path):
    s = str(path).lower()
    return any(k in s for k in ["mask", "groundtruth", "ground_truth", "annotation", "segmentation", "gt/"])

def normalize_label(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (bool, np.bool_)): return int(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        if float(v) in [0.0, 1.0]: return int(v)
    s = str(v).strip().lower()
    positive = ["glaucoma", "glaucomatous", "positive", "abnormal", "yes", "true", "case", "g"]
    negative = ["normal", "healthy", "non-glaucoma", "nonglaucoma", "negative", "no", "false", "control", "n"]
    if s in positive or any(s.startswith(x) for x in ["glaucoma", "positive"]): return 1
    if s in negative or any(s.startswith(x) for x in ["normal", "healthy", "negative"]): return 0
    try:
        z = float(s)
        if z in [0, 1]: return int(z)
    except Exception: pass
    return np.nan

def _json_records(obj):
    records = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict): records.append(item)
            records.extend(_json_records(item))
    elif isinstance(obj, dict):
        # A dictionary may itself be a record or a filename -> label mapping.
