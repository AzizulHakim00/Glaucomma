# ---------------------------- 1. DATA DISCOVERY ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

if CFG["manual_data_dir"]:
    DATA_ROOT = Path(CFG["manual_data_dir"])
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"manual_data_dir does not exist: {DATA_ROOT}")
else:
    import kagglehub
    section("Downloading or locating Kaggle dataset")
    DATA_ROOT = Path(kagglehub.dataset_download(CFG["kaggle_dataset"]))
print("Dataset root:", DATA_ROOT)


def infer_source(path):
    text = str(path).lower()
    if "origa" in text: return "ORIGA"
    if "refuge" in text: return "REFUGE"
    if "g1020" in text or "g_1020" in text: return "G1020"
    return None


def normalize_key(value):
    stem = Path(str(value)).stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def normalize_label(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return np.nan
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1): return int(value)
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0): return int(value)
    text = str(value).strip().lower()
    if text in {"1", "g", "glaucoma", "glaucomatous", "positive", "abnormal", "yes", "true"}: return 1
    if text in {"0", "n", "normal", "healthy", "negative", "control", "non-glaucoma", "nonglaucoma", "no", "false"}: return 0
    return np.nan


def normalize_laterality(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return "U"
    text = str(value).strip().lower()
    if text in {"r", "right", "od", "right eye", "re"}: return "R"
    if text in {"l", "left", "os", "left eye", "le"}: return "L"
    return "U"


def infer_split(path):
    parts = [str(x).lower() for x in Path(path).parts]
    if any(x in {"train", "training"} for x in parts): return "train"
    if any(x in {"validation", "valid", "val", "offline-validation", "offline_validation"} for x in parts): return "validation"
    if any(x in {"test", "testing", "onsite-test", "onsite_test"} for x in parts): return "test"
    return "unspecified"


def is_mask_path(path):
    text = str(path).lower().replace("\\", "/")
    stem = Path(path).stem.lower()
    mask_tokens = ["/mask", "/groundtruth", "/ground_truth", "/segmentation", "optic_disc", "optic_cup"]
    if any(t in text for t in mask_tokens): return True
    return bool(re.search(r"(?:^|[_-])(mask|disc|cup|od|oc|gt)(?:$|[_-])", stem))


def _json_records(obj):
    records = []
    if isinstance(obj, list):
        for v in obj: records.extend(_json_records(v))
    elif isinstance(obj, dict):
        records.append(obj)
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)) and re.search(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", str(k), re.I):
                records.append({"image": k, "label": v})
            records.extend(_json_records(v))
    return records


def load_annotation_tables(root):
    tables = []
    files = list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")) + list(root.rglob("*.xls"))
    for p in files:
        try:
            df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                tables.append((p, df))
        except Exception:
            pass
    for p in root.rglob("*.json"):
        try:
            recs = _json_records(json.loads(p.read_text(encoding="utf-8", errors="ignore")))
            if recs:
                df = pd.json_normalize(recs, sep="_")
                if not df.empty: tables.append((p, df))
        except Exception:
            pass
    return tables


TABLES = load_annotation_tables(DATA_ROOT)


def column_map(df):
    return {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in df.columns}


def choose_column(lower, names):
    return next((lower[n] for n in names if n in lower), None)


def build_annotation_index(tables):
    idx, lat_idx = {}, {}
    file_names = ["image", "filename", "file", "img", "imgname", "img_name", "image_name", "imageid", "image_id", "id", "name"]
    label_names = ["glaucoma", "label", "class", "diagnosis", "target", "binary", "binarylabels", "binary_labels", "is_glaucoma", "glaucoma_label"]
    lat_names = ["laterality", "eye", "side", "eye_side", "left_right"]
    for path, df in tables:
        lower = column_map(df)
        fcol = choose_column(lower, file_names)
        lcol = choose_column(lower, label_names)
        latcol = choose_column(lower, lat_names)
        if fcol is None:
            for c in df.columns:
                vals = df[c].astype(str)
                if vals.str.contains(r"\.(jpg|jpeg|png|tif|tiff|bmp)$", case=False, regex=True).mean() > 0.20:
                    fcol = c; break
        if lcol is None:
            for c in df.columns:
                mapped = df[c].map(normalize_label)
                if mapped.notna().mean() > 0.75 and mapped.nunique(dropna=True) <= 2:
                    lcol = c; break
        if fcol is None: continue
        src = infer_source(path)
        for _, row in df.iterrows():
            key = normalize_key(row.get(fcol))
            if not key: continue
            if lcol is not None:
                y = normalize_label(row.get(lcol))
                if not pd.isna(y):
                    idx[(src, key)] = int(y)
                    idx[(None, key)] = int(y)
            if latcol is not None:
                lat = normalize_laterality(row.get(latcol))
                if lat != "U":
                    lat_idx[(src, key)] = lat
                    lat_idx[(None, key)] = lat
    return idx, lat_idx


LABEL_INDEX, LATERALITY_INDEX = build_annotation_index(TABLES)


def fallback_label(path):
    parts = [x.lower() for x in Path(path).parts]
    for token in parts:
        if token in {"glaucoma", "glaucomatous", "positive", "abnormal"}: return 1
        if token in {"normal", "healthy", "negative", "control", "non-glaucoma", "nonglaucoma"}: return 0
    stem = Path(path).stem.lower()
    if infer_source(path) == "REFUGE":
        if re.fullmatch(r"g\d+", stem): return 1
        if re.fullmatch(r"n\d+", stem): return 0
    return np.nan


def mask_kind(path):
    text = str(path).lower().replace("\\", "/")
    stem = Path(path).stem.lower()
    if re.search(r"(?:^|[_-])(cup|oc)(?:$|[_-])", stem) or "/cup" in text: return "cup"
    if re.search(r"(?:^|[_-])(disc|disk|od)(?:$|[_-])", stem) or "/disc" in text: return "disc"
    return "combined"


def mask_key(path):
    key = normalize_key(path)
    for suffix in ["segmentation", "groundtruth", "groundtruths", "mask", "disc", "disk", "cup", "od", "oc", "gt"]:
        if key.endswith(suffix): key = key[:-len(suffix)]
    return key


def build_mask_index(root):
    idx = defaultdict(lambda: {"combined": [], "disc": [], "cup": []})
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and is_mask_path(p):
            src = infer_source(p)
            idx[(src, mask_key(p))][mask_kind(p)].append(p)
            idx[(src, normalize_key(p))][mask_kind(p)].append(p)
    return idx


MASK_INDEX = build_mask_index(DATA_ROOT)


def _mask_rank(path, image_path):
    text = str(path).lower().replace("\\", "/")
    image_text = str(image_path).lower()
    representation_penalty = int(("cropped" in text) != ("cropped" in image_text)) + int(("square" in text) != ("square" in image_text))
    return (representation_penalty, len(text), text)


def choose_masks(source, image_path):
    key = normalize_key(image_path)
    candidates = {"combined": [], "disc": [], "cup": []}
    for lookup in [(source, key), (source, mask_key(image_path))]:
        if lookup in MASK_INDEX:
            for kind in candidates: candidates[kind].extend(MASK_INDEX[lookup][kind])
    if not any(candidates.values()):
        for (src, k), vals in MASK_INDEX.items():
            if src == source and (k == key or (len(k) > 3 and (k in key or key in k))):
                for kind in candidates: candidates[kind].extend(vals[kind])
    out = {}
    for kind, vals in candidates.items():
        vals = sorted(set(vals), key=lambda p: _mask_rank(p, image_path))
        out[kind] = str(vals[0]) if vals else None
    return out


def image_fingerprint(path):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None: return None
    arr = cv2.resize(arr, (17, 16), interpolation=cv2.INTER_AREA)
    bits = arr[:, 1:] > arr[:, :-1]
    packed = np.packbits(bits.astype(np.uint8)).tobytes()
    return hashlib.sha1(packed).hexdigest()


def discover_metadata(root):
    rows = []
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS and not is_mask_path(p)]
    for i, p in enumerate(images):
        src = infer_source(p)
        if src not in CFG["sources"]: continue
        key = normalize_key(p)
        y = LABEL_INDEX.get((src, key), LABEL_INDEX.get((None, key), np.nan))
        origin = "annotation_table" if not pd.isna(y) else "unresolved"
        if pd.isna(y):
            y = fallback_label(p)
            if not pd.isna(y): origin = "folder_or_canonical_filename"
        lat = LATERALITY_INDEX.get((src, key), LATERALITY_INDEX.get((None, key), "U"))
        masks = choose_masks(src, p)
        parts = [q.lower() for q in p.parts]
        rank = 0 if "images" in parts else (1 if "images_square" in parts else (2 if "images_cropped" in parts else 3))
        rows.append({
            "image_path": str(p), "source": src, "dataset_split": infer_split(p),
            "label": y, "label_origin": origin, "laterality": lat,
            "combined_mask_path": masks["combined"], "disc_mask_path": masks["disc"], "cup_mask_path": masks["cup"],
            "representation_rank": rank, "image_key": key,
        })
    meta = pd.DataFrame(rows)
    if meta.empty: raise RuntimeError(f"No source images found under {root}")
    meta = meta.sort_values(["source", "image_key", "representation_rank"]).drop_duplicates(["source", "image_key"], keep="first")
    meta["fingerprint"] = [image_fingerprint(p) for p in meta.image_path]
    STORE.save_df(meta, "metadata_all_discovered.csv")

    unresolved = meta.label.isna()
    excluded = meta.loc[unresolved].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = "missing_verified_public_label"
        STORE.save_df(excluded, "excluded_unlabelled_images.csv")
        display(Markdown("### Unlabelled images excluded safely"))
        display(excluded.groupby(["source", "dataset_split"]).size().rename("excluded").reset_index())
    meta = meta.loc[~unresolved].copy()
    meta["label"] = meta["label"].astype(int)
    for src in CFG["sources"]:
        labels = sorted(meta.loc[meta.source == src, "label"].unique().tolist())
        if labels != [0, 1]:
            raise RuntimeError(f"{src} does not contain both verified classes after filtering: {labels}")
    return meta.reset_index(drop=True), excluded.reset_index(drop=True)


META, EXCLUDED_UNLABELLED = discover_metadata(DATA_ROOT)
STORE.save_df(META, "metadata.csv")

section("Dataset audit")
audit = META.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "Normal", 1: "Glaucoma"})
for c in ["Normal", "Glaucoma"]:
    if c not in audit.columns: audit[c] = 0
audit["Total labelled"] = audit["Normal"] + audit["Glaucoma"]
audit["With any mask"] = META[["combined_mask_path", "disc_mask_path", "cup_mask_path"]].notna().any(axis=1).groupby(META.source).sum()
audit["Known laterality"] = (META.laterality != "U").groupby(META.source).sum()
audit["Excluded unlabeled"] = EXCLUDED_UNLABELLED.groupby("source").size().reindex(audit.index, fill_value=0) if not EXCLUDED_UNLABELLED.empty else 0
display(audit.reset_index())
STORE.save_df(audit.reset_index(), "dataset_audit.csv")

cross_dup = META.dropna(subset=["fingerprint"]).groupby("fingerprint").filter(lambda g: g.source.nunique() > 1)
if not cross_dup.empty:
    STORE.save_df(cross_dup.sort_values(["fingerprint", "source"]), "cross_source_duplicate_candidates.csv")
    display(Markdown(f"**Cross-source duplicate candidates detected:** {cross_dup.fingerprint.nunique()}. They will be removed from training against each held-out fold."))
else:
    STORE.save_df(pd.DataFrame(columns=META.columns), "cross_source_duplicate_candidates.csv")
