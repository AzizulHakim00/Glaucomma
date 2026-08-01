        records.append(obj)
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)) and re.search(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", str(k), re.I):
                records.append({"image": k, "label": v})
            records.extend(_json_records(v))
    return records

def load_tables(root):
    tables = []
    for p in list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")):
        try:
            df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
            df.columns = [str(c).strip() for c in df.columns]
            tables.append((p, df))
        except Exception:
            continue
    for p in root.rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            recs = _json_records(obj)
            if recs:
                df = pd.json_normalize(recs, sep="_")
                if not df.empty: tables.append((p, df))
        except Exception:
            continue
    return tables

TABLES = load_tables(DATA_ROOT)

def build_label_index(tables):
    idx = {}
    file_candidates = ["image", "filename", "file", "img", "imgname", "img_name", "name", "image_name", "imageid", "id"]
    label_candidates = ["glaucoma", "label", "class", "diagnosis", "target", "binary", "binarylabels", "binary_labels", "is_glaucoma", "glaucoma_label"]
    for path, df in tables:
        lower = {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in df.columns}
        fcol = next((lower[c] for c in file_candidates if c in lower), None)
        lcol = next((lower[c] for c in label_candidates if c in lower), None)
        if fcol is None:
            for c in df.columns:
                vals = df[c].astype(str)
                if vals.str.contains(r"\.(jpg|jpeg|png|tif|tiff|bmp)$", case=False, regex=True).mean() > .20:
                    fcol = c; break
        if lcol is None:
            for c in df.columns:
                mapped = df[c].map(normalize_label)
                if mapped.notna().mean() > .75 and mapped.nunique(dropna=True) <= 2:
                    lcol = c; break
        if fcol is None or lcol is None: continue
        src = infer_source(path)
        for _, r in df[[fcol, lcol]].dropna(subset=[fcol]).iterrows():
            y = normalize_label(r[lcol])
            if pd.isna(y): continue
            key = norm_key(r[fcol])
            idx[(src, key)] = int(y)
            idx[(None, key)] = int(y)
    return idx

LABEL_INDEX = build_label_index(TABLES)

def folder_label(path):
    parts = [x.lower() for x in path.parts]
    for p in parts:
        if p in {"glaucoma", "glaucomatous", "positive", "abnormal", "g"}: return 1
        if p in {"normal", "healthy", "negative", "control", "non-glaucoma", "nonglaucoma", "n"}: return 0
    # REFUGE training images commonly use g#### for glaucoma and n#### for normal.
    stem = Path(path).stem.lower()
    if infer_source(path) == "REFUGE":
        if re.fullmatch(r"g\d+", stem): return 1
        if re.fullmatch(r"n\d+", stem): return 0
    return np.nan

def infer_dataset_split(path):
    parts = [str(x).lower() for x in Path(path).parts]
    if any(x in {"train", "training"} for x in parts): return "train"
    if any(x in {"validation", "valid", "val", "offline-validation", "offline_validation"} for x in parts): return "validation"
    if any(x in {"test", "testing", "onsite-test", "onsite_test"} for x in parts): return "test"
    return "unspecified"

def build_mask_index(root):
    masks = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and is_mask_path(p):
            src = infer_source(p); key = norm_key(p)
            # Remove common mask suffixes to improve matching.
            key2 = re.sub(r"(mask|segmentation|groundtruth|groundtruths|gt|od|oc|disc|cup)$", "", key)
            masks.setdefault((src, key), []).append(p)
            masks.setdefault((src, key2), []).append(p)
    return masks

MASK_INDEX = build_mask_index(DATA_ROOT)

def choose_mask(src, image_path):
    key = norm_key(image_path)
    candidates = MASK_INDEX.get((src, key), [])
    if not candidates:
        for (s, k), vals in MASK_INDEX.items():
            if s == src and (k == key or k in key or key in k):
                candidates.extend(vals)
                if len(candidates) >= 3: break
    if not candidates: return None
    # Prefer combined/cropped mask close to image path characteristics.
    cropped = "cropped" in str(image_path).lower()
    candidates = sorted(set(candidates), key=lambda p: ("cropped" in str(p).lower()) != cropped)
    return str(candidates[0])

def discover_metadata(root):
    rows = []
    all_images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS and not is_mask_path(p)]
    # Prefer original Images folders; avoid duplicates from Images_Cropped when originals exist.
    for p in all_images:
        src = infer_source(p)
        if src not in CFG["sources"]: continue
        y = LABEL_INDEX.get((src, norm_key(p)), LABEL_INDEX.get((None, norm_key(p)), np.nan))
        label_origin = "annotation_table" if not pd.isna(y) else "unresolved"
        if pd.isna(y):
            y = folder_label(p)
            if not pd.isna(y): label_origin = "folder_or_filename"
        parts_lower = [q.lower() for q in p.parts]
        repr_rank = 0 if "images" in parts_lower else (1 if "images_square" in parts_lower else (2 if "images_cropped" in parts_lower else 3))
        rows.append({
            "image_path": str(p), "mask_path": choose_mask(src, p),
            "source": src, "dataset_split": infer_dataset_split(p),
            "label": y, "label_origin": label_origin,
            "is_cropped": int("cropped" in str(p).lower()),
            "representation_rank": repr_rank, "image_key": norm_key(p)
        })
    meta = pd.DataFrame(rows)
    if meta.empty: raise RuntimeError(f"No images found under {root}")

    # Prefer uncropped version per source/key; retain cropped only when no original exists.
    meta = meta.sort_values(["source", "image_key", "representation_rank", "is_cropped"]).drop_duplicates(["source", "image_key"], keep="first")
    STORE.save_df(meta, "metadata_all_discovered.csv")

    unresolved = meta["label"].isna()
    excluded = meta.loc[unresolved].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = "missing_public_label_in_downloaded_package"
        STORE.save_df(excluded, "excluded_unlabelled_images.csv")
        excluded_summary = (
            excluded.groupby(["source", "dataset_split"]).size()
            .rename("Excluded unlabeled").reset_index()
        )
        display(Markdown(
            "### Unlabelled images excluded safely\n"
            "The downloaded package contains images without accessible glaucoma labels. "
            "They are **not assigned guessed labels** and are excluded from supervised training, "
            "validation, calibration, and testing to prevent invalid results or leakage."
        ))
        display(excluded_summary)
        display(Markdown("A complete list was saved as `excluded_unlabelled_images.csv` in both Colab and Drive outputs."))

    meta = meta.loc[~unresolved].copy()
    if meta.empty:
        raise RuntimeError(
            "No labelled images remain after safely excluding images without public labels. "
            "Provide a verified annotation file through CFG['manual_data_dir']."
        )
    meta["label"] = meta["label"].astype(int)
    meta = meta.drop_duplicates(["source", "image_path"]).reset_index(drop=True)

    # Scientific validity check: every configured source used in a binary fold must contain both classes.
    invalid_sources = []
    for src in CFG["sources"]:
        src_labels = sorted(meta.loc[meta["source"] == src, "label"].unique().tolist())
        if src_labels != [0, 1]:
            invalid_sources.append((src, src_labels, int((meta["source"] == src).sum())))
    if invalid_sources:
        details = "; ".join(f"{s}: labels={labs}, n={n}" for s, labs, n in invalid_sources)
        raise RuntimeError(
            "At least one configured source does not contain both verified classes after filtering: "
            f"{details}. Add verified labels or remove that source from CFG['sources']/CFG['fold_targets']."
        )
    return meta, excluded

META, EXCLUDED_UNLABELLED = discover_metadata(DATA_ROOT)
STORE.save_df(META, "metadata.csv")

section("Dataset audit")
audit = META.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "Normal", 1: "Glaucoma"})
for col in ["Normal", "Glaucoma"]:
    if col not in audit.columns: audit[col] = 0
audit["Total labelled"] = audit["Normal"] + audit["Glaucoma"]
audit["With mask"] = META.groupby("source")["mask_path"].apply(lambda x: x.notna().sum())
if not EXCLUDED_UNLABELLED.empty:
    excluded_counts = EXCLUDED_UNLABELLED.groupby("source").size()
    audit["Excluded unlabeled"] = excluded_counts.reindex(audit.index, fill_value=0)
else:
    audit["Excluded unlabeled"] = 0
audit = audit[["Normal", "Glaucoma", "Total labelled", "With mask", "Excluded unlabeled"]]
display(audit)

audit_split = META.groupby(["source", "dataset_split", "label"]).size().unstack(fill_value=0).rename(columns={0: "Normal", 1: "Glaucoma"})
for col in ["Normal", "Glaucoma"]:
    if col not in audit_split.columns: audit_split[col] = 0
audit_split["Total labelled"] = audit_split["Normal"] + audit_split["Glaucoma"]
display(Markdown("### Labelled images by original package split"))
display(audit_split.reset_index())

# ---------------------------- 7. IMAGE + MASK UTILITIES ----------------------------