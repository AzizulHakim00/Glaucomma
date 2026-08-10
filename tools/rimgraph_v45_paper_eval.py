#!/usr/bin/env python3
"""Paper-grade result auditor for RimGraph-DG V4.5.

This script is intentionally read-only with respect to training checkpoints.
It audits completed experiment artifacts, validates that the full RimGraph stage
was actually trained, computes cross-domain summaries, and performs paired
stratified bootstrap comparisons between the global baseline and RimGraph when
both prediction files are present for the same held-out domain/seed.

Expected run root (Google Drive example):
  /content/drive/MyDrive/Glaucomma_RimGraphDG/paper_run_v45

Outputs are written under <run_root>/paper_eval_v45 by default.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TARGETS = ("ORIGA", "REFUGE", "G1020")
DEFAULT_SEEDS = (2029, 2030, 2031)
MODELS = ("global_baseline", "rimgraph_v4")


@dataclass
class ArtifactStatus:
    held_out_source: str
    seed: int
    model_type: str
    completed: bool
    metrics_exists: bool
    predictions_exists: bool
    history_exists: bool
    full_stage_seen: Optional[bool]
    finite_val_disc_dice: Optional[bool]
    finite_val_cup_dice: Optional[bool]
    notes: str = ""


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan")


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if np.unique(y).size > 1 else float("nan")


def _prediction_key(df: pd.DataFrame) -> pd.Series:
    if "path" in df.columns:
        return df["path"].astype(str)
    if "image_path" in df.columns:
        return df["image_path"].astype(str)
    raise ValueError("Prediction CSV needs a 'path' or 'image_path' column for paired comparison")


def _prob_col(df: pd.DataFrame) -> str:
    if "prob_calibrated" in df.columns:
        return "prob_calibrated"
    if "prob_raw" in df.columns:
        return "prob_raw"
    raise ValueError("Prediction CSV has neither prob_calibrated nor prob_raw")


def _audit_one(run_root: Path, target: str, seed: int, model: str) -> ArtifactStatus:
    rel = run_root / "folds" / target / f"seed_{seed}" / model
    completed_path = rel / "COMPLETED.json"
    metrics_path = rel / "metrics.json"
    pred_path = rel / "test_predictions.csv"
    history_path = rel / "history.csv"

    full_stage_seen = None
    finite_disc = None
    finite_cup = None
    notes: List[str] = []

    if model == "rimgraph_v4" and history_path.exists():
        h = pd.read_csv(history_path)
        if "stage" in h.columns:
            full_stage_seen = bool((h["stage"].astype(str) == "full").any())
        else:
            full_stage_seen = False
            notes.append("history missing stage column")
        if "val_dice_disc" in h.columns:
            finite_disc = bool(np.isfinite(pd.to_numeric(h["val_dice_disc"], errors="coerce")).any())
        else:
            finite_disc = False
        if "val_dice_cup" in h.columns:
            finite_cup = bool(np.isfinite(pd.to_numeric(h["val_dice_cup"], errors="coerce")).any())
        else:
            finite_cup = False

    completed = completed_path.exists()
    if completed and metrics_path.exists():
        try:
            m = _read_json(metrics_path)
            if str(m.get("held_out_source", target)) != target:
                notes.append("metrics held_out_source mismatch")
            if int(m.get("seed", seed)) != seed:
                notes.append("metrics seed mismatch")
        except Exception as exc:
            notes.append(f"metrics JSON unreadable: {exc}")

    return ArtifactStatus(
        held_out_source=target,
        seed=seed,
        model_type=model,
        completed=completed,
        metrics_exists=metrics_path.exists(),
        predictions_exists=pred_path.exists(),
        history_exists=history_path.exists(),
        full_stage_seen=full_stage_seen,
        finite_val_disc_dice=finite_disc,
        finite_val_cup_dice=finite_cup,
        notes="; ".join(notes),
    )


def audit_artifacts(run_root: Path, targets: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    rows = [asdict(_audit_one(run_root, t, int(s), m)) for s in seeds for t in targets for m in MODELS]
    return pd.DataFrame(rows)


def collect_metrics(run_root: Path, targets: Sequence[str], seeds: Sequence[int]) -> pd.DataFrame:
    rows: List[dict] = []
    for seed in seeds:
        for target in targets:
            for model in MODELS:
                p = run_root / "folds" / target / f"seed_{seed}" / model / "metrics.json"
                if not p.exists():
                    continue
                rec = _read_json(p)
                rec.setdefault("held_out_source", target)
                rec.setdefault("seed", int(seed))
                rec.setdefault("model_type", model)
                rows.append(rec)
    return pd.DataFrame(rows)


def stratified_paired_bootstrap(
    baseline: pd.DataFrame,
    full: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict:
    b = baseline.copy()
    f = full.copy()
    b["_key"] = _prediction_key(b)
    f["_key"] = _prediction_key(f)
    bprob, fprob = _prob_col(b), _prob_col(f)
    keep_b = b[["_key", "label", bprob]].rename(columns={"label": "label_b", bprob: "p_b"})
    keep_f = f[["_key", "label", fprob]].rename(columns={"label": "label_f", fprob: "p_f"})
    x = keep_b.merge(keep_f, on="_key", how="inner", validate="one_to_one")
    if x.empty:
        raise ValueError("No paired prediction rows matched between baseline and RimGraph")
    if not np.array_equal(x.label_b.to_numpy(), x.label_f.to_numpy()):
        raise ValueError("Paired predictions have label disagreement")

    y = x.label_b.to_numpy(dtype=int)
    pb = x.p_b.to_numpy(dtype=float)
    pf = x.p_f.to_numpy(dtype=float)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Paired bootstrap requires both classes")

    if metric == "auroc":
        fn = _safe_auc
    elif metric == "auprc":
        fn = _safe_ap
    else:
        raise ValueError(metric)

    point_b = fn(y, pb)
    point_f = fn(y, pf)
    point_delta = point_f - point_b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.r_[rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        deltas[i] = fn(y[idx], pf[idx]) - fn(y[idx], pb[idx])
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    p_two = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    p_two = min(1.0, max(0.0, p_two))
    return {
        "metric": metric,
        "n_paired": int(len(x)),
        "baseline": float(point_b),
        "rimgraph": float(point_f),
        "delta": float(point_delta),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "bootstrap_p_two_sided": float(p_two),
    }


def paired_comparisons(
    run_root: Path,
    targets: Sequence[str],
    seeds: Sequence[int],
    n_boot: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for seed in seeds:
        for target in targets:
            base = run_root / "folds" / target / f"seed_{seed}" / "global_baseline" / "test_predictions.csv"
            full = run_root / "folds" / target / f"seed_{seed}" / "rimgraph_v4" / "test_predictions.csv"
            if not (base.exists() and full.exists()):
                continue
            b = pd.read_csv(base)
            f = pd.read_csv(full)
            for metric in ("auroc", "auprc"):
                rec = stratified_paired_bootstrap(b, f, metric, n_boot=n_boot, seed=int(seed) + (0 if metric == "auroc" else 10000))
                rec.update({"held_out_source": target, "seed": int(seed)})
                rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    preferred = ["auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "brier", "ece", "dice_disc", "dice_cup"]
    rows: List[dict] = []
    for model, group in metrics.groupby("model_type"):
        for metric in preferred:
            if metric not in group.columns:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                continue
            lower_better = metric in {"brier", "ece"}
            rows.append({
                "model_type": model,
                "metric": metric,
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
                "worst_observed": float(vals.max() if lower_better else vals.min()),
            })
    return pd.DataFrame(rows)


def mask_audit_summary(run_root: Path) -> pd.DataFrame:
    p = run_root / "mask_validity_by_source.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def build_readiness(audit: pd.DataFrame, metrics: pd.DataFrame, comparisons: pd.DataFrame, targets: Sequence[str], seeds: Sequence[int]) -> dict:
    required = len(targets) * len(seeds) * len(MODELS)
    completed = int(audit.completed.sum()) if not audit.empty else 0
    rim = audit[audit.model_type == "rimgraph_v4"] if not audit.empty else pd.DataFrame()
    rim_full_ok = bool(rim.full_stage_seen.map(lambda x: bool(x) if pd.notna(x) else False).all()) if len(rim) == len(targets) * len(seeds) else False
    rim_dice_ok = bool((rim.finite_val_disc_dice.map(lambda x: bool(x) if pd.notna(x) else False) & rim.finite_val_cup_dice.map(lambda x: bool(x) if pd.notna(x) else False)).all()) if len(rim) == len(targets) * len(seeds) else False
    paired_needed = len(targets) * len(seeds) * 2
    paired_done = int(len(comparisons))
    return {
        "required_model_runs": required,
        "completed_model_runs": completed,
        "all_model_runs_complete": completed == required,
        "all_rimgraph_runs_reached_full_stage": rim_full_ok,
        "all_rimgraph_runs_have_finite_validation_dice": rim_dice_ok,
        "required_paired_comparisons": paired_needed,
        "completed_paired_comparisons": paired_done,
        "paper_grade_core_results_ready": bool(completed == required and rim_full_ok and rim_dice_ok and paired_done == paired_needed),
    }


def write_ablation_manifest(out_dir: Path) -> Path:
    manifest = {
        "policy": "Every ablation is a separate fresh training run. Never mutate or resume the primary V4.5 checkpoint into an ablation.",
        "primary": "Full RimGraph-DG",
        "minimum_required": [
            {"name": "GlobalBaseline", "global": True, "segmentation": False, "local": False, "structural": False, "graph": False, "domain_adv": False, "prototype": False},
            {"name": "Global+Seg", "global": True, "segmentation": True, "local": False, "structural": False, "graph": False, "domain_adv": False, "prototype": False},
            {"name": "AnatomyFusion_noGraph", "global": True, "segmentation": True, "local": True, "structural": True, "graph": False, "domain_adv": False, "prototype": False},
            {"name": "RimGraph_noDG", "global": True, "segmentation": True, "local": True, "structural": True, "graph": True, "domain_adv": False, "prototype": False},
            {"name": "RimGraph_DANN", "global": True, "segmentation": True, "local": True, "structural": True, "graph": True, "domain_adv": True, "prototype": False},
            {"name": "Full_RimGraph_DG", "global": True, "segmentation": True, "local": True, "structural": True, "graph": True, "domain_adv": True, "prototype": True},
        ],
        "reviewer_critical": [
            "Rim sectors -> pooled MLP versus Rim sectors -> GAT",
            "Raw rim-sector vector versus reflection/orientation-robust rim spectrum",
            "12 sectors versus one nearby alternative (e.g. 8 or 16) on one development protocol only",
        ],
        "report": ["AUROC", "AUPRC", "sensitivity", "specificity", "F1", "MCC", "Brier", "ECE", "Disc Dice", "Cup Dice"],
    }
    path = out_dir / "ablation_manifest_v45.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_root = args.run_root.expanduser().resolve()
    if not run_root.exists():
        raise SystemExit(f"Run root does not exist: {run_root}")
    out = args.out.expanduser().resolve() if args.out else (run_root / "paper_eval_v45")
    out.mkdir(parents=True, exist_ok=True)

    audit = audit_artifacts(run_root, args.targets, args.seeds)
    metrics = collect_metrics(run_root, args.targets, args.seeds)
    comparisons = paired_comparisons(run_root, args.targets, args.seeds, n_boot=args.bootstrap)
    aggregate = aggregate_metrics(metrics)
    mask_summary = mask_audit_summary(run_root)
    readiness = build_readiness(audit, metrics, comparisons, args.targets, args.seeds)

    audit.to_csv(out / "experiment_completeness.csv", index=False)
    metrics.to_csv(out / "collected_metrics.csv", index=False)
    comparisons.to_csv(out / "paired_bootstrap_baseline_vs_rimgraph.csv", index=False)
    aggregate.to_csv(out / "aggregate_metrics.csv", index=False)
    if not mask_summary.empty:
        mask_summary.to_csv(out / "mask_validity_by_source_copy.csv", index=False)
    (out / "paper_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    write_ablation_manifest(out)

    print("=== RimGraph-DG V4.5 Paper Evaluation Audit ===")
    print(f"Run root: {run_root}")
    print(f"Output:   {out}")
    print(json.dumps(readiness, indent=2))
    if not comparisons.empty:
        print("\nPaired bootstrap comparisons:")
        print(comparisons.round(4).to_string(index=False))
    else:
        print("\nNo complete baseline-vs-RimGraph prediction pairs yet; this is expected while training is incomplete.")


if __name__ == "__main__":
    main()
