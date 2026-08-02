# Glaucomma — RimGraph-DG V4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/Glaucomma/blob/main/Glaucomma_RimGraphDG_Single_Cell.ipynb)

A reproducible Google Colab research pipeline for cross-dataset glaucoma screening using ORIGA, REFUGE, and G1020.

## Why V4

V4 replaces the unstable deepest-feature segmentation design with a staged, testable research pipeline:

- global classification baseline for every held-out dataset;
- multi-scale ConvNeXt FPN with GroupNorm for optic-disc/cup segmentation;
- high-resolution differentiable optic-disc ROI pooling;
- 12-sector polar rim graph built from high-resolution FPN features;
- separate validity handling for disc, cup, and vCDR annotations;
- source/class-balanced mini-batches;
- staged activation: global+segmentation, anatomy fusion, then graph/domain objectives;
- cross-source perceptual-duplicate detection and fold-specific leakage removal;
- laterality-aware canonicalization when verified eye-side metadata is available;
- identity, standard-temperature, and robust-temperature calibration compared on source validation only;
- bootstrap confidence intervals, calibration, XAI, baseline-versus-full comparison, and an automatic scientific diagnostic report.

## Run

Open the notebook using the badge, select a T4 GPU runtime, and run its only cell. The notebook is pinned to an immutable V4 code revision and verifies the combined runner with SHA-256 before execution.

Default execution runs one seed for:

1. held-out ORIGA;
2. held-out REFUGE;
3. held-out G1020;
4. a global ConvNeXt baseline and the staged RimGraph-DG V4 model in every fold.

For a quick loader/runtime check, temporarily set `fast_dev_run=True`. Smoke-test outputs are not paper results.

For final multi-seed statistics after the first complete V4 run is reviewed, use:

```python
'seeds': [2029, 2030, 2031]
```

Enable `run_optuna=True` only after the untuned baseline and full V4 results are scientifically evaluated.

## Output locations

- Colab: `/content/Glaucomma_runs/paper_run_v4/`
- Drive: `/content/drive/MyDrive/Glaucomma_RimGraphDG/paper_run_v4/`
- ZIP: `/content/drive/MyDrive/Glaucomma_RimGraphDG/paper_run_v4.zip`

Main outputs include:

- `dataset_audit.csv`
- `excluded_unlabelled_images.csv`
- `cross_source_duplicate_candidates.csv`
- `fold_model_summary.csv`
- `aggregate_metrics.csv`
- `baseline_vs_full.csv`
- `all_external_predictions.csv`
- `scientific_diagnostic_report.md`
- `artifact_manifest.csv`
- split CSVs, `.pt` checkpoints, histories, calibration tables, predictions, figures, and XAI files for each source/seed/model.

## Scientific safeguards

The held-out source is never used for training, validation, early stopping, hyperparameter selection, threshold selection, or probability calibration. Unlabelled challenge images are excluded rather than assigned guessed labels. Fold-specific duplicate candidates matching the held-out source are removed from training. Missing segmentation annotations remain valid classification samples but do not receive fabricated mask or vCDR targets.

## Validation status

GitHub Actions compiles the clean V4 runner and executes an end-to-end synthetic three-source experiment covering model training, segmentation, graph fusion, calibration, checkpoint save/load, and mirrored artifacts. This validates the software path, not the medical performance. Actual publication readiness depends on the resulting external metrics, segmentation quality, confidence intervals, ablations, and independent external validation.
