# Glaucomma — RimGraph-DG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/Glaucomma/blob/main/Glaucomma_RimGraphDG_Single_Cell.ipynb)

A reproducible Google Colab research pipeline for source-independent glaucoma screening using ORIGA, REFUGE, and G1020.

## Main design

- One executable Colab cell with organized live output
- Leave-one-dataset-out external validation
- ConvNeXt-based global/local representation
- Optic-disc and optic-cup multi-task segmentation
- 12-sector polar rim graph attention
- vCDR consistency learning
- class-conditional cross-domain prototype alignment
- source-adversarial learning
- optional Optuna tuning using source validation data only
- worst-source temperature scaling
- visual XAI, clock-sector attention, calibration and error analysis
- automatic resume after interruption

## Run in Colab

Click the **Open in Colab** badge above, select a GPU runtime, and run the notebook's only cell.

The cell automatically:

1. mounts Google Drive;
2. finds or downloads the Kaggle dataset `arnavjain1/glaucoma-datasets`;
3. audits labels, images and masks;
4. runs ORIGA, REFUGE and G1020 leave-one-source-out folds;
5. displays live epoch tables and figures;
6. saves checkpoints, predictions, metrics, figures and manifests to both Colab and Drive.

## Output locations

- Colab: `/content/Glaucomma_runs/<run_id>/`
- Drive: `/content/drive/MyDrive/Glaucomma_RimGraphDG/<run_id>/`

Each fold saves:

- `best_model.pt`
- `last_checkpoint.pt`
- `history.csv`
- `test_predictions.csv`
- `metrics.json`
- calibration selection results
- ROC, PR, confusion-matrix and calibration figures
- segmentation/XAI examples
- sector-attention summary

The complete run also saves `fold_summary.csv`, `aggregate_metrics.csv`, `all_external_predictions.csv`, an SHA-256 artifact manifest and a ZIP bundle.

## Important configuration

At the top of the runner:

- set `fast_dev_run=True` only for a smoke test;
- keep `fast_dev_run=False` for research experiments;
- set `run_optuna=True` for source-only hyperparameter tuning;
- optionally set `manual_data_dir` when the dataset is already stored in Drive.

## Reproducibility and leakage control

The held-out source is not used for fitting, early stopping, hyperparameter selection, threshold selection or temperature scaling. All split CSVs, effective hyperparameters, checkpoints, environment details and checksums are retained.

The notebook is pinned to an immutable runner commit and verifies the reconstructed runner with SHA-256 before execution. GitHub Actions also checks that the runner compiles and that the notebook contains exactly one code cell.

## Status

This is an initial research implementation. Syntax and notebook structure are validated, but full GPU training results must be produced by running the notebook in Colab with the dataset available.