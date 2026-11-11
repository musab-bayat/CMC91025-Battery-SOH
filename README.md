# CMC-91025 — final manuscript analysis code

**Multi-Domain Feature Engineering for Interpretable Cross-Battery Lithium-Ion Battery State-of-Health Prediction**

This repository contains only the MATLAB analysis scripts needed for the **final manuscript's reported methods and results**. It does not contain preliminary experiments, alternative development scripts, reviewer-specific side analyses, or figure restyling utilities.

## Data

Obtain the original NASA PCoE Li-ion Battery Aging datasets for **B0005, B0006, B0007, and B0018** from the NASA PCoE Data Set Repository. Place `B0005.mat`, `B0006.mat`, `B0007.mat`, and `B0018.mat` in this directory. NASA data are not included in this code package.

NASA source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## Requirements

MATLAB with **Statistics and Machine Learning Toolbox** (tree ensembles, SVR, `shapley`) and **Deep Learning Toolbox** (LSTM). 

## Run

From this directory in MATLAB:

```matlab
reproduce_CMC91025
```

The runner executes these steps in sequence. Individual stages can also be run separately after their prerequisites are created.

| Order | Script | Final-manuscript role |
| --- | --- | --- |
| 1 | `run_preprocessing_stage1.m` + `build_battery_dataset.m` | NASA discharge-cycle extraction, common-cutoff preprocessing and nine-feature inputs; produces `NASA_corrected_features.mat` |
| 2 | `run_feature_screening_stage2.m` | Feature screening, variability and correlation checks |
| 3 | `run_lstm_stage3c_nested.m` | Nested battery-held-out LSTM tuning and evaluation; produces `stage3c_selected_configs.csv` |
| 4 | `run_stage4_energy_ablation_benchmarks_MACRO.m` | Original Stage 4 benchmark and LSTM evaluation; baseline classical-model results (not the final tuned classical rows in Table 2) |
| 5 | `run_stage4_nested_tuned_classical.m` | Nested three-fold battery-level tuning of Bagged Trees, LSBoost, and SVR; **final Table 2 classical-model results** in `r2c5_nested_overall_macro.csv` |
| 6 | `run_persistence_baseline_stage4.m` | One-cycle-ahead persistence reference for Table 2 |
| 7 | `run_stage5_rf_importance_energy_ablation_MACRO.m` | Expanded-training Bagged Trees, EnergyWh ablation (Table 4), held-out-battery grouped-lag permutation importance (Figure 7) |
| 8 | `run_stage6_rf_shap_MACRO.m` | Battery-macro SHAP interpretation of expanded-training Bagged Trees (Figure 8) |
| 9 | `run_multihorizon_evaluation.m` | Fixed-configuration 1-, 5-, 10-, and 20-cycle prediction-horizon sensitivity (Table 3) |
| 10 | `run_joint_feature_ablation.m` | Expanded-training Bagged Trees with all features, without EnergyWh, and without both EnergyWh and Vslope (Table 4) |

All original scientific computation scripts are preserved **without modification**, including the added original reviewer-revision nested-tuning script; `reproduce_CMC91025.m` is a new convenience launcher. Original stage filenames are retained for auditability. Outputs are written to the current working directory (the repository directory when using the launcher).

## Interpretation and evaluation

Each of the four NASA cells is held out as the test battery in turn. LSTM model selection uses development batteries only. The original Stage 4 benchmark uses the same selected temporal window across model families. The subsequent nested-tuned classical stage uses the same LSTM-selected window and deterministic outer train/validation/test split, and supplies the revised manuscript Table 2 classical-model rows. The Table 2 LSTM row comes from Stage 4; the persistence row comes from the persistence stage. The post-benchmark Bagged Trees analysis fits on all three non-test batteries; permutation importance and SHAP are aggregated at the held-out battery level. The persistence reference assumes observed SOH from the immediately preceding cycle.

## Reproducibility scope

The scripts implement preprocessing, feature extraction, battery-held-out model evaluation, prediction-horizon sensitivity, feature ablation, permutation importance and SHAP analysis. The original NASA data must be obtained separately. Figure-generation and figure-formatting scripts are not distributed.

The multi-horizon sensitivity script applies the same eligible input-window endpoints across horizons, restricting each endpoint to have at least 20 subsequent cycles. Both supplementary analysis scripts export battery-level and macro-averaged CSV results for comparison with manuscript tables.

## Table 2 provenance

Use `r2c5_nested_overall_macro.csv` for Bagged Trees, LSBoost and SVR; use the original Stage 4 benchmark output for LSTM and the persistence stage output for the persistence baseline. Do **not** use the untuned classical-model rows in `stage4_benchmark_*` as the revised Table 2 classical results. The nested script writes `r2c5_nested_inner_scores.csv`, `r2c5_nested_selected.csv`, `r2c5_nested_outer_runs.csv`, `r2c5_nested_predictions.csv`, `r2c5_nested_by_battery.csv`, and `r2c5_nested_overall_macro.csv`. The nested hyperparameter grids and model fitting procedures are preserved in the original script.

Reference values from the previously recorded original nested run (RMSE macro mean): Bagged Trees 0.021473, LSBoost 0.023858, SVR 0.065888. These are reference checks, not a claim that the packaged runner has been executed or independently verified in this environment.
