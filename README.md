# CMC-91025 — final manuscript analysis code

**Multi-Domain Feature Engineering for Interpretable Cross-Battery Lithium-Ion Battery State-of-Health Prediction**

This repository contains only the MATLAB analysis scripts needed for the **final manuscript's reported methods and results**. It does not contain preliminary experiments, alternative development scripts, reviewer-specific side analyses, or figure restyling utilities.

## Data

Obtain the original NASA PCoE Li-ion Battery Aging datasets for **B0005, B0006, B0007, and B0018** from the NASA PCoE Data Set Repository. Place `B0005.mat`, `B0006.mat`, `B0007.mat`, and `B0018.mat` in this directory. NASA data are not included in this code package.

NASA source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## Requirements

MATLAB with **Statistics and Machine Learning Toolbox** (tree ensembles, SVR, `shapley`) and **Deep Learning Toolbox** (LSTM). MATLAB version compatibility has **not yet been tested**; the analysis has not been executed in this preparation environment.

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
| 4 | `run_stage4_energy_ablation_benchmarks_MACRO.m` | Common battery-held-out comparison: Bagged Trees, LSBoost, SVR, LSTM; battery-macro results for Table 2 |
| 5 | `run_persistence_baseline_stage4.m` | One-cycle-ahead persistence reference for Table 2 |
| 6 | `run_stage5_rf_importance_energy_ablation_MACRO.m` | Expanded-training Bagged Trees, EnergyWh ablation (Table 3), held-out-battery grouped-lag permutation importance (Figure 7) |
| 7 | `run_stage6_rf_shap_MACRO.m` | Battery-macro SHAP interpretation of expanded-training Bagged Trees (Figure 8) |

All original scientific computation scripts are preserved **without modification**; `reproduce_CMC91025.m` is a new convenience launcher. Original stage filenames are retained for auditability. Outputs are written to the current working directory (the repository directory when using the launcher).

## Interpretation and evaluation

Each of the four NASA cells is held out as the test battery in turn. LSTM model selection uses development batteries only. The common model benchmark uses the same selected temporal window across model families. The post-benchmark Bagged Trees analysis fits on all three non-test batteries; permutation importance and SHAP are aggregated at the held-out battery level. The persistence reference assumes observed SOH from the immediately preceding cycle.

## Verification status

Source-level dependencies and file names have been inspected. **This package has not been executed end-to-end in MATLAB**, and matching of newly generated outputs to the final manuscript's numeric tables remains a required verification step before describing it as fully reproduced. Do not claim code execution or numerical reproduction until that check is complete.
