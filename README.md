# RES-CHD: stability of SHAP explanations for CHD prediction under data scarcity

MS thesis, Troy University, 2026. Author: Sayem Sarwar. Supervisor: Dr. Arteta.

## What it studies
Four models (logistic regression, random forest, XGBoost, MLP) are trained on NHANES 2005-2018 (32,118 adults, 4.03% self-reported coronary heart disease, CHD) with fewer and fewer survey cycles (levels S1 to S7; S7 uses only 2017-2018). The **Explanation Stability Index (ESI)** is the mean Spearman correlation between the SHAP feature ranking at S1 and at each later level (S2 to S7).

## Main findings
- The four models have similar AUC at S1 (0.82 to 0.86) but different ESI. Primary run: RF 0.952, LR 0.940, XGBoost 0.860, MLP 0.658.
- Over repeated independent splits the MLP had the lowest ESI every time (10 splits, mean 0.72); RF and LR were similar (about 0.91).
- ESI depends on training choices. For example, RF's ESI ranged from 0.75 to 0.99 across four imbalance methods while its AUC changed by less than 0.01.
- Removing older cycles caused 5.3 to 6.4 times more feature drift than random subsampling of the same size, but ESI was not lower.
- The models detect CHD poorly (best F1 about 0.31 at 4% prevalence). This is a study of explanation stability, not a clinical tool.

Limits: self-reported labels, only eight features, one survey, and a descriptive external check (UCI Cleveland: 297 patients, four models).

## Data
- NHANES 2005-2018 public-use files from the CDC. `download_nhanes.py` downloads them, `Data_Set_Preparing.ipynb` builds the cleaned table, and `nhanes_2005_2018_unified.csv` (included; 32,118 rows) is the input of the pipeline.
- `python dataset_funnel.py raw_nhanes` reproduces the cohort steps (70,190 participants to 32,118). `python null_report.py raw_nhanes` reports the records removed for missing values.
- UCI Cleveland Heart Disease data (297 complete records) for the external check; `cleveland_validation.py` downloads it.

## Setup
Python 3.12. `requirements_experiments.txt` lists the package versions recorded when the experiments ran, and `requirements_figures.txt` lists the versions used to regenerate the figures. A fresh install from these files has not been tested, and some packages (for example SHAP and NumPy) have strict version limits, so results may differ in the last digit with other versions.

## Run
1. Main pipeline (phases 1 to 6): `python run_all.py` (several hours; Phase 3 explains the MLP with KernelExplainer).
2. Additional experiments used in the paper and thesis: see the table below.
3. Figures: `python make_fig_esi_trajectory.py`, `python make_fig_drift.py`, `python make_fig1_paper.py`.

## Random seeds
All scripts use seed 42, except the repeated-split experiment (`split_robustness.py`), which uses seeds 100 to 109 (RF, MLP) and 100 to 104 (LR, XGBoost). Each scarcity level has its own stratified 80/20 split.

## Which script makes which number
| Item | Script | Output |
|---|---|---|
| Cohort numbers | `dataset_funnel.py`, `null_report.py` | `dataset_funnel.csv` |
| Scarcity levels (training and test sizes) | `phase1_preprocessing.py` | `data/scarcity_levels/`, `data/preprocessed/scarcity_summary.csv` |
| Primary models, ESI, per-level correlations | `phase2_training.py` to `phase4_framework.py` | `results/phase5/final_consolidated_results.csv`, `results/phase5f/calibration_stability.csv` |
| Calibration (ECE, Brier) | `phase5f_calibration.py` | `results/phase5f/calibration_metrics.csv` |
| Distribution drift | `phase5b_distribution_shift.py` | `results/phase5b/*.csv` |
| Ablations (thresholds, Kendall, features) | `phase5_ablation.py` | `results/phase5/ablation_*.csv` |
| Repeated splits | `split_robustness.py` | `outputs/sr_*/` |
| Noise floor (retraining agreement) | `noise_floor.py` | `outputs/noise_floor/` |
| MLP variants | `mlp_sensitivity.py`, `split_robustness.py` | `outputs/mlp_sensitivity/`, `outputs/sr_mlp_*/` |
| Exact SHAP check for the MLP | `mlp_kernelshap_check.py` | `outputs/mlp_check/` |
| Alternative metrics, bootstrap convergence, SMOTE caps | `claims_experiments.py` (`--task core`, `--task smote`) | `outputs/claims/` |
| Matched random subsampling | `random_vs_temporal_esi.py` | `outputs/random_vs_temporal/` |
| Patient-level stability, compound risk | `lss_aligned.py` | `outputs/lss_aligned/` |
| Imbalance methods, cutoffs | `imbalance_experiments.py` | `outputs/imbalance_v2/` |
| Confusion matrices by method, tuning | `defense_experiments.py` | `outputs/defense/` |
| ESI of tuned models, top-5 features | `defense_esi_tuned.py`, `defense_topk.py` | `outputs/defense/esi_tuned.csv`, `esi_topk.csv` |
| 5-fold cross-validation | `cv_s1.py` | `outputs/cv/` |
| External check (Cleveland) | `cleveland_validation.py` | `results/cleveland/`, `results/final/cleveland_esi_with_ci.csv` |
| Figures | `make_fig_esi_trajectory.py`, `make_fig_drift.py`, `make_fig1_paper.py` | `outputs/figures/` |

Commands for the repeated splits:
```
python split_robustness.py --seeds 5 --models XGBoost LogisticRegression --n-eval 300 --outdir outputs/sr_fast300
python split_robustness.py --seeds 10 --models RandomForest --n-eval 300 --outdir outputs/sr_rf10
python split_robustness.py --seeds 10 --models MLP --outdir outputs/sr_mlp10
```

`phase5d_local_stability.py` and `phase5e_confidence_stability.py` are earlier versions of the patient-level analysis; the reported patient-level numbers come from `lss_aligned.py`.

## Reference outputs
`reference_outputs/` holds copies of the result files behind the reported numbers (same folder layout as the scripts write), so a number can be checked without rerunning the experiments.

## Notes
- In the main run, the MLP's SHAP values use KernelExplainer with 100 sampled coalitions on 200 patients (an approximation). The repeated-split and fixed-patient experiments use 300 patients with the exact computation.
- Hyperparameters were fixed, not tuned, in the main analysis (tuning is a separate experiment in `defense_experiments.py`).
- The HIGH (0.85) and LOW (0.70) ESI bands are descriptive, not validated clinical limits.
