# RES-CHD: A Reliability-Aware Explainability Stability Framework
### for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

> **MS Thesis — Troy University | Department of Computer Science | Spring 2026**
> **Author:** Sayem Sarwar | **Supervisor:** Dr. Arteta

---

## Overview

Most machine learning papers for clinical heart disease prediction focus on **accuracy** — but accuracy alone does not tell clinicians whether the AI's *explanations* can be trusted. When a hospital has limited patient data, SHAP-based explanations can become unreliable in ways that standard metrics like AUC cannot detect.

**RES-CHD** is a reliability-aware framework that measures, quantifies, and flags explanation stability under progressive data scarcity. It answers a question no prior CHD framework has addressed:

> *"When training data is limited, can we still trust the AI's explanation for why it flagged this patient as high risk?"*

---

## Key Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Explanation Stability Index (ESI)** | Novel bootstrapped metric with 95% CI quantifying SHAP explanation consistency across data levels |
| 2 | **Temporal Scarcity Simulation** | 7 scarcity levels using real NHANES temporal cycles — 5.3–6.4× harder than random subsampling |
| 3 | **Three-Dimensional Reliability Framework** | Combines predictive accuracy (AUC), explanation stability (ESI), and calibration (ECE) |
| 4 | **Local Patient-Level Stability** | Per-patient Local Stability Score (LSS) — high-risk patients receive least stable explanations |
| 5 | **Compound Risk Framework** | Identifies patients with both uncertain predictions AND unstable explanations simultaneously |
| 6 | **External Validation** | ESI rankings perfectly preserved on UCI Cleveland dataset (Spearman ρ=1.00, p<0.001) |

---

## Results Summary

| Model | AUC@S1 | ESI (Bootstrapped) | 95% CI | Reliability | Compound Risk@S7 |
|---|---|---|---|---|---|
| Random Forest | 0.836 | 0.9517 | [0.937, 0.966] | ✅ HIGH | 24.9% |
| Logistic Regression | 0.859 | 0.9402 | [0.933, 0.948] | ✅ HIGH | 18.2% |
| XGBoost | 0.849 | 0.8598 | [0.837, 0.888] | ⚠️ HIGH* | 42.2% |
| MLP (ANN) | 0.822 | 0.6576 | [0.520, 0.762] | ❌ LOW | 31.5% |

**Key finding:** All four models have similar AUC — but ESI reveals dramatic differences in explanation reliability that AUC cannot detect. All 6 pairwise ESI differences are statistically significant (Mann-Whitney U, p<0.001).

---

## Dataset

- **Primary:** NHANES 2005–2018 — 32,118 patients, 7 biennial cycles, 4.03% CHD prevalence
- **External Validation:** UCI Cleveland Heart Disease Dataset — 297 patients
- **Features:** Age, Sex, Systolic BP, Diastolic BP, HDL Cholesterol, Total Cholesterol, BMI, Smoking

> **Note:** NHANES data is publicly available from the CDC. Download `nhanes_2005_2018_unified.csv` and place it in the project root before running.

---

## Pipeline Architecture

```
Phase 1  →  Data Preprocessing & Scarcity Simulation (7 levels S1–S7)
Phase 2  →  Model Training (28 models: 4 architectures × 7 levels)
Phase 3  →  SHAP Extraction & Global ESI Computation
Phase 4  →  RES-CHD Framework (Bootstrapped ESI, Reliability Flags)
Phase 4B →  MLP Bootstrap (SHAP-array method, B=200)
Phase 5  →  Ablation Study (5 experiments)
Phase 5B →  Distributional Shift Analysis (KL, Wasserstein, PSI, KS)
Phase 5C →  ESI Baseline Comparison (vs Jaccard, MARC, RV coefficient)
Phase 5D →  Local SHAP Stability (per-patient LSS)
Phase 5E →  Compound Risk Framework (confidence × explanation stability)
Phase 5F →  Calibration Analysis (Brier score, ECE, CSI)
Phase 6  →  Publication Figures (18 figures, PNG + PDF)
Cleveland→  External Validation on UCI Cleveland Dataset
```

---

## Quickstart

### Requirements

```bash
pip install pandas numpy scipy scikit-learn xgboost imbalanced-learn shap matplotlib joblib
```

Python 3.12 recommended. Pin numpy to 1.26.4 for compatibility:

```bash
pip install numpy==1.26.4
```

### Run the full pipeline

```bash
python run_all.py
```

Reproduces all results in approximately **40 minutes**. All outputs saved automatically to structured directories.

### Run a specific phase

```bash
python run_all.py --phase 3        # SHAP extraction only
python run_all.py --from-phase 5b  # start from distributional shift
python run_all.py --check-only     # verify all outputs exist
```

---

## Project Structure

```
RES-CHD/
├── phase1_preprocessing.py       # Data cleaning & scarcity splits
├── phase2_training.py            # Model training (28 models)
├── phase3_shap.py                # SHAP extraction & ESI computation
├── phase4_framework.py           # RES-CHD reliability framework
├── phase4b_mlp_bootstrap.py      # MLP bootstrapped ESI
├── phase5_ablation.py            # Ablation study
├── phase5b_distribution_shift.py # Distributional shift analysis
├── phase5c_esi_baselines.py      # ESI vs alternative metrics
├── phase5d_local_stability.py    # Per-patient local stability
├── phase5e_confidence_stability.py # Compound risk framework
├── phase5f_calibration.py        # Calibration analysis
├── phase6_visualization.py       # Publication figures (Fig 1-10)
├── phase6b_master_figures.py     # Updated figures (Fig 11-18)
├── cleveland_validation.py       # External validation
├── build_master_table.py         # Master results table
├── final_improvements.py         # Statistical significance tests
├── run_all.py                    # Master reproducibility script
└── .gitignore
```

---

## Reproducibility

This pipeline is fully reproducible. Fixed random seed (`42`) is used throughout all phases. Running `python run_all.py` from scratch produces identical results to those reported in the thesis.

```
✓ 14/14 phases complete
✓ 43/43 expected outputs verified
✓ Total runtime: ~40 minutes
```

---

## Novel Findings

1. **Temporal data scarcity** introduces 5.3–6.4× more distributional drift than random subsampling — validating ecological realism of the simulation
2. **ESI detects instability 5 levels earlier than Jaccard Top-3** for MLP (at S2 vs S7)
3. **High-risk patients** (older, higher SBP) receive the least stable explanations (p<0.001)
4. **Compound risk** grows from 10–30% at full data to 18–42% at minimum data — invisible to AUC or ESI alone
5. **Calibration remains stable** while ESI degrades — models keep predicting correctly but stop explaining correctly
6. **ESI rankings preserved** on UCI Cleveland (ρ=1.00, p<0.001) — framework generalizes across datasets

---

## Citation

```bibtex
@mastersthesis{sarwar2026reschd,
  author    = {Sayem Sarwar},
  title     = {RES-CHD: A Reliability-Aware Explainability Stability Framework
               for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity},
  school    = {Troy University},
  year      = {2026},
  type      = {M.S. Thesis},
  address   = {Troy, Alabama}
}
```

---

## License

This project is part of an ongoing MS thesis. Code is available for academic and research purposes. Please cite appropriately if you use this work.

---

*Department of Computer Science · Troy University · Troy, Alabama · 2026*
