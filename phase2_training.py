"""
RES-CHD Phase 2 — Model Training Pipeline (v2 — Corrected)
===========================================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Inputs  (from Phase 1 v2 outputs):
  - data/scarcity_levels/S{1..7}_train.csv
  - data/scarcity_levels/S{1..7}_test.csv
  - data/scarcity_levels/S{1..7}_scaler.pkl   (loaded for metadata only — data already scaled)

Outputs:
  - models/S{1..7}_{ModelName}.pkl            (trained model per level per model)
  - results/metrics_per_level.csv             (AUC, PR-AUC, F1, Precision, Recall per row)
  - results/metrics_summary.csv               (mean ± std across levels per model)
  - logs/training_report.txt

Key fixes vs v1:
  1. Loads S{n}_train.csv / S{n}_test.csv  — matches Phase 1 v2 output format
  2. SMOTE applied consistently to ALL models — no double-compensation with scale_pos_weight
  3. Threshold tuning removed — PR-AUC is the primary metric for imbalanced data (4% CHD)
     Fixed threshold of 0.5 reported alongside; defensible in thesis methods section
  4. Scaler PKL loaded and referenced in metadata for full reproducibility audit
  5. Per-level class balance logged so imbalance trend S1→S7 is visible in logs
  6. Model hyperparameters moved to a single CONFIG dict for easy ablation

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib
"""

import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data" / "scarcity_levels"
MODELS_DIR   = BASE_DIR / "models"
RESULTS_DIR  = BASE_DIR / "results"
LOG_DIR      = BASE_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET          = "chd"
RANDOM_SEED     = 42
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]
THRESHOLD       = 0.5    # fixed — PR-AUC is primary metric for imbalanced data

# ── Hyperparameter config (single place for ablation study) ───────────────────
MODEL_CONFIG = {
    "XGBoost": dict(
        n_estimators    = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        eval_metric     = "auc",
        use_label_encoder=False,
        random_state    = RANDOM_SEED,
        n_jobs          = -1,
        # NOTE: no scale_pos_weight — SMOTE handles imbalance uniformly
    ),
    "RandomForest": dict(
        n_estimators = 300,
        max_depth    = None,
        random_state = RANDOM_SEED,
        n_jobs       = -1,
        # NOTE: no class_weight="balanced" — SMOTE handles imbalance uniformly
    ),
    "LogisticRegression": dict(
        max_iter     = 1000,
        solver       = "lbfgs",
        random_state = RANDOM_SEED,
        # NOTE: no class_weight="balanced" — SMOTE handles imbalance uniformly
    ),
    "MLP": dict(
        hidden_layer_sizes = (64, 32),   # reduced: thesis focus is SHAP, not MLP tuning
        activation         = "relu",
        max_iter           = 200,        # early_stopping will cut this further
        early_stopping     = True,
        validation_fraction= 0.1,        # internal val for early stopping only
        n_iter_no_change   = 10,         # stop if no improvement for 10 rounds
        random_state       = RANDOM_SEED,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_split(level: str) -> tuple:
    """
    Load pre-scaled train/test CSVs produced by Phase 1 v2.
    Scaler PKL is loaded for metadata audit only — data is already scaled.
    """
    train_path  = DATA_DIR / f"{level}_train.csv"
    test_path   = DATA_DIR / f"{level}_test.csv"
    scaler_path = DATA_DIR / f"{level}_scaler.pkl"

    assert train_path.exists(), f"Missing: {train_path}"
    assert test_path.exists(),  f"Missing: {test_path}"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    return X_train, X_test, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# SMOTE  (applied to ALL models — consistent imbalance handling)
# ══════════════════════════════════════════════════════════════════════════════
def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    SMOTE on training set only.

    Sampling strategy: cap minority at 30% of majority class.
      - Avoids exploding training set at large scarcity levels (S1/S2)
      - e.g. S1: majority=24658 → minority capped at 7397 (not 24658)
      - Keeps meaningful class balance without 50K+ row training sets
      - k_neighbors auto-reduced for very small minority classes (S6/S7)
    """
    n_minority = int(y_train.sum())
    n_majority = int((y_train == 0).sum())
    k = min(5, n_minority - 1)

    if k < 1:
        log.warning("  SMOTE skipped — too few minority samples. "
                    "Using original training data.")
        return X_train.values, y_train.values

    # Cap minority at 30% of majority → balanced but not oversized
    target_minority = int(n_majority * 0.30)
    target_minority = max(target_minority, n_minority)  # never shrink existing minority

    sampling_strategy = target_minority / n_majority

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k,
        random_state=RANDOM_SEED,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    log.info(f"  SMOTE  before: pos={n_minority}, neg={n_majority}  "
             f"after: pos={int(y_res.sum())}, neg={int((y_res==0).sum())}  "
             f"total={len(y_res)}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def build_model(name: str):
    cfg = MODEL_CONFIG[name].copy()
    if name == "XGBoost":
        return XGBClassifier(**cfg)
    elif name == "RandomForest":
        return RandomForestClassifier(**cfg)
    elif name == "LogisticRegression":
        return LogisticRegression(**cfg)
    elif name == "MLP":
        return MLPClassifier(**cfg)
    raise ValueError(f"Unknown model: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, level: str, name: str) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    auc    = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)   # primary metric
    f1     = f1_score(y_test, y_pred, zero_division=0)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    log.info(
        f"    {name:<20}  AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
        f"F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  Spec={specificity:.4f}"
    )

    return {
        "level":       level,
        "model":       name,
        "auc":         round(auc,         4),
        "pr_auc":      round(pr_auc,      4),   # primary for imbalanced data
        "f1":          round(f1,          4),
        "precision":   round(prec,        4),
        "recall":      round(rec,         4),
        "specificity": round(specificity, 4),
        "threshold":   THRESHOLD,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 2 — Model Training Pipeline  v2 (corrected)")
    log.info("=" * 65)

    all_results: list = []

    for level in SCARCITY_LEVELS:
        log.info(f"\n{'='*65}")
        log.info(f"  Scarcity level : {level}")
        log.info(f"{'='*65}")

        X_train, X_test, y_train, y_test, scaler = load_split(level)

        n_pos   = int(y_train.sum())
        n_neg   = int((y_train == 0).sum())
        imb_ratio = n_neg / n_pos if n_pos > 0 else float("inf")

        log.info(f"  Train size     : {len(X_train):>6}  "
                 f"(pos={n_pos}, neg={n_neg}, ratio=1:{imb_ratio:.1f})")
        log.info(f"  Test size      : {len(X_test):>6}  "
                 f"(pos={int(y_test.sum())}, neg={int((y_test==0).sum())})")
        log.info(f"  Scaler loaded  : {'yes' if scaler else 'not found'}")

        # SMOTE on train
        X_tr_sm, y_tr_sm = apply_smote(X_train, y_train)

        for name in MODEL_CONFIG:
            log.info(f"\n  ── {name} ──")
            model = build_model(name)
            model.fit(X_tr_sm, y_tr_sm)

            metrics = evaluate(model, X_test, y_test, level, name)
            all_results.append(metrics)

            # Save model
            model_path = MODELS_DIR / f"{level}_{name}.pkl"
            joblib.dump(model, model_path)
            log.info(f"    Saved → {model_path.name}")

    # ── Per-level results CSV ────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "metrics_per_level.csv"
    df_results.to_csv(results_path, index=False)
    log.info(f"\n  Saved per-level metrics  : {results_path}")

    # ── Summary: mean ± std per model across S1–S7 ──────────────────────────
    summary_rows = []
    for model_name, grp in df_results.groupby("model"):
        for metric in ["auc", "pr_auc", "f1", "precision", "recall"]:
            summary_rows.append({
                "model":  model_name,
                "metric": metric,
                "mean":   round(grp[metric].mean(), 4),
                "std":    round(grp[metric].std(),  4),
                "min":    round(grp[metric].min(),  4),
                "max":    round(grp[metric].max(),  4),
            })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "metrics_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    log.info(f"  Saved summary metrics    : {summary_path}")

    # ── Final print ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("Phase 2 complete — AUC by model and scarcity level:")
    log.info("=" * 65)
    pivot = df_results.pivot(index="level", columns="model", values="auc")
    log.info("\n" + pivot.to_string())

    log.info("\nPR-AUC by model and scarcity level (primary metric):")
    pivot_pr = df_results.pivot(index="level", columns="model", values="pr_auc")
    log.info("\n" + pivot_pr.to_string())

    log.info("\nAll models saved. Ready for Phase 3 (SHAP extraction).  ✓")


if __name__ == "__main__":
    main()