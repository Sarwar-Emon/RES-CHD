"""
RES-CHD Phase 1 — Data Preprocessing Pipeline (v2 — Corrected)
===============================================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Dataset : nhanes_2005_2018_unified.csv
Outputs :
  - data/preprocessed/nhanes_base.csv            (cleaned, unscaled — cycle preserved)
  - data/preprocessed/scarcity_summary.csv        (per-level statistics)
  - data/preprocessed/metadata.json               (pipeline config for downstream scripts)
  - data/scarcity_levels/S{1..7}_train.csv        (train split, scaler fitted on this)
  - data/scarcity_levels/S{1..7}_test.csv         (test split, scaler applied — no leakage)
  - data/scarcity_levels/S{1..7}_scaler.pkl       (per-level fitted StandardScaler)
  - logs/preprocessing_report.txt

Key fixes vs v1:
  1. Scaler is fit per-level on train set only  → eliminates cross-level leakage
  2. Null handling added (median impute continuous, mode impute binary)
  3. Cycle column explicitly preserved through all steps via df_base
  4. y alignment made explicit via shared index reset on X_train / y_train
  5. Winsorization clipping bounds saved per-feature in metadata (reproducibility)

SMOTE is still deferred to training time inside model pipeline — no change there.

Requirements:
    pip install pandas numpy scikit-learn imbalanced-learn joblib
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
PREPROCESSED = DATA_DIR / "preprocessed"
SCARCITY_DIR = DATA_DIR / "scarcity_levels"
LOG_DIR      = BASE_DIR / "logs"

for d in [PREPROCESSED, SCARCITY_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocessing_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_CSV             = BASE_DIR / "nhanes_2005_2018_unified.csv"
TARGET              = "chd"
CYCLE_COL           = "cycle"
ID_COLS             = ["SEQN"]          # identifiers — drop early
BINARY_FEATURES     = ["sex", "smoking"]
CONTINUOUS_FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
ALL_FEATURES        = CONTINUOUS_FEATURES + BINARY_FEATURES
WINSORIZE_CLIP      = (0.01, 0.99)      # 1st–99th percentile

# Scarcity levels: cumulative newest → oldest
# S1 = all 7 cycles (~32K)  |  S7 = most-recent cycle only (~4.3K)
CYCLE_ORDER = [
    "2017_2018",
    "2015_2016",
    "2013_2014",
    "2011_2012",
    "2009_2010",
    "2007_2008",
    "2005_2006",
]

TEST_SIZE   = 0.20
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: Path) -> pd.DataFrame:
    log.info("=" * 65)
    log.info("STEP 1 — Loading raw data")
    log.info("=" * 65)

    df = pd.read_csv(path)
    log.info(f"Shape             : {df.shape}")
    log.info(f"Columns           : {df.columns.tolist()}")
    log.info(f"CHD positive      : {int(df[TARGET].sum())}  "
             f"({df[TARGET].mean()*100:.2f}%)")
    log.info(f"CHD negative      : {int((df[TARGET] == 0).sum())}")
    log.info(f"Cycles present    : {sorted(df[CYCLE_COL].unique().tolist())}")

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if null_cols.empty:
        log.info("Missing values    : none detected")
    else:
        log.info(f"Missing values    :\n{null_cols.to_string()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Drop identifier columns
# ══════════════════════════════════════════════════════════════════════════════
def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\n" + "=" * 65)
    log.info("STEP 2 — Dropping identifier columns")
    log.info("=" * 65)

    before = df.columns.tolist()
    df = df.drop(columns=ID_COLS, errors="ignore")
    dropped = [c for c in before if c not in df.columns]
    log.info(f"Dropped           : {dropped}")
    log.info(f"Shape after       : {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Handle missing values
#
#   Strategy:
#     - Continuous : median imputation  (robust to outliers)
#     - Binary     : mode imputation    (preserves most-frequent class)
#     - Target/cycle: assert no nulls   (data integrity check)
# ══════════════════════════════════════════════════════════════════════════════
def handle_nulls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log.info("\n" + "=" * 65)
    log.info("STEP 3 — Handling missing values")
    log.info("=" * 65)

    df = df.copy()
    impute_values: dict = {}

    # Target and cycle must be complete
    for col in [TARGET, CYCLE_COL]:
        n_null = df[col].isnull().sum()
        assert n_null == 0, (
            f"Column '{col}' has {n_null} missing values — cannot proceed. "
            f"Remove or manually inspect these rows."
        )

    # Continuous — median
    for col in CONTINUOUS_FEATURES:
        n_null = df[col].isnull().sum()
        median = df[col].median()
        impute_values[col] = {"strategy": "median", "value": float(median)}
        if n_null > 0:
            df[col] = df[col].fillna(median)
            log.info(f"  {col:<12}  imputed {n_null:>4} nulls  "
                     f"→ median={median:.4f}")
        else:
            log.info(f"  {col:<12}  no nulls")

    # Binary — mode
    for col in BINARY_FEATURES:
        n_null = df[col].isnull().sum()
        mode   = df[col].mode()[0]
        impute_values[col] = {"strategy": "mode", "value": float(mode)}
        if n_null > 0:
            df[col] = df[col].fillna(mode)
            log.info(f"  {col:<12}  imputed {n_null:>4} nulls  "
                     f"→ mode={mode:.0f}")
        else:
            log.info(f"  {col:<12}  no nulls")

    remaining = df[ALL_FEATURES].isnull().sum().sum()
    assert remaining == 0, f"Unexpected nulls remain after imputation: {remaining}"
    log.info(f"\n  Post-imputation nulls in features: {remaining}  ✓")
    return df, impute_values


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Winsorize continuous features (clip outliers)
# ══════════════════════════════════════════════════════════════════════════════
def winsorize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log.info("\n" + "=" * 65)
    log.info("STEP 4 — Winsorizing continuous features (1st–99th pct)")
    log.info("=" * 65)

    df = df.copy()
    clip_bounds: dict = {}

    for col in CONTINUOUS_FEATURES:
        lo = df[col].quantile(WINSORIZE_CLIP[0])
        hi = df[col].quantile(WINSORIZE_CLIP[1])
        n_clipped = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)
        clip_bounds[col] = {"low": float(lo), "high": float(hi)}
        log.info(f"  {col:<12}  clipped {n_clipped:>4} rows  "
                 f"→ [{lo:.4f}, {hi:.4f}]")

    return df, clip_bounds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Verify binary feature integrity
# ══════════════════════════════════════════════════════════════════════════════
def verify_binary(df: pd.DataFrame) -> pd.DataFrame:
    log.info("\n" + "=" * 65)
    log.info("STEP 5 — Verifying binary features (expect 0/1 only)")
    log.info("=" * 65)

    for col in BINARY_FEATURES:
        vals = sorted(df[col].dropna().unique().tolist())
        log.info(f"  {col:<10}  unique values: {vals}")
        assert set(vals).issubset({0.0, 1.0, 0, 1}), (
            f"Unexpected values in '{col}': {vals}"
        )
    log.info("  All binary features confirmed as 0 / 1  ✓")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Build progressive scarcity splits (S1–S7)
#          with per-split scaler fitted on TRAIN only
#
#  Fix vs v1:
#   - Scaler fit inside this function, per split, on train rows only
#   - cycle column sourced from df_base (preserved explicitly)
#   - y alignment is explicit: both X and y reset_index before concat
# ══════════════════════════════════════════════════════════════════════════════
def build_scarcity_splits(df_base: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    df_base : cleaned, winsorized, NOT yet scaled dataframe.
              Must still contain CYCLE_COL.

    For each level Si:
      1. Slice rows whose cycle is in CYCLE_ORDER[:n_cycles]
      2. Split 80/20 stratified on TARGET
      3. Fit StandardScaler on X_train only
      4. Apply scaler to X_train and X_test
      5. Store everything; save CSVs + scaler pkl
    """
    log.info("\n" + "=" * 65)
    log.info("STEP 6 — Building scarcity splits S1–S7 (per-split scaling)")
    log.info("=" * 65)

    splits: dict      = {}
    summary_rows: list = []

    for level in range(1, 8):
        n_cycles        = 8 - level                         # S1→7, S7→1
        selected_cycles = CYCLE_ORDER[:n_cycles]
        label           = f"S{level}"

        # ── Slice rows for this level ──────────────────────────────────────
        mask   = df_base[CYCLE_COL].isin(selected_cycles)
        subset = df_base[mask].copy()

        X = subset[ALL_FEATURES].reset_index(drop=True)
        y = subset[TARGET].reset_index(drop=True)

        # ── Stratified train / test split ──────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size   = TEST_SIZE,
            stratify    = y,
            random_state= RANDOM_SEED,
        )
        # Reset indices so positional alignment is guaranteed
        X_train = X_train.reset_index(drop=True)
        X_test  = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)

        # ── Fit scaler on TRAIN ONLY — no leakage ─────────────────────────
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled  = X_test.copy()

        X_train_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(
            X_train[CONTINUOUS_FEATURES]
        )
        X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(   # ← transform only
            X_test[CONTINUOUS_FEATURES]
        )

        # ── Store split ────────────────────────────────────────────────────
        splits[label] = {
            "X_train":        X_train_scaled,
            "X_test":         X_test_scaled,
            "y_train":        y_train,
            "y_test":         y_test,
            "scaler":         scaler,
            "cycles":         selected_cycles,
            "n_total":        len(subset),
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "pos_train":      int(y_train.sum()),
            "neg_train":      int((y_train == 0).sum()),
            "chd_rate_train": float(y_train.mean()),
            "chd_rate_test":  float(y_test.mean()),
        }

        log.info(
            f"\n  {label}  cycles={n_cycles}  total={len(subset):>6}  "
            f"train={len(X_train):>5}  test={len(X_test):>4}  "
            f"CHD_train={y_train.mean()*100:.2f}%  "
            f"CHD_test={y_test.mean()*100:.2f}%"
        )
        log.info(f"       cycles : {selected_cycles}")
        log.info(
            f"       scaler : mean(age)_train={scaler.mean_[0]:.3f}  "
            f"std(age)_train={np.sqrt(scaler.var_[0]):.3f}  "
            f"[fitted on train only ✓]"
        )

        summary_rows.append({
            "level":      label,
            "n_cycles":   n_cycles,
            "cycles":     ", ".join(selected_cycles),
            "total":      len(subset),
            "train":      len(X_train),
            "test":       len(X_test),
            "pos_train":  int(y_train.sum()),
            "neg_train":  int((y_train == 0).sum()),
            "chd_rate_%": round(y_train.mean() * 100, 2),
        })

    return splits, pd.DataFrame(summary_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save all outputs
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(
    df_base:     pd.DataFrame,
    splits:      dict,
    summary:     pd.DataFrame,
    impute_vals: dict,
    clip_bounds: dict,
) -> None:
    log.info("\n" + "=" * 65)
    log.info("STEP 7 — Saving outputs")
    log.info("=" * 65)

    # ── Base cleaned dataset (unscaled — cycle preserved for reference) ──
    base_path = PREPROCESSED / "nhanes_base.csv"
    df_base.to_csv(base_path, index=False)
    log.info(f"  Saved base CSV            : {base_path}")

    # ── Per-level: train CSV, test CSV, scaler pkl ───────────────────────
    for name, split in splits.items():
        train_df         = split["X_train"].copy()
        train_df[TARGET] = split["y_train"].values   # safe: both reset_index'd

        test_df          = split["X_test"].copy()
        test_df[TARGET]  = split["y_test"].values

        train_path = SCARCITY_DIR / f"{name}_train.csv"
        test_path  = SCARCITY_DIR / f"{name}_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        scaler_path = SCARCITY_DIR / f"{name}_scaler.pkl"
        joblib.dump(split["scaler"], scaler_path)

        log.info(
            f"  {name}  train={len(train_df):>5}  test={len(test_df):>4}  "
            f"→ {train_path.name}  {test_path.name}  {scaler_path.name}"
        )

    # ── Scarcity summary table ────────────────────────────────────────────
    summary_path = PREPROCESSED / "scarcity_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"  Saved scarcity summary    : {summary_path}")

    # ── Metadata JSON ─────────────────────────────────────────────────────
    meta = {
        "pipeline_version":     "2.0",
        "continuous_features":  CONTINUOUS_FEATURES,
        "binary_features":      BINARY_FEATURES,
        "target":               TARGET,
        "id_cols_dropped":      ID_COLS,
        "winsorize_clip":       list(WINSORIZE_CLIP),
        "winsorize_bounds":     clip_bounds,
        "imputation":           impute_vals,
        "scaler":               "StandardScaler (per-split, fit on train only)",
        "test_size":            TEST_SIZE,
        "random_seed":          RANDOM_SEED,
        "cycle_order":          CYCLE_ORDER,
        "scarcity_levels": {
            name: {
                "n_cycles":      len(split["cycles"]),
                "cycles":        split["cycles"],
                "n_total":       split["n_total"],
                "n_train":       split["n_train"],
                "n_test":        split["n_test"],
                "pos_train":     split["pos_train"],
                "neg_train":     split["neg_train"],
                "chd_rate_train_pct": round(split["chd_rate_train"] * 100, 2),
                "chd_rate_test_pct":  round(split["chd_rate_test"]  * 100, 2),
            }
            for name, split in splits.items()
        },
    }
    meta_path = PREPROCESSED / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Saved metadata JSON       : {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 1 — Preprocessing Pipeline  v2 (corrected)")
    log.info("=" * 65)

    df                   = load_data(RAW_CSV)
    df                   = drop_identifiers(df)
    df, impute_vals      = handle_nulls(df)
    df, clip_bounds      = winsorize(df)
    df                   = verify_binary(df)

    # df_base is clean + winsorized but NOT scaled.
    # Scaling happens per-split inside build_scarcity_splits.
    df_base = df.copy()

    splits, summary = build_scarcity_splits(df_base)
    save_outputs(df_base, splits, summary, impute_vals, clip_bounds)

    log.info("\n" + "=" * 65)
    log.info("Phase 1 complete — scarcity level summary:")
    log.info("=" * 65)
    log.info("\n" + summary.to_string(index=False))
    log.info("\nAll outputs saved.")


if __name__ == "__main__":
    main()