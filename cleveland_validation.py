"""
RES-CHD External Validation — UCI Cleveland Heart Disease Dataset
=================================================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Purpose:
  Validate the RES-CHD framework on an independent dataset to demonstrate
  that findings generalize beyond NHANES. This closes the single-dataset
  limitation and significantly strengthens the publication case.

  Key question:
  Does the model ranking on ESI (RF > LR > XGB > MLP) hold on
  a completely different CHD dataset from a different country,
  different time period, and different clinical setting?

  If yes → RES-CHD generalizes. Your framework is dataset-independent.
  If rankings differ → also interesting, discuss why in limitations.

Dataset:
  UCI Heart Disease Dataset — Cleveland Clinic Foundation
  303 patients, 13 features, binary heart disease label (0/1)
  Source: https://archive.ics.uci.edu/dataset/45/heart+disease
  Download: processed.cleveland.data from UCI repository

  The Cleveland dataset is the most widely used heart disease
  benchmark in ML literature. Every reviewer will recognize it.

Feature mapping to NHANES:
  Cleveland → NHANES equivalent
  age       → age (direct match)
  sex       → sex (direct match, 1=male 0=female)
  trestbps  → sbp (resting blood pressure, direct match)
  chol      → total_chol (serum cholesterol, direct match)
  thalach   → no direct match (max heart rate) — dropped
  oldpeak   → no direct match (ST depression) — dropped
  ca        → no direct match (vessels) — dropped
  thal      → no direct match (thalassemia) — dropped
  cp        → no direct match (chest pain type) — keep as clinical feature
  fbs       → no direct match (fasting blood sugar) — keep as clinical feature
  restecg   → no direct match (resting ECG) — keep as clinical feature
  exang     → no direct match (exercise angina) — keep as clinical feature
  slope     → no direct match (slope ST segment) — keep as clinical feature

  We use ALL 13 Cleveland features for the validation — not just the
  NHANES subset. This is correct because we are validating the FRAMEWORK
  (ESI metric, stability methodology) not the specific feature set.

Scarcity simulation:
  Cleveland has only 303 patients — too few for temporal cycle splits.
  We use random subsampling at 5 levels:
  L1: 100% (303 pts) — full data baseline
  L2:  80% (242 pts)
  L3:  60% (182 pts)
  L4:  40% (121 pts)
  L5:  20%  (61 pts) — extreme scarcity

  We acknowledge this uses random subsampling (not temporal cycles)
  and note this likely underestimates true scarcity effects per Phase 5B.

Outputs:
  - data/cleveland/cleveland_processed.csv
  - results/cleveland/metrics.csv           (AUC, PR-AUC per model per level)
  - results/cleveland/esi_scores.csv        (ESI per model)
  - results/cleveland/rank_stability.csv    (Spearman rho per level)
  - results/cleveland/comparison_table.csv  (NHANES vs Cleveland ESI)
  - results/cleveland/figures/fig_cleveland_esi.png
  - results/cleveland/figures/fig_cleveland_vs_nhanes.png
  - logs/cleveland_validation.txt

HOW TO GET THE DATA:
  Option 1 (recommended): Download from UCI
    URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
    Save as: data/cleveland/processed.cleveland.data

  Option 2: The script will attempt to download it automatically
    using urllib if the file is not found locally.

Requirements:
    pip install pandas numpy scipy scikit-learn xgboost imbalanced-learn shap matplotlib joblib
"""

import logging
import warnings
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils import resample as sk_resample
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CLE_DIR    = BASE_DIR / "data" / "cleveland"
RES_DIR    = BASE_DIR / "results" / "cleveland"
FIG_DIR    = RES_DIR / "figures"
LOG_DIR    = BASE_DIR / "logs"

for d in [CLE_DIR, RES_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cleveland_validation.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
TARGET       = "hd"   # heart disease binary (0/1)
TEST_SIZE    = 0.2

# Cleveland feature names
CLEVELAND_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

# Features used for modeling (drop problematic ones with many missing)
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope"
]

# Scarcity levels — random subsampling fractions
SCARCITY_LEVELS = ["L1", "L2", "L3", "L4", "L5"]
SCARCITY_FRACS  = [1.00, 0.80, 0.60, 0.40, 0.20]

MODEL_NAMES = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]

COLORS = {
    "XGBoost":            "#27AE60",
    "RandomForest":       "#C0392B",
    "LogisticRegression": "#2980B9",
    "MLP":                "#8E44AD",
}

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND PREPROCESS CLEVELAND DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_cleveland_data() -> pd.DataFrame:
    """
    Load Cleveland Heart Disease data.
    Tries local file first, then downloads from UCI.
    """
    local_path = CLE_DIR / "processed.cleveland.data"

    if not local_path.exists():
        log.info(f"  Local file not found. Downloading from UCI ...")
        try:
            urllib.request.urlretrieve(DATA_URL, local_path)
            log.info(f"  Downloaded to {local_path}")
        except Exception as e:
            log.error(
                f"  Download failed: {e}\n"
                f"  Please manually download from:\n"
                f"  {DATA_URL}\n"
                f"  And save to: {local_path}"
            )
            raise

    df = pd.read_csv(local_path, header=None, names=CLEVELAND_COLS,
                     na_values="?")
    log.info(f"  Loaded Cleveland data: {df.shape}")
    log.info(f"  Missing values: {df.isnull().sum().sum()}")
    log.info(f"  Target distribution:\n{df['target'].value_counts().to_string()}")

    return df


def preprocess_cleveland(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess Cleveland data.
    - Binarize target (0 = no HD, 1 = any HD)
    - Drop rows with missing values (only 6 rows have missing ca/thal)
    - Scale features
    """
    # Binarize target: original has values 0-4, we use 0 vs 1+
    df = df.copy()
    df[TARGET] = (df["target"] > 0).astype(int)
    df = df.drop("target", axis=1)

    # Drop rows with missing values
    df_clean = df.dropna().reset_index(drop=True)
    log.info(f"  After dropping missing: {df_clean.shape}")
    log.info(
        f"  HD prevalence: "
        f"{df_clean[TARGET].mean()*100:.1f}% "
        f"({df_clean[TARGET].sum()}/{len(df_clean)})"
    )

    # Keep only features we use
    feature_cols = [f for f in FEATURES if f in df_clean.columns]
    df_final     = df_clean[feature_cols + [TARGET]]

    return df_final


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SIMULATE SCARCITY LEVELS
# ══════════════════════════════════════════════════════════════════════════════
def create_scarcity_splits(df: pd.DataFrame) -> dict:
    """
    Create scarcity levels by random subsampling.
    L1 = full data, L5 = 20% of data.
    Each level uses a stratified train/test split.
    """
    log.info(f"\n  Creating {len(SCARCITY_LEVELS)} scarcity levels ...")
    splits = {}
    np.random.seed(RANDOM_SEED)

    # L1 = full dataset split
    X_full = df[FEATURES].values
    y_full = df[TARGET].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_full
    )

    for level, frac in zip(SCARCITY_LEVELS, SCARCITY_FRACS):
        if frac == 1.0:
            X_train = X_train_full
            y_train = y_train_full
        else:
            # Subsample training set with stratification
            n_sample = max(int(len(X_train_full) * frac), 20)
            idx      = sk_resample(
                np.arange(len(X_train_full)),
                n_samples=n_sample,
                random_state=RANDOM_SEED,
                stratify=y_train_full
            )
            X_train = X_train_full[idx]
            y_train = y_train_full[idx]

        # Scale per split (fit on train, apply to test — no leakage)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        splits[level] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test":  X_test_scaled,
            "y_test":  y_test,
            "n_train": len(X_train),
            "frac":    frac,
        }

        log.info(
            f"  {level} ({int(frac*100)}%)  "
            f"train={len(X_train)}  "
            f"test={len(X_test)}  "
            f"HD_train={y_train.mean()*100:.1f}%"
        )

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
def build_model(model_name: str):
    """Build model with same hyperparameters as NHANES pipeline."""
    if model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, verbosity=0
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_SEED,
            class_weight="balanced"
        )
    elif model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300,
            random_state=RANDOM_SEED, early_stopping=True
        )


def apply_smote(X_train: np.ndarray,
                y_train: np.ndarray) -> tuple:
    """Apply SMOTE if minority class has enough samples."""
    minority_count = int(y_train.sum())
    if minority_count < 6:
        log.warning(f"  Too few minority samples ({minority_count}) for SMOTE — skipping")
        return X_train, y_train
    try:
        k = min(5, minority_count - 1)
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
        return sm.fit_resample(X_train, y_train)
    except Exception as e:
        log.warning(f"  SMOTE failed: {e} — using original data")
        return X_train, y_train


def train_all_models(splits: dict) -> dict:
    """Train all models across all scarcity levels."""
    log.info(f"\n{'='*65}")
    log.info("  Training models across scarcity levels")
    log.info(f"{'='*65}")

    trained = {}

    for level in SCARCITY_LEVELS:
        X_train = splits[level]["X_train"]
        y_train = splits[level]["y_train"]
        X_test  = splits[level]["X_test"]
        y_test  = splits[level]["y_test"]

        for model_name in MODEL_NAMES:
            log.info(f"\n  {level}  {model_name}  n_train={len(X_train)}")

            # Apply SMOTE
            X_res, y_res = apply_smote(X_train, y_train)

            # Build and train
            model = build_model(model_name)
            model.fit(X_res, y_res)

            # Evaluate
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = model.predict(X_test).astype(float)

            try:
                auc    = roc_auc_score(y_test, y_prob)
                prauc  = average_precision_score(y_test, y_prob)
            except Exception:
                auc = prauc = np.nan

            log.info(f"  AUC={auc:.4f}  PR-AUC={prauc:.4f}")

            trained[(level, model_name)] = {
                "model":  model,
                "y_prob": y_prob,
                "y_test": y_test,
                "auc":    auc,
                "prauc":  prauc,
            }

    return trained


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE SHAP AND ESI
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap_ranks(model, model_name: str,
                       X_train: np.ndarray,
                       X_test:  np.ndarray) -> np.ndarray:
    """Compute global SHAP importance ranks for Cleveland validation."""
    feat_names = FEATURES

    if model_name in ("XGBoost", "RandomForest"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

    elif model_name == "LogisticRegression":
        bg        = X_train.mean(axis=0).reshape(1, -1)
        explainer = shap.LinearExplainer(model, bg)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

    elif model_name == "MLP":
        np.random.seed(RANDOM_SEED)
        bg_idx   = np.random.choice(len(X_train),
                                     size=min(50, len(X_train)), replace=False)
        bg       = X_train[bg_idx]
        eval_idx = np.random.choice(len(X_test),
                                     size=min(50, len(X_test)), replace=False)
        X_eval   = X_test[eval_idx]

        def predict_fn(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_vals = explainer.shap_values(X_eval, nsamples=50, silent=True)
        X_test    = X_eval

    shap_arr   = np.array(shap_vals)
    global_imp = np.abs(shap_arr).mean(axis=0)
    order      = np.argsort(global_imp)[::-1]
    ranks      = np.empty_like(order)
    ranks[order] = np.arange(1, len(global_imp) + 1)
    return ranks


def compute_esi(trained: dict, splits: dict) -> tuple:
    """
    Compute ESI for each model across Cleveland scarcity levels.
    Returns ESI scores and per-level stability DataFrame.
    """
    log.info(f"\n{'='*65}")
    log.info("  Computing SHAP ranks and ESI")
    log.info(f"{'='*65}")

    all_ranks    = {}
    stability    = []
    esi_scores   = []

    for model_name in MODEL_NAMES:
        log.info(f"\n  {model_name}:")

        # Compute ranks at each level
        for level in SCARCITY_LEVELS:
            model   = trained[(level, model_name)]["model"]
            X_train = splits[level]["X_train"]
            X_test  = splits[level]["X_test"]

            try:
                ranks = compute_shap_ranks(model, model_name, X_train, X_test)
                all_ranks[(level, model_name)] = ranks
                log.info(
                    f"  {level}  ranks: "
                    f"{dict(zip(FEATURES[:5], ranks[:5]))}"
                )
            except Exception as e:
                log.warning(f"  SHAP failed {level} {model_name}: {e}")

        # Compute ESI — Spearman rho S1 vs L2-L5
        l1_ranks  = all_ranks.get(("L1", model_name))
        if l1_ranks is None:
            continue

        rho_vals = []
        for level in SCARCITY_LEVELS[1:]:
            li_ranks = all_ranks.get((level, model_name))
            if li_ranks is None:
                continue
            rho, pval = spearmanr(l1_ranks, li_ranks)
            rho_vals.append(float(rho))

            stability.append({
                "model":        model_name,
                "baseline":     "L1",
                "comparison":   level,
                "spearman_rho": round(float(rho),  4),
                "p_value":      round(float(pval),  6),
            })

            log.info(
                f"  {level} vs L1: rho={rho:.4f}  p={pval:.4f}"
            )

        esi = float(np.mean(rho_vals)) if rho_vals else np.nan
        log.info(f"  ESI = {esi:.4f}")

        esi_scores.append({
            "model": model_name,
            "esi":   round(esi, 4),
        })

    return pd.DataFrame(stability), pd.DataFrame(esi_scores)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — COMPARE WITH NHANES RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def build_comparison_table(esi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Cleveland ESI with NHANES ESI.
    Checks whether model rankings are preserved.
    """
    # NHANES ESI from Phase 3
    nhanes_esi = {
        "RandomForest":       0.9524,
        "LogisticRegression": 0.9365,
        "XGBoost":            0.8571,
        "MLP":                0.6587,
    }

    rows = []
    for _, row in esi_df.iterrows():
        model      = row["model"]
        clev_esi   = row["esi"]
        nhanes_val = nhanes_esi.get(model, np.nan)
        delta      = clev_esi - nhanes_val if not np.isnan(nhanes_val) else np.nan

        rows.append({
            "model":         model,
            "nhanes_esi":    nhanes_val,
            "cleveland_esi": clev_esi,
            "delta":         round(float(delta), 4) if not np.isnan(delta) else None,
            "ranking_preserved": None,  # filled below
        })

    comp_df = pd.DataFrame(rows)

    # Check if rankings are preserved
    nhanes_ranking   = comp_df.sort_values("nhanes_esi",   ascending=False)["model"].tolist()
    cleveland_ranking = comp_df.sort_values("cleveland_esi", ascending=False)["model"].tolist()

    log.info(f"\n  NHANES ranking   : {nhanes_ranking}")
    log.info(f"  Cleveland ranking: {cleveland_ranking}")
    log.info(f"  Rankings identical: {nhanes_ranking == cleveland_ranking}")

    # Spearman rank correlation between NHANES and Cleveland ESI
    n_esi = [nhanes_esi.get(m, np.nan) for m in comp_df["model"]]
    c_esi = comp_df["cleveland_esi"].tolist()
    valid = [(n, c) for n, c in zip(n_esi, c_esi)
             if not (np.isnan(n) or np.isnan(c))]
    if len(valid) >= 3:
        rho, p = spearmanr([v[0] for v in valid], [v[1] for v in valid])
        log.info(f"  ESI rank correlation (NHANES vs Cleveland): rho={rho:.4f}  p={p:.4f}")

    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


def plot_cleveland_esi(stab_df: pd.DataFrame,
                       esi_df:  pd.DataFrame) -> None:
    """ESI stability curves for Cleveland dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: ESI curves
    ax    = axes[0]
    levels = SCARCITY_LEVELS[1:]   # L2-L5

    for model_name in MODEL_NAMES:
        m_df = stab_df[stab_df["model"] == model_name]
        rhos = [float(m_df[m_df["comparison"]==l]["spearman_rho"].values[0])
                if len(m_df[m_df["comparison"]==l]) > 0 else np.nan
                for l in levels]
        ax.plot(range(len(levels)), rhos,
               color=COLORS[model_name], linewidth=2,
               marker="o", markersize=6, label=model_name)

    ax.axhline(0.85, color="gray", linestyle="--",
              linewidth=1, alpha=0.6, label="ESI=0.85 threshold")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(
        [f"{l}\n({int(f*100)}%)" for l, f in
         zip(levels, SCARCITY_FRACS[1:])], fontsize=9
    )
    ax.set_ylabel("Spearman ρ vs L1 baseline", fontsize=10)
    ax.set_title("Cleveland — ESI across scarcity levels", fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=9)

    # Right: Overall ESI comparison
    ax2     = axes[1]
    models  = esi_df["model"].tolist()
    esi_vals = esi_df["esi"].tolist()
    colors  = [COLORS[m] for m in models]

    bars = ax2.bar(range(len(models)), esi_vals,
                  color=colors, alpha=0.82,
                  edgecolor="white", linewidth=0.5)

    ax2.axhline(0.85, color="gray", linestyle="--",
               linewidth=1, alpha=0.6)

    for bar, val in zip(bars, esi_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(
        [m.replace("Logistic", "Log.") for m in models],
        fontsize=9
    )
    ax2.set_ylabel("ESI (mean Spearman ρ)", fontsize=10)
    ax2.set_title("Cleveland — Overall ESI per model", fontsize=11)
    ax2.set_ylim(0, 1.1)

    fig.suptitle(
        "Fig — External Validation on UCI Cleveland Heart Disease Dataset\n"
        "ESI stability assessment on independent dataset (n=303, 5 scarcity levels)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_cleveland_esi.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_cleveland_esi.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_cleveland_esi.png / .pdf")


def plot_nhanes_vs_cleveland(comp_df: pd.DataFrame) -> None:
    """Side-by-side ESI comparison NHANES vs Cleveland."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = comp_df["model"].tolist()
    x      = np.arange(len(models))
    w      = 0.35

    nhanes_vals   = comp_df["nhanes_esi"].tolist()
    clev_vals     = comp_df["cleveland_esi"].tolist()

    bars1 = ax.bar(x - w/2, nhanes_vals, w,
                  label="NHANES (n=32,118)",
                  color=[COLORS[m] for m in models],
                  alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, clev_vals, w,
                  label="Cleveland (n=303)",
                  color=[COLORS[m] for m in models],
                  alpha=0.45, edgecolor=[COLORS[m] for m in models],
                  linewidth=1.5, hatch="///")

    ax.axhline(0.85, color="gray", linestyle="--",
              linewidth=1, alpha=0.6, label="HIGH threshold (ESI=0.85)")

    # Value labels
    for bar, val in zip(bars1, nhanes_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.01,
               f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, clev_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.01,
               f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("Logistic", "Log.").replace("Regression", "Reg.")
         for m in models], fontsize=9
    )
    ax.set_ylabel("ESI (Explanation Stability Index)", fontsize=10)
    ax.set_title(
        "Fig — ESI Comparison: NHANES vs Cleveland\n"
        "Model ranking preservation across independent datasets",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    path = FIG_DIR / "fig_cleveland_vs_nhanes.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_cleveland_vs_nhanes.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_cleveland_vs_nhanes.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  External Validation — UCI Cleveland Heart Disease")
    log.info("=" * 65)
    log.info(f"Dataset  : Cleveland Clinic Foundation (n~303)")
    log.info(f"Levels   : {SCARCITY_LEVELS} ({[int(f*100) for f in SCARCITY_FRACS]}%)")
    log.info(f"Models   : {MODEL_NAMES}")
    log.info(f"Scarcity : Random subsampling (temporal cycles unavailable)")

    # ── Step 1: Load and preprocess ───────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Step 1 — Loading and preprocessing Cleveland data")
    log.info(f"{'='*65}")
    df_raw  = load_cleveland_data()
    df      = preprocess_cleveland(df_raw)
    df.to_csv(CLE_DIR / "cleveland_processed.csv", index=False)
    log.info(f"  Saved cleveland_processed.csv")

    # ── Step 2: Create scarcity splits ────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Step 2 — Creating scarcity levels")
    log.info(f"{'='*65}")
    splits = create_scarcity_splits(df)

    # ── Step 3: Train models ──────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Step 3 — Training models")
    log.info(f"{'='*65}")
    trained = train_all_models(splits)

    # Save metrics
    metrics_rows = []
    for (level, model_name), data in trained.items():
        metrics_rows.append({
            "level":   level,
            "model":   model_name,
            "n_train": splits[level]["n_train"],
            "auc":     round(data["auc"],   4),
            "pr_auc":  round(data["prauc"], 4),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RES_DIR / "metrics.csv", index=False)

    # ── Step 4: SHAP and ESI ──────────────────────────────────────────────
    stab_df, esi_df = compute_esi(trained, splits)
    stab_df.to_csv(RES_DIR / "rank_stability.csv", index=False)
    esi_df.to_csv(RES_DIR / "esi_scores.csv", index=False)

    # ── Step 5: Comparison table ──────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Step 5 — NHANES vs Cleveland comparison")
    log.info(f"{'='*65}")
    comp_df = build_comparison_table(esi_df)
    comp_df.to_csv(RES_DIR / "comparison_table.csv", index=False)

    # ── Visualizations ────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Generating figures ...")
    log.info(f"{'='*65}")
    plot_cleveland_esi(stab_df, esi_df)
    plot_nhanes_vs_cleveland(comp_df)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("Cleveland Validation complete — Key findings:")
    log.info(f"{'='*65}")

    log.info("\n  Cleveland ESI results:")
    log.info(f"\n{esi_df.sort_values('esi', ascending=False).to_string(index=False)}")

    log.info("\n  NHANES vs Cleveland comparison:")
    log.info(f"\n{comp_df[['model','nhanes_esi','cleveland_esi','delta']].to_string(index=False)}")

    log.info("\n  Predictive performance on Cleveland (AUC at L1):")
    l1_metrics = metrics_df[metrics_df["level"] == "L1"]
    log.info(f"\n{l1_metrics[['model','auc','pr_auc']].to_string(index=False)}")

    log.info(
        "\n  GENERALIZABILITY CLAIM:"
        "\n  If model rankings are preserved on Cleveland, write:"
        "\n  'ESI rankings (RF > LR > XGB > MLP) were preserved on the"
        "\n   independent Cleveland dataset, confirming that RES-CHD"
        "\n   findings generalize beyond the NHANES population.'"
        "\n"
        "\n  If rankings differ, write:"
        "\n  'ESI rankings differed on Cleveland, suggesting that"
        "\n   explanation stability may be dataset-dependent, warranting"
        "\n   future investigation across diverse clinical populations.'"
    )

    log.info("\nExternal validation complete. ✓")


if __name__ == "__main__":
    main()