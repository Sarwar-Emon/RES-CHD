"""
RES-CHD Phase 3 — SHAP Extraction Pipeline
===========================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Inputs  (from Phase 1 & 2 outputs):
  - data/scarcity_levels/S{1..7}_test.csv        (scaled test sets)
  - models/S{1..7}_{ModelName}.pkl               (trained models)

Outputs:
  - shap/global/S{1..7}_{Model}_global_shap.csv  (mean |SHAP| per feature — global importance)
  - shap/local/S{1..7}_{Model}_local_shap.csv    (per-patient SHAP values — local attribution)
  - shap/ranks/S{1..7}_{Model}_ranks.csv         (feature importance rank per level)
  - shap/stability/rank_stability.csv             (Spearman rank correlation vs S1 baseline)
  - shap/stability/esi_scores.csv                 (Explanation Stability Index per model/feature)
  - logs/shap_report.txt

What is computed:
  Global SHAP  : mean(|shap_value|) per feature per level → shows which features matter overall
  Local SHAP   : raw shap matrix (n_test × n_features) per level → per-patient attribution
  Feature ranks: rank of each feature by global importance (rank 1 = most important)
  Rank stability: Spearman ρ between feature ranks at Si vs S1 baseline
  ESI score    : mean Spearman ρ across S2–S7 vs S1 — core Phase 5 metric

Model-specific SHAP explainer selection:
  XGBoost      → shap.TreeExplainer      (exact, fast)
  RandomForest → shap.TreeExplainer      (exact, fast)
  LogisticReg  → shap.LinearExplainer    (exact, uses training background)
  MLP          → shap.KernelExplainer    (model-agnostic, slow — uses sampled background)

Requirements:
    pip install shap pandas numpy scipy joblib
"""

import json
import joblib
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import shap

warnings.filterwarnings("ignore")
shap.initjs()   # safe to call even outside notebooks

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data" / "scarcity_levels"
MODELS_DIR    = BASE_DIR / "models"
SHAP_DIR      = BASE_DIR / "shap"
LOG_DIR       = BASE_DIR / "logs"

for d in [
    SHAP_DIR / "global",
    SHAP_DIR / "local",
    SHAP_DIR / "ranks",
    SHAP_DIR / "stability",
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "shap_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET          = "chd"
RANDOM_SEED     = 42
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]
MODEL_NAMES     = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
FEATURES        = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]

# KernelExplainer background sample size (MLP only — larger = more accurate but slower)
KERNEL_BG_SAMPLES  = 100
# KernelExplainer test evaluation sample (MLP only — full test set is too slow)
KERNEL_EVAL_SAMPLES = 200


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_test_set(level: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{level}_test.csv"
    assert path.exists(), f"Missing test CSV: {path}"
    df = pd.read_csv(path)
    X  = df[FEATURES]
    y  = df[TARGET]
    return X, y


def load_train_set(level: str) -> pd.DataFrame:
    """Training set needed as background for LinearExplainer and KernelExplainer."""
    path = DATA_DIR / f"{level}_train.csv"
    assert path.exists(), f"Missing train CSV: {path}"
    df = pd.read_csv(path)
    return df[FEATURES]


def load_model(level: str, model_name: str):
    path = MODELS_DIR / f"{level}_{model_name}.pkl"
    assert path.exists(), f"Missing model: {path}"
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINER FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def build_explainer(model, model_name: str,
                    X_train: pd.DataFrame,
                    X_test:  pd.DataFrame):
    """
    Returns (explainer, shap_values_array, X_evaluated)

    X_evaluated may be a subset of X_test for slow explainers (MLP KernelExplainer).
    shap_values_array shape: (n_samples, n_features)
    For binary classifiers we take SHAP values for class 1 (CHD positive).
    """
    if model_name in ("XGBoost", "RandomForest"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)

        # SHAP >= 0.45 returns different shapes depending on model:
        #   RandomForest -> 3D array (n_samples, n_features, n_classes) OR list [cls0, cls1]
        #   XGBoost      -> 2D array (n_samples, n_features) for binary
        # Always extract class-1 slice for CHD positive
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]
        # else: already 2D (XGBoost binary) — use as-is

        return explainer, shap_vals, X_test

    elif model_name == "LogisticRegression":
        # Use training data mean as background (standard for LinearExplainer)
        bg          = X_train.mean(axis=0).values.reshape(1, -1)
        explainer   = shap.LinearExplainer(model, bg)
        shap_vals   = explainer.shap_values(X_test)

        # LinearExplainer may return list for binary — take class 1 if so
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        return explainer, shap_vals, X_test

    elif model_name == "MLP":
        # KernelExplainer — model-agnostic but slow
        # Use sampled background + sampled evaluation set
        np.random.seed(RANDOM_SEED)

        bg_idx  = np.random.choice(len(X_train), size=min(KERNEL_BG_SAMPLES, len(X_train)), replace=False)
        bg      = X_train.iloc[bg_idx]

        eval_idx = np.random.choice(len(X_test), size=min(KERNEL_EVAL_SAMPLES, len(X_test)), replace=False)
        X_eval   = X_test.iloc[eval_idx].reset_index(drop=True)

        # Wrap predict_proba to return P(CHD=1) only
        def predict_fn(X):
            return model.predict_proba(X)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_vals = explainer.shap_values(X_eval, nsamples=100, silent=True)

        return explainer, shap_vals, X_eval

    raise ValueError(f"Unknown model: {model_name}")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP COMPUTATION PER LEVEL × MODEL
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap(level: str, model_name: str) -> dict:
    """
    Returns dict with:
      global_importance : pd.Series  — mean |SHAP| per feature
      local_df          : pd.DataFrame — raw SHAP matrix (n_samples × n_features)
      ranks             : pd.Series  — feature rank (1 = most important)
    """
    log.info(f"  Computing SHAP  {level} / {model_name} ...")

    X_test,  y_test  = load_test_set(level)
    X_train          = load_train_set(level)
    model            = load_model(level, model_name)

    _, shap_vals, X_eval = build_explainer(model, model_name, X_train, X_test)

    # Ensure shap_vals is a 2D numpy array (n_samples × n_features)
    shap_arr = np.array(shap_vals)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)

    # ── Global importance: mean absolute SHAP ─────────────────────────────
    global_imp = pd.Series(
        np.abs(shap_arr).mean(axis=0),
        index=FEATURES,
        name=f"{level}_{model_name}",
    )

    # ── Feature rank (1 = highest importance) ─────────────────────────────
    ranks = global_imp.rank(ascending=False, method="min").astype(int)
    ranks.name = f"{level}_{model_name}"

    # ── Local SHAP dataframe ───────────────────────────────────────────────
    local_df = pd.DataFrame(shap_arr, columns=FEATURES)
    local_df.insert(0, "level",  level)
    local_df.insert(1, "model",  model_name)

    log.info(
        f"    Done  n_eval={len(X_eval)}  "
        f"top feature: {global_imp.idxmax()} ({global_imp.max():.4f})"
    )

    return {
        "global_importance": global_imp,
        "local_df":          local_df,
        "ranks":             ranks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STABILITY ANALYSIS — SPEARMAN RANK CORRELATION vs S1 BASELINE
# ══════════════════════════════════════════════════════════════════════════════
def compute_stability(all_ranks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    all_ranks : {(level, model_name): pd.Series of feature ranks}

    Returns:
      stability_df : rows = (model, level_pair), cols = spearman_rho, p_value
      esi_df       : rows = model, ESI = mean Spearman ρ across S2–S7 vs S1
    """
    log.info("\n  Computing rank stability (Spearman ρ vs S1 baseline) ...")

    stability_rows = []
    esi_rows       = []

    for model_name in MODEL_NAMES:
        baseline_ranks = all_ranks.get(("S1", model_name))
        if baseline_ranks is None:
            log.warning(f"  No S1 ranks for {model_name} — skipping stability")
            continue

        rho_values = []

        for level in SCARCITY_LEVELS[1:]:    # S2 … S7 vs S1
            level_ranks = all_ranks.get((level, model_name))
            if level_ranks is None:
                continue

            rho, pval = spearmanr(baseline_ranks.values, level_ranks.values)

            stability_rows.append({
                "model":        model_name,
                "baseline":     "S1",
                "comparison":   level,
                "spearman_rho": round(float(rho),  4),
                "p_value":      round(float(pval), 6),
                "significant":  pval < 0.05,   # flag non-significant comparisons
            })
            rho_values.append(float(rho))

            sig = "" if pval < 0.05 else "  [NOT SIGNIFICANT — interpret with caution]"
            log.info(f"    {model_name}  S1 vs {level}  ρ={rho:.4f}  p={pval:.4f}{sig}")

        # ESI = mean Spearman ρ across all S2–S7 comparisons
        esi = float(np.mean(rho_values)) if rho_values else float("nan")
        esi_rows.append({
            "model":     model_name,
            "esi_score": round(esi, 4),
            "n_levels":  len(rho_values),
            "interpretation": (
                "high stability"   if esi >= 0.85 else
                "moderate stability" if esi >= 0.70 else
                "low stability"
            ),
        })
        log.info(f"    {model_name}  ESI = {esi:.4f}")

    stability_df = pd.DataFrame(stability_rows)
    esi_df       = pd.DataFrame(esi_rows).sort_values("esi_score", ascending=False)

    return stability_df, esi_df


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(
    all_global:    dict,
    all_local:     list,
    all_ranks:     dict,
    stability_df:  pd.DataFrame,
    esi_df:        pd.DataFrame,
) -> None:
    log.info("\n  Saving all SHAP outputs ...")

    # ── Global importance CSVs ────────────────────────────────────────────
    for (level, model_name), series in all_global.items():
        path = SHAP_DIR / "global" / f"{level}_{model_name}_global_shap.csv"
        series.to_csv(path, header=True)

    # ── Local SHAP CSVs ───────────────────────────────────────────────────
    for local_df in all_local:
        level      = local_df["level"].iloc[0]
        model_name = local_df["model"].iloc[0]
        path = SHAP_DIR / "local" / f"{level}_{model_name}_local_shap.csv"
        local_df.to_csv(path, index=False)

    # ── Rank CSVs ─────────────────────────────────────────────────────────
    for (level, model_name), ranks in all_ranks.items():
        path = SHAP_DIR / "ranks" / f"{level}_{model_name}_ranks.csv"
        ranks.to_csv(path, header=True)

    # ── Combined global importance table (all levels × models) ────────────
    global_combined = pd.DataFrame(
        {f"{lvl}_{mdl}": s for (lvl, mdl), s in all_global.items()}
    ).T
    global_combined.index.name = "level_model"
    global_combined.to_csv(SHAP_DIR / "global" / "all_global_shap.csv")
    log.info(f"  Saved combined global SHAP  : shap/global/all_global_shap.csv")

    # ── Combined rank table ───────────────────────────────────────────────
    rank_combined = pd.DataFrame(
        {f"{lvl}_{mdl}": r for (lvl, mdl), r in all_ranks.items()}
    ).T
    rank_combined.index.name = "level_model"
    rank_combined.to_csv(SHAP_DIR / "ranks" / "all_ranks.csv")
    log.info(f"  Saved combined ranks        : shap/ranks/all_ranks.csv")

    # ── Stability CSV ─────────────────────────────────────────────────────
    stability_df.to_csv(SHAP_DIR / "stability" / "rank_stability.csv", index=False)
    log.info(f"  Saved rank stability        : shap/stability/rank_stability.csv")

    # ── ESI scores CSV ────────────────────────────────────────────────────
    esi_df.to_csv(SHAP_DIR / "stability" / "esi_scores.csv", index=False)
    log.info(f"  Saved ESI scores            : shap/stability/esi_scores.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 3 — SHAP Extraction Pipeline")
    log.info("=" * 65)
    log.info(f"Models     : {MODEL_NAMES}")
    log.info(f"Levels     : {SCARCITY_LEVELS}")
    log.info(f"Features   : {FEATURES}")
    log.info(f"MLP sample : background={KERNEL_BG_SAMPLES}, eval={KERNEL_EVAL_SAMPLES}")

    all_global: dict = {}
    all_local:  list = []
    all_ranks:  dict = {}

    for level in SCARCITY_LEVELS:
        log.info(f"\n{'='*65}")
        log.info(f"  Scarcity level : {level}")
        log.info(f"{'='*65}")

        for model_name in MODEL_NAMES:
            try:
                result = compute_shap(level, model_name)
                all_global[(level, model_name)] = result["global_importance"]
                all_local.append(result["local_df"])
                all_ranks[(level, model_name)]  = result["ranks"]

            except Exception as e:
                log.error(f"  FAILED  {level}/{model_name}: {e}")
                continue

    # ── Stability analysis ────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Stability Analysis (Spearman ρ vs S1 baseline)")
    log.info(f"{'='*65}")
    stability_df, esi_df = compute_stability(all_ranks)

    # ── Save everything ───────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Saving outputs")
    log.info(f"{'='*65}")
    save_outputs(all_global, all_local, all_ranks, stability_df, esi_df)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("Phase 3 complete — ESI scores by model:")
    log.info(f"{'='*65}")
    log.info("\n" + esi_df.to_string(index=False))

    log.info(f"\n{'='*65}")
    log.info("Global feature importance at S1 (full data baseline):")
    log.info(f"{'='*65}")
    for model_name in MODEL_NAMES:
        key = ("S1", model_name)
        if key in all_global:
            imp = all_global[key].sort_values(ascending=False)
            log.info(f"\n  {model_name}:")
            for feat, val in imp.items():
                bar = "█" * int(val / imp.max() * 20)
                log.info(f"    {feat:<12} {val:.4f}  {bar}")

    log.info("\nAll SHAP outputs saved. Ready for Phase 4 (RES-CHD framework).  ✓")


if __name__ == "__main__":
    main()