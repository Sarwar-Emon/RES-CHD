"""
RES-CHD Phase 4B — MLP Bootstrapped ESI
=========================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Purpose:
  Phase 4 skipped MLP bootstrap because KernelExplainer is too slow to
  rerun for each bootstrap iteration. This module computes MLP bootstrapped
  ESI using the same SHAP-array bootstrap approach used for tree/linear
  models in Phase 4 — bootstrapping over already-computed local SHAP
  matrices from Phase 3 rather than rerunning the explainer.

  This closes the asymmetry in Phase 4 results where RF, LR, and XGBoost
  have proper 95% CI but MLP only has a point estimate.

Method:
  For each scarcity level Si (S2–S7):
    1. Load MLP local SHAP matrix from Phase 3
       (n_patients × n_features, already computed)
    2. For b in 1..N_BOOTSTRAP:
       a. Resample patient rows with replacement
       b. Compute mean(|SHAP|) per feature on resampled rows
       c. Rank features by resampled importance
       d. Compute Spearman rho vs S1 point-estimate baseline
    3. CI = [percentile(2.5), percentile(97.5)]

This is statistically valid — it quantifies how sensitive MLP global
feature importance ranking is to which patients are included, using
the same method as all other models.

Inputs:
  - shap/local/S{1..7}_MLP_local_shap.csv    (from Phase 3)
  - shap/ranks/S1_MLP_ranks.csv              (S1 baseline, from Phase 3)
  - results/phase4/bootstrapped_esi.csv       (existing Phase 4 results)
  - results/phase4/reliability_flags.csv      (existing Phase 4 flags)

Outputs:
  - results/phase4/bootstrapped_esi.csv       (updated with MLP row)
  - results/phase4/bootstrap_distributions/MLP_{level}_bootstrap.csv
  - results/phase4/reliability_flags.csv      (updated MLP flags)
  - results/phase4/res_chd_report.csv         (updated master report)
  - logs/phase4b_mlp_bootstrap.txt

Requirements:
    pip install pandas numpy scipy
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
SHAP_DIR   = BASE_DIR / "shap"
RESULTS    = BASE_DIR / "results"
P4_DIR     = RESULTS / "phase4"
BOOT_DIR   = P4_DIR / "bootstrap_distributions"
LOG_DIR    = BASE_DIR / "logs"

for d in [BOOT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase4b_mlp_bootstrap.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "MLP"
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]
FEATURES        = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
N_BOOTSTRAP     = 200      # More bootstraps for MLP since it is fast now
RANDOM_SEED     = 42
CI_ALPHA        = 0.95

# Reliability thresholds (same as Phase 4)
ESI_GREEN = 0.85
ESI_AMBER = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_local_shap(level: str) -> np.ndarray:
    """Load local SHAP matrix from Phase 3."""
    path = SHAP_DIR / "local" / f"{level}_{MODEL_NAME}_local_shap.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing local SHAP file: {path}")
    df       = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c in FEATURES]
    return df[feat_cols].values   # shape (n_patients, n_features)


def shap_array_to_ranks(shap_arr: np.ndarray) -> np.ndarray:
    """Compute feature ranks from SHAP array. Rank 1 = most important."""
    global_imp = np.abs(shap_arr).mean(axis=0)
    order      = np.argsort(global_imp)[::-1]
    ranks      = np.empty_like(order)
    ranks[order] = np.arange(1, len(global_imp) + 1)
    return ranks


def load_s1_baseline_ranks() -> np.ndarray:
    """Load MLP S1 point-estimate baseline ranks from Phase 3."""
    path = SHAP_DIR / "ranks" / f"S1_{MODEL_NAME}_ranks.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing S1 ranks: {path}")
    df = pd.read_csv(path, index_col=0)
    return df.iloc[:, 0].values


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP MLP ESI
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_mlp_esi() -> dict:
    """
    Bootstrap ESI for MLP using Phase 3 local SHAP arrays.
    Returns dict: level → {mean, std, ci_lower, ci_upper, rho_values}
    """
    log.info(f"  Loading S1 MLP baseline ranks ...")
    s1_ranks = load_s1_baseline_ranks()
    log.info(f"  S1 baseline ranks: {dict(zip(FEATURES, s1_ranks))}")

    boot_results = {}
    np.random.seed(RANDOM_SEED)

    for level in SCARCITY_LEVELS[1:]:   # S2 … S7
        log.info(f"\n  Bootstrapping MLP  {level}  (B={N_BOOTSTRAP}) ...")

        try:
            shap_arr = load_local_shap(level)
        except FileNotFoundError as e:
            log.warning(f"  {e} — skipping {level}")
            continue

        n_patients = shap_arr.shape[0]
        log.info(f"  SHAP matrix shape: {shap_arr.shape}")

        rho_values = []
        for b in range(N_BOOTSTRAP):
            # Resample patient rows with replacement
            idx           = np.random.choice(n_patients, size=n_patients, replace=True)
            shap_resample = shap_arr[idx]
            ranks_si      = shap_array_to_ranks(shap_resample)
            rho, _        = spearmanr(s1_ranks, ranks_si)
            rho_values.append(float(rho))

        rho_arr = np.array(rho_values)
        alpha   = 1 - CI_ALPHA

        result = {
            "rho_values":  rho_arr,
            "mean":        float(np.mean(rho_arr)),
            "std":         float(np.std(rho_arr)),
            "ci_lower":    float(np.percentile(rho_arr, 100 * alpha / 2)),
            "ci_upper":    float(np.percentile(rho_arr, 100 * (1 - alpha / 2))),
            "n_bootstrap": len(rho_arr),
            "n_patients":  n_patients,
        }
        boot_results[level] = result

        log.info(
            f"  MLP  {level}  "
            f"mean_rho={result['mean']:.4f}  "
            f"95%CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  "
            f"std={result['std']:.4f}  "
            f"n_patients={n_patients}"
        )

        # Save bootstrap distribution
        dist_df   = pd.DataFrame({
            "bootstrap_index": range(len(rho_arr)),
            "spearman_rho":    rho_arr,
        })
        dist_path = BOOT_DIR / f"MLP_{level}_bootstrap.csv"
        dist_df.to_csv(dist_path, index=False)

    return boot_results


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE OVERALL MLP ESI FROM LEVEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_mlp_overall_esi(boot_results: dict) -> dict:
    """Aggregate per-level bootstrap results into overall ESI with CI."""
    level_means  = [v["mean"]     for v in boot_results.values()]
    level_stds   = [v["std"]      for v in boot_results.values()]
    level_lows   = [v["ci_lower"] for v in boot_results.values()]
    level_highs  = [v["ci_upper"] for v in boot_results.values()]

    esi_mean  = float(np.mean(level_means))
    esi_std   = float(np.sqrt(np.mean(np.array(level_stds) ** 2)))
    esi_lower = float(np.mean(level_lows))
    esi_upper = float(np.mean(level_highs))

    if esi_mean >= ESI_GREEN:
        reliability = "HIGH"
    elif esi_mean >= ESI_AMBER:
        reliability = "MODERATE"
    else:
        reliability = "LOW"

    return {
        "model":          MODEL_NAME,
        "esi_mean":       round(esi_mean,  4),
        "esi_std":        round(esi_std,   4),
        "esi_ci_lower":   round(esi_lower, 4),
        "esi_ci_upper":   round(esi_upper, 4),
        "n_levels":       len(boot_results),
        "n_bootstrap":    N_BOOTSTRAP,
        "ci_level":       f"{int(CI_ALPHA*100)}%",
        "reliability":    reliability,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UPDATE PHASE 4 OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def update_bootstrapped_esi_csv(mlp_esi: dict) -> None:
    """Replace MLP row in bootstrapped_esi.csv with proper values."""
    path = P4_DIR / "bootstrapped_esi.csv"
    if not path.exists():
        log.warning(f"  bootstrapped_esi.csv not found — creating new")
        df = pd.DataFrame([mlp_esi])
    else:
        df = pd.read_csv(path)
        # Remove old MLP row if exists
        df = df[df["model"] != MODEL_NAME]
        # Add new MLP row
        df = pd.concat([df, pd.DataFrame([mlp_esi])], ignore_index=True)
        # Sort by esi_mean descending
        df = df.sort_values("esi_mean", ascending=False).reset_index(drop=True)

    df.to_csv(path, index=False)
    log.info(f"  Updated bootstrapped_esi.csv")
    log.info(f"\n{df[['model','esi_mean','esi_ci_lower','esi_ci_upper','reliability']].to_string(index=False)}")


def update_reliability_flags(boot_results: dict,
                              phase3_stability: pd.DataFrame) -> None:
    """Update MLP flags in reliability_flags.csv with bootstrapped values."""
    flags_path = P4_DIR / "reliability_flags.csv"
    if not flags_path.exists():
        log.warning("  reliability_flags.csv not found — skipping flag update")
        return

    flags_df = pd.read_csv(flags_path)

    # Remove old MLP rows (except S1 baseline)
    flags_df = flags_df[
        ~((flags_df["model"] == MODEL_NAME) & (flags_df["level"] != "S1"))
    ]

    new_rows = []
    for level in SCARCITY_LEVELS[1:]:
        boot_data  = boot_results.get(level, {})
        boot_mean  = boot_data.get("mean",     None)
        boot_lower = boot_data.get("ci_lower", None)
        boot_upper = boot_data.get("ci_upper", None)

        # Point estimate from Phase 3
        phase3_row = phase3_stability[
            (phase3_stability["model"] == MODEL_NAME) &
            (phase3_stability["comparison"] == level)
        ]
        point_rho   = float(phase3_row["spearman_rho"].values[0]) if len(phase3_row) > 0 else None
        significant = bool(phase3_row["significant"].values[0])   if len(phase3_row) > 0 else False

        # Determine flag
        if boot_mean is None:
            flag  = "UNKNOWN"
            basis = "No bootstrap data"
            note  = "Cannot assess reliability"
        elif boot_mean >= ESI_GREEN:
            flag  = "GREEN"
            basis = f"Bootstrapped ESI = {boot_mean:.4f} >= {ESI_GREEN}"
            note  = "Explanation reliable."
            if boot_lower < ESI_GREEN:
                note += " Note: lower CI crosses threshold."
        elif boot_mean >= ESI_AMBER:
            flag  = "AMBER"
            basis = f"Bootstrapped ESI = {boot_mean:.4f} in [{ESI_AMBER}, {ESI_GREEN})"
            note  = "Moderate reliability. Use with clinical judgment."
        else:
            flag  = "RED"
            basis = f"Bootstrapped ESI = {boot_mean:.4f} < {ESI_AMBER}"
            note  = "Explanation unreliable. Do not use for clinical decisions."

        if not significant:
            note += " [Point estimate p >= 0.05 — interpret with caution]"

        new_rows.append({
            "model":          MODEL_NAME,
            "level":          level,
            "esi_point":      round(point_rho,  4) if point_rho  else None,
            "esi_boot_mean":  round(boot_mean,  4) if boot_mean  else None,
            "esi_boot_lower": round(boot_lower, 4) if boot_lower else None,
            "esi_boot_upper": round(boot_upper, 4) if boot_upper else None,
            "flag":           flag,
            "flag_basis":     basis,
            "clinical_note":  note,
        })

        log.info(
            f"  MLP  {level}  [{flag:<6}]  "
            f"boot={boot_mean:.4f}  "
            f"CI=[{boot_lower:.4f},{boot_upper:.4f}]  "
            f"point={point_rho:.4f}"
        )

    new_df   = pd.DataFrame(new_rows)
    flags_df = pd.concat([flags_df, new_df], ignore_index=True)
    flags_df.to_csv(flags_path, index=False)
    log.info(f"  Updated reliability_flags.csv")


def update_res_chd_report() -> None:
    """Rebuild master report from updated CSVs."""
    boot_df  = pd.read_csv(P4_DIR / "bootstrapped_esi.csv")
    flags_df = pd.read_csv(P4_DIR / "reliability_flags.csv")

    master = flags_df.merge(
        boot_df[["model", "esi_mean", "esi_std",
                 "esi_ci_lower", "esi_ci_upper", "reliability"]],
        on="model", how="left"
    )
    path = P4_DIR / "res_chd_report.csv"
    master.to_csv(path, index=False)
    log.info(f"  Updated res_chd_report.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 4B — MLP Bootstrapped ESI")
    log.info("=" * 65)
    log.info(f"Model          : {MODEL_NAME}")
    log.info(f"Bootstrap B    : {N_BOOTSTRAP}")
    log.info(f"CI level       : {int(CI_ALPHA*100)}%")
    log.info(f"Method         : SHAP-array bootstrap (Phase 3 local SHAP files)")

    # ── Run bootstrap ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Bootstrapping MLP ESI across S2–S7 ...")
    log.info(f"{'='*65}")
    boot_results = bootstrap_mlp_esi()

    if not boot_results:
        log.error("  No bootstrap results — check Phase 3 SHAP files exist")
        return

    # ── Compute overall ESI ───────────────────────────────────────────────
    mlp_esi = compute_mlp_overall_esi(boot_results)

    log.info(f"\n{'='*65}")
    log.info("  MLP Overall Bootstrapped ESI:")
    log.info(f"{'='*65}")
    log.info(f"  ESI mean    : {mlp_esi['esi_mean']:.4f}")
    log.info(f"  ESI std     : {mlp_esi['esi_std']:.4f}")
    log.info(f"  95% CI      : [{mlp_esi['esi_ci_lower']:.4f}, {mlp_esi['esi_ci_upper']:.4f}]")
    log.info(f"  Reliability : {mlp_esi['reliability']}")

    # ── Load Phase 3 stability for flag updates ───────────────────────────
    phase3_stab = pd.read_csv(SHAP_DIR / "stability" / "rank_stability.csv")

    # ── Update Phase 4 outputs ────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Updating Phase 4 output files ...")
    log.info(f"{'='*65}")
    update_bootstrapped_esi_csv(mlp_esi)
    update_reliability_flags(boot_results, phase3_stab)
    update_res_chd_report()

    # ── Final summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("Phase 4B complete — Updated ESI table (all 4 models):")
    log.info(f"{'='*65}")

    final_df = pd.read_csv(P4_DIR / "bootstrapped_esi.csv")
    cols     = ["model", "esi_mean", "esi_ci_lower",
                "esi_ci_upper", "reliability"]
    log.info("\n" + final_df[cols].to_string(index=False))

    log.info(f"\n{'='*65}")
    log.info("MLP flag summary after bootstrap:")
    log.info(f"{'='*65}")
    flags_df = pd.read_csv(P4_DIR / "reliability_flags.csv")
    mlp_flags = flags_df[flags_df["model"] == MODEL_NAME][
        ["level", "esi_boot_mean", "esi_boot_lower", "esi_boot_upper", "flag"]
    ]
    log.info("\n" + mlp_flags.to_string(index=False))

    log.info("\nAll Phase 4 outputs updated with MLP bootstrap results. ✓")
    log.info("MLP asymmetry resolved — all 4 models now have bootstrapped CI.")


if __name__ == "__main__":
    main()