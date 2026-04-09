"""
RES-CHD Phase 5F — Calibration Analysis
========================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Purpose:
  AUC measures whether the model RANKS patients correctly (high risk vs low risk).
  Calibration measures whether the model's PROBABILITIES are ACCURATE.

  Simple example:
    AUC = 0.85 means the model correctly ranks 85% of CHD vs non-CHD pairs.
    Good calibration means: when the model says "30% CHD risk", roughly 30%
    of those patients actually have CHD.

  Why calibration matters for your thesis:
  A model can have high AUC but terrible calibration — it ranks patients
  correctly but its probability estimates are systematically wrong.
  In clinical settings, doctors use these probabilities to make decisions
  like "refer for further testing" or "prescribe medication."
  If probabilities are miscalibrated under data scarcity, clinical
  decisions based on them are unreliable — EVEN IF AUC looks fine.

  This adds a third dimension to your framework:
    Dimension 1: Predictive accuracy (AUC, PR-AUC) — Phase 2
    Dimension 2: Explanation stability (ESI) — Phases 3-4
    Dimension 3: Probability calibration — THIS PHASE

  Key question: Does calibration degrade faster or slower than ESI
  under progressive data scarcity? If calibration degrades first,
  it is a leading indicator that should trigger explanation review.

Four metrics computed:

  1. Brier Score
     Mean squared error between predicted probability and true label.
     Lower = better. Range [0, 1].
     Brier < 0.1: excellent | 0.1-0.2: good | > 0.2: poor

  2. Expected Calibration Error (ECE)
     Average gap between predicted probability and actual outcome rate,
     measured across probability bins.
     ECE = 0: perfectly calibrated | ECE > 0.1: poorly calibrated

  3. Reliability Diagram (calibration curve)
     Visual: predicted probability (x-axis) vs actual CHD rate (y-axis).
     Perfect calibration = diagonal line.
     Above diagonal = underconfident | Below diagonal = overconfident

  4. Calibration Stability Index (CSI) — NEW METRIC
     Similar to ESI but for calibration:
     How much does the calibration curve change across scarcity levels?
     CSI = 1 - mean(|ECE_Si - ECE_S1|) / ECE_S1
     High CSI = calibration is stable under scarcity
     Low CSI = calibration degrades under scarcity

Inputs:
  - data/scarcity_levels/S{1..7}_test.csv
  - models/S{1..7}_{Model}.pkl

Outputs:
  - results/phase5f/calibration_metrics.csv
  - results/phase5f/calibration_stability.csv
  - results/phase5f/reliability_diagrams/
  - results/phase5f/figures/fig_calibration_curves.png
  - results/phase5f/figures/fig_calibration_vs_esi.png
  - results/phase5f/figures/fig_brier_score_trends.png
  - logs/phase5f_report.txt

Requirements:
    pip install pandas numpy scipy scikit-learn matplotlib joblib
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "scarcity_levels"
MODELS_DIR = BASE_DIR / "models"
SHAP_DIR   = BASE_DIR / "shap"
RESULTS    = BASE_DIR / "results"
P4_DIR     = RESULTS / "phase4"
P5F_DIR    = RESULTS / "phase5f"
FIG_DIR    = P5F_DIR / "figures"
LOG_DIR    = BASE_DIR / "logs"

for d in [P5F_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase5f_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]
MODEL_NAMES     = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
FEATURES        = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
TARGET          = "chd"
N_BINS          = 10    # calibration bins

COLORS = {
    "XGBoost":            "#27AE60",
    "RandomForest":       "#C0392B",
    "LogisticRegression": "#2980B9",
    "MLP":                "#8E44AD",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_model(level: str, model_name: str):
    path = MODELS_DIR / f"{level}_{model_name}.pkl"
    assert path.exists(), f"Missing: {path}"
    return joblib.load(path)


def load_test(level: str) -> tuple:
    path = DATA_DIR / f"{level}_test.csv"
    df   = pd.read_csv(path)
    return df[FEATURES], df[TARGET].values


def get_proba(model, model_name: str, X: pd.DataFrame) -> np.ndarray:
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return model.predict(X).astype(float)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = N_BINS) -> float:
    """
    Expected Calibration Error.
    Divide predictions into n_bins equal-width probability bins.
    ECE = weighted average of |mean_predicted - fraction_positive| per bin.
    """
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece        = 0.0
    n          = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask      = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        bin_n    = int(mask.sum())
        ece     += (bin_n / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_overconfidence(y_true: np.ndarray,
                           y_prob: np.ndarray) -> float:
    """
    Overconfidence = mean(predicted_prob) - mean(actual_rate).
    Positive = model predicts higher CHD risk than reality (overconfident).
    Negative = model predicts lower CHD risk than reality (underconfident).
    """
    return float(y_prob.mean() - y_true.mean())


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — CALIBRATION METRICS PER MODEL PER LEVEL
# ══════════════════════════════════════════════════════════════════════════════
def compute_calibration_metrics() -> pd.DataFrame:
    """
    Compute Brier score, ECE, and overconfidence per model per level.
    """
    log.info(f"\n{'='*65}")
    log.info("  Analysis 1 — Calibration metrics per model per level")
    log.info(f"{'='*65}")

    rows = []

    for model_name in MODEL_NAMES:
        log.info(f"\n  {model_name}:")
        log.info(f"  {'Level':<6}  {'Brier':<8}  {'ECE':<8}  "
                 f"{'Overconf':<10}  {'Interpretation'}")
        log.info(f"  {'-'*60}")

        for level in SCARCITY_LEVELS:
            try:
                X_test, y_test = load_test(level)
                model          = load_model(level, model_name)
                y_prob         = get_proba(model, model_name, X_test)
            except Exception as e:
                log.warning(f"  Failed {model_name} {level}: {e}")
                continue

            brier  = brier_score_loss(y_test, y_prob)
            ece    = compute_ece(y_test, y_prob)
            overconf = compute_overconfidence(y_test, y_prob)

            # Brier skill score vs naive baseline (always predict prevalence)
            prevalence    = float(y_test.mean())
            brier_baseline = prevalence * (1 - prevalence)
            brier_skill   = 1 - (brier / brier_baseline) if brier_baseline > 0 else 0

            # Calibration quality label
            if ece < 0.05:
                cal_quality = "EXCELLENT"
            elif ece < 0.10:
                cal_quality = "GOOD"
            elif ece < 0.15:
                cal_quality = "MODERATE"
            else:
                cal_quality = "POOR"

            log.info(
                f"  {level:<6}  "
                f"{brier:.4f}    "
                f"{ece:.4f}    "
                f"{overconf:+.4f}      "
                f"{cal_quality}"
            )

            rows.append({
                "model":          model_name,
                "level":          level,
                "brier_score":    round(float(brier),       4),
                "brier_skill":    round(float(brier_skill), 4),
                "ece":            round(float(ece),         4),
                "overconfidence": round(float(overconf),    4),
                "cal_quality":    cal_quality,
                "n_test":         len(y_test),
                "chd_prevalence": round(float(prevalence),  4),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — CALIBRATION STABILITY INDEX (CSI)
#
# How stable is calibration across scarcity levels?
# CSI measures how much ECE changes from S1 baseline under scarcity.
# High CSI = calibration is stable = trustworthy under scarcity
# Low CSI  = calibration degrades = cannot trust probabilities at low data
# ══════════════════════════════════════════════════════════════════════════════
def compute_calibration_stability(cal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calibration Stability Index per model — analogous to ESI.
    """
    log.info(f"\n{'='*65}")
    log.info("  Analysis 2 — Calibration Stability Index (CSI)")
    log.info(f"{'='*65}")

    rows = []

    for model_name in MODEL_NAMES:
        m_df    = cal_df[cal_df["model"] == model_name]
        s1_ece  = m_df[m_df["level"] == "S1"]["ece"].values[0] \
                  if len(m_df[m_df["level"] == "S1"]) > 0 else np.nan

        if np.isnan(s1_ece):
            continue

        log.info(f"\n  {model_name}  S1 ECE baseline = {s1_ece:.4f}")

        level_csi = []
        for level in SCARCITY_LEVELS[1:]:
            si_ece = m_df[m_df["level"] == level]["ece"].values[0] \
                     if len(m_df[m_df["level"] == level]) > 0 else np.nan
            if np.isnan(si_ece):
                continue

            # CSI = 1 - relative ECE change
            ece_change = abs(si_ece - s1_ece)
            csi        = max(0, 1 - ece_change / (s1_ece + 1e-6))
            level_csi.append(csi)

            # Compare calibration trend to ESI
            esi_row = None
            try:
                esi_df  = pd.read_csv(SHAP_DIR / "stability" / "rank_stability.csv")
                esi_row = esi_df[
                    (esi_df["model"] == model_name) &
                    (esi_df["comparison"] == level)
                ]
                esi_val = float(esi_row["spearman_rho"].values[0]) \
                          if len(esi_row) > 0 else np.nan
            except Exception:
                esi_val = np.nan

            log.info(
                f"  {level}  ECE={si_ece:.4f}  "
                f"CSI={csi:.4f}  "
                f"ESI={esi_val:.4f}"
                + (" [CAL degrades faster]" if not np.isnan(esi_val) and csi < esi_val else
                   " [ESI degrades faster]" if not np.isnan(esi_val) and csi > esi_val else "")
            )

            rows.append({
                "model":   model_name,
                "level":   level,
                "s1_ece":  round(s1_ece,  4),
                "si_ece":  round(si_ece,  4),
                "csi":     round(csi,     4),
                "esi":     round(esi_val, 4) if not np.isnan(esi_val) else None,
                "cal_degrades_faster": csi < esi_val if not np.isnan(esi_val) else None,
            })

        mean_csi = float(np.mean(level_csi)) if level_csi else np.nan
        log.info(f"  Mean CSI = {mean_csi:.4f}")

    return pd.DataFrame(rows)


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
    "grid.linestyle":    "--",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


def plot_calibration_curves(cal_df: pd.DataFrame) -> None:
    """
    Reliability diagrams for each model at S1 and S7.
    Shows how calibration changes from full data to minimum data.
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col_idx, model_name in enumerate(MODEL_NAMES):
        for row_idx, level in enumerate(["S1", "S7"]):
            ax = axes[row_idx, col_idx]

            try:
                X_test, y_test = load_test(level)
                model          = load_model(level, model_name)
                y_prob         = get_proba(model, model_name, X_test)

                prob_true, prob_pred = calibration_curve(
                    y_test, y_prob, n_bins=N_BINS, strategy="uniform"
                )

                ece = compute_ece(y_test, y_prob)
                brier = brier_score_loss(y_test, y_prob)

                ax.plot(prob_pred, prob_true,
                       color=COLORS[model_name],
                       linewidth=2, marker="o", markersize=5,
                       label=f"Model (ECE={ece:.3f})")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1,
                       alpha=0.5, label="Perfect calibration")
                ax.fill_between([0, 1], [0, 1], [0, 1],
                               alpha=0.05, color="gray")

                ax.set_xlim(0, 0.5)
                ax.set_ylim(0, 0.5)
                ax.set_xlabel("Mean predicted probability", fontsize=8)
                ax.set_ylabel("Fraction positive", fontsize=8)

                title_level = "Full data (S1)" if level == "S1" else "Min data (S7)"
                ax.set_title(
                    f"{model_name}\n{title_level}\nBrier={brier:.3f}  ECE={ece:.3f}",
                    fontsize=9, fontweight="bold",
                    color=COLORS[model_name]
                )
                ax.legend(fontsize=7)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}",
                       ha="center", va="center", transform=ax.transAxes,
                       fontsize=8)

    fig.suptitle(
        "Fig — Reliability Diagrams: Calibration at S1 (Full Data) vs S7 (Minimum Data)\n"
        "Diagonal = perfect calibration. Deviation shows systematic over/under confidence.",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_calibration_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_calibration_curves.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_calibration_curves.png / .pdf")


def plot_brier_trends(cal_df: pd.DataFrame) -> None:
    """
    Brier score across scarcity levels — shows calibration degradation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    levels    = SCARCITY_LEVELS

    # Left: Brier score
    ax = axes[0]
    for model_name in MODEL_NAMES:
        m_df = cal_df[cal_df["model"] == model_name].sort_values("level")
        ax.plot(range(len(m_df)), m_df["brier_score"].values,
               color=COLORS[model_name], linewidth=2,
               marker="o", markersize=5, label=model_name)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Brier Score (lower = better)", fontsize=10)
    ax.set_title("Brier Score across scarcity levels", fontsize=11)
    ax.legend(fontsize=9)

    # Right: ECE
    ax2 = axes[1]
    for model_name in MODEL_NAMES:
        m_df = cal_df[cal_df["model"] == model_name].sort_values("level")
        ax2.plot(range(len(m_df)), m_df["ece"].values,
                color=COLORS[model_name], linewidth=2,
                marker="o", markersize=5, label=model_name)

    ax2.axhline(0.10, color="gray", linestyle="--",
               linewidth=1, alpha=0.6, label="ECE=0.10 threshold")
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.set_ylabel("Expected Calibration Error (lower = better)", fontsize=10)
    ax2.set_title("ECE across scarcity levels", fontsize=11)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Fig — Calibration Degradation Under Progressive Data Scarcity\n"
        "Both Brier score and ECE reveal whether probability estimates remain trustworthy",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_brier_score_trends.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_brier_score_trends.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_brier_score_trends.png / .pdf")


def plot_calibration_vs_esi(stab_df: pd.DataFrame) -> None:
    """
    Compare CSI vs ESI across models and levels.
    Shows whether calibration or explanation stability degrades first.
    """
    if stab_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    valid   = stab_df.dropna(subset=["csi", "esi"])

    for model_name in MODEL_NAMES:
        m_df = valid[valid["model"] == model_name]
        if m_df.empty:
            continue

        ax.scatter(m_df["esi"], m_df["csi"],
                  color=COLORS[model_name], s=80, alpha=0.8,
                  label=model_name, zorder=3)

        # Connect points by level order
        m_sorted = m_df.sort_values("level")
        ax.plot(m_sorted["esi"], m_sorted["csi"],
               color=COLORS[model_name], alpha=0.3,
               linewidth=1, linestyle="--")

        # Label S7 point (most extreme)
        s7 = m_df[m_df["level"] == "S7"]
        if not s7.empty:
            ax.annotate(
                f"{model_name[:3]} S7",
                (float(s7["esi"]), float(s7["csi"])),
                textcoords="offset points",
                xytext=(5, 5), fontsize=8,
                color=COLORS[model_name]
            )

    # Diagonal reference
    lims = [0.5, 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.4,
           label="CSI = ESI (equal degradation)")

    ax.set_xlabel("ESI (Explanation Stability)", fontsize=10)
    ax.set_ylabel("CSI (Calibration Stability)", fontsize=10)
    ax.set_title(
        "Fig — Calibration Stability vs Explanation Stability\n"
        "Below diagonal: calibration degrades faster than explanations\n"
        "Above diagonal: explanations degrade faster than calibration",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    path = FIG_DIR / "fig_calibration_vs_esi.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_calibration_vs_esi.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_calibration_vs_esi.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 5F — Calibration Analysis")
    log.info("=" * 65)
    log.info("Purpose  : Third dimension of model reliability — probability calibration")
    log.info("Metrics  : Brier score, ECE, Calibration Stability Index (CSI)")
    log.info("Question : Does calibration degrade faster or slower than ESI?")

    # ── Analysis 1: Calibration metrics ──────────────────────────────────
    cal_df = compute_calibration_metrics()
    cal_df.to_csv(P5F_DIR / "calibration_metrics.csv", index=False)
    log.info(f"\n  Saved calibration_metrics.csv ({len(cal_df)} rows)")

    # ── Analysis 2: Calibration stability ────────────────────────────────
    stab_df = compute_calibration_stability(cal_df)
    if not stab_df.empty:
        stab_df.to_csv(P5F_DIR / "calibration_stability.csv", index=False)
        log.info(f"\n  Saved calibration_stability.csv")

    # ── Visualizations ────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Generating figures ...")
    log.info(f"{'='*65}")

    plot_calibration_curves(cal_df)
    plot_brier_trends(cal_df)
    plot_calibration_vs_esi(stab_df)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("Phase 5F complete — Key findings:")
    log.info(f"{'='*65}")

    log.info("\n  Calibration quality at S1 (full data):")
    s1_df = cal_df[cal_df["level"] == "S1"]
    for _, row in s1_df.iterrows():
        log.info(
            f"  {row['model']:<22}  "
            f"Brier={row['brier_score']:.4f}  "
            f"ECE={row['ece']:.4f}  "
            f"Quality={row['cal_quality']}"
        )

    log.info("\n  Calibration quality at S7 (minimum data):")
    s7_df = cal_df[cal_df["level"] == "S7"]
    for _, row in s7_df.iterrows():
        log.info(
            f"  {row['model']:<22}  "
            f"Brier={row['brier_score']:.4f}  "
            f"ECE={row['ece']:.4f}  "
            f"Quality={row['cal_quality']}"
        )

    log.info("\n  ECE change S1 → S7 (positive = calibration got worse):")
    for model_name in MODEL_NAMES:
        s1_ece = cal_df[(cal_df["model"]==model_name) &
                        (cal_df["level"]=="S1")]["ece"].values
        s7_ece = cal_df[(cal_df["model"]==model_name) &
                        (cal_df["level"]=="S7")]["ece"].values
        if len(s1_ece) > 0 and len(s7_ece) > 0:
            delta = s7_ece[0] - s1_ece[0]
            log.info(
                f"  {model_name:<22}  "
                f"S1={s1_ece[0]:.4f}  →  "
                f"S7={s7_ece[0]:.4f}  "
                f"(Δ={delta:+.4f})"
            )

    if not stab_df.empty:
        log.info("\n  Cases where calibration degrades faster than ESI:")
        faster = stab_df[stab_df["cal_degrades_faster"] == True]
        log.info(f"  {len(faster)}/{len(stab_df)} level-model pairs")

    log.info(
        "\n  INTERPRETATION:"
        "\n  If calibration degrades faster than ESI — probability estimates"
        "\n  become unreliable before rankings change. Clinical decisions based"
        "\n  on absolute risk thresholds (e.g. refer if risk > 20%) are affected"
        "\n  earlier than decisions based on relative risk ranking."
        "\n  Your three-dimensional framework (AUC + ESI + calibration) captures"
        "\n  this nuance that no single metric can detect alone."
    )

    log.info("\nPhase 5F complete. ✓")
    log.info("Calibration is the third pillar of the RES-CHD reliability framework.")


if __name__ == "__main__":
    main()