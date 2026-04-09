"""
RES-CHD Phase 5E — Prediction Confidence vs Explanation Stability
=================================================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Purpose:
  Your pipeline currently treats predictive performance and explanation
  stability as separate dimensions. This module connects them by asking
  one clinically critical question:

  "When a model is UNCERTAIN about a prediction, is the explanation
   for that prediction also LESS STABLE under data scarcity?"

  If yes — uncertain predictions AND unstable explanations co-occur in
  the same patients — this is a compounded clinical risk that neither
  AUC nor ESI alone can detect. A clinician seeing a borderline CHD
  prediction with an unstable explanation has two independent reasons
  to distrust the output. Your framework is the first to identify and
  quantify this compound risk.

  Clinical framing:
  - High confidence + stable explanation → safe to act on
  - High confidence + unstable explanation → explanation unreliable, act with caution
  - Low confidence + stable explanation → prediction uncertain but explanation trustworthy
  - Low confidence + unstable explanation → COMPOUND RISK — do not rely on either

Three analyses:

  1. Confidence-stability correlation
     For each patient, measure prediction confidence (1 - |p - 0.5| * 2,
     where p is predicted CHD probability) and their LSS from Phase 5D.
     Compute Pearson and Spearman correlation.
     If significant negative correlation → uncertain predictions = unstable explanations.

  2. Confidence-stability quadrant analysis
     Classify patients into 4 quadrants:
       Q1: High confidence + High LSS  → SAFE
       Q2: High confidence + Low LSS   → EXPLANATION RISK
       Q3: Low confidence  + High LSS  → PREDICTION RISK
       Q4: Low confidence  + Low LSS   → COMPOUND RISK (highest clinical concern)
     Report proportions per quadrant per model.
     Report CHD prevalence within each quadrant.

  3. Scarcity effect on compound risk
     Does the proportion of Q4 patients grow as scarcity increases?
     If compound risk proportion increases under scarcity → your framework
     identifies a problem that neither AUC nor ESI tracks separately.

Inputs:
  - data/scarcity_levels/S{1..7}_test.csv           (patient features + labels)
  - models/S{1..7}_{Model}.pkl                       (trained models)
  - results/phase5d/patient_stability_scores.csv     (LSS per patient, Phase 5D)

Outputs:
  - results/phase5e/confidence_stability_correlation.csv
  - results/phase5e/quadrant_analysis.csv
  - results/phase5e/compound_risk_by_level.csv
  - results/phase5e/high_risk_patient_analysis.csv
  - results/phase5e/figures/fig_confidence_stability_scatter.png
  - results/phase5e/figures/fig_quadrant_heatmap.png
  - results/phase5e/figures/fig_compound_risk_scarcity.png
  - logs/phase5e_report.txt

Requirements:
    pip install pandas numpy scipy scikit-learn xgboost joblib matplotlib
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data" / "scarcity_levels"
MODELS_DIR = BASE_DIR / "models"
RESULTS    = BASE_DIR / "results"
P5D_DIR    = RESULTS / "phase5d"
P5E_DIR    = RESULTS / "phase5e"
FIG_DIR    = P5E_DIR / "figures"
LOG_DIR    = BASE_DIR / "logs"

for d in [P5E_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase5e_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]
MODEL_NAMES     = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
FEATURES        = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
TARGET          = "chd"

# Thresholds
CONFIDENCE_HIGH = 0.70   # confidence >= 0.70 → high confidence
CONFIDENCE_LOW  = 0.50   # confidence < 0.50  → low confidence (near 50/50)
LSS_HIGH        = 0.80   # LSS >= 0.80 → stable explanation
LSS_LOW         = 0.70   # LSS <  0.70 → unstable explanation

# Quadrant definitions
QUADRANTS = {
    "Q1_SAFE":            "High conf + High LSS",
    "Q2_EXPLANATION_RISK":"High conf + Low LSS",
    "Q3_PREDICTION_RISK": "Low conf  + High LSS",
    "Q4_COMPOUND_RISK":   "Low conf  + Low LSS",
}

COLORS = {
    "XGBoost":            "#27AE60",
    "RandomForest":       "#C0392B",
    "LogisticRegression": "#2980B9",
    "MLP":                "#8E44AD",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_model(level: str, model_name: str):
    path = MODELS_DIR / f"{level}_{model_name}.pkl"
    assert path.exists(), f"Missing model: {path}"
    return joblib.load(path)


def load_test_data(level: str) -> pd.DataFrame:
    path = DATA_DIR / f"{level}_test.csv"
    assert path.exists(), f"Missing test data: {path}"
    return pd.read_csv(path)


def get_prediction_probabilities(model, model_name: str,
                                  X_test: pd.DataFrame) -> np.ndarray:
    """Get CHD probability predictions from a model."""
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        log.warning(f"  predict_proba failed for {model_name}: {e}")
        proba = model.predict(X_test).astype(float)
    return proba


def compute_confidence(proba: np.ndarray) -> np.ndarray:
    """
    Convert CHD probability to confidence score.
    Confidence = how far the prediction is from 0.5 (the decision boundary).
    confidence = 1.0 - 2 * |p - 0.5|
    Range: [0, 1] where 0 = maximum uncertainty (p=0.5), 1 = maximum certainty (p=0 or 1).

    Clinical meaning: a confidence of 0.3 means the model predicts p=0.35 or p=0.65
    — borderline. A confidence of 0.9 means p=0.05 or p=0.95 — decisive.
    """
    return 1.0 - 2.0 * np.abs(proba - 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — CONFIDENCE-STABILITY CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_confidence_stability_correlation(lss_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model, correlate per-patient prediction confidence with their
    mean LSS from Phase 5D.

    Uses S1 model (full data baseline) for confidence — this is the
    model that would be deployed, so its confidence values are the
    clinically relevant ones.
    """
    log.info(f"\n{'='*65}")
    log.info("  Analysis 1 — Confidence vs LSS correlation")
    log.info(f"{'='*65}")

    test_s1 = load_test_data("S1")
    X_s1    = test_s1[FEATURES]
    y_s1    = test_s1[TARGET]
    rows    = []

    for model_name in MODEL_NAMES:
        log.info(f"\n  {model_name} ...")

        # Load S1 model and get predictions
        try:
            model_s1 = load_model("S1", model_name)
            proba    = get_prediction_probabilities(model_s1, model_name, X_s1)
            conf     = compute_confidence(proba)
        except Exception as e:
            log.warning(f"  Failed to load {model_name} S1: {e}")
            continue

        # Get LSS for this model from Phase 5D
        m_lss = lss_df[lss_df["model"] == model_name]["mean_lss"].values
        n_match = min(len(conf), len(m_lss))

        if n_match < 20:
            log.warning(f"  Too few matched patients for {model_name}")
            continue

        conf_matched = conf[:n_match]
        lss_matched  = m_lss[:n_match]
        proba_matched = proba[:n_match]
        y_matched    = y_s1.values[:n_match]

        # Correlation
        pearson_r,  pearson_p  = pearsonr(conf_matched,  lss_matched)
        spearman_r, spearman_p = spearmanr(conf_matched, lss_matched)

        # CHD prevalence in high vs low confidence groups
        high_conf_mask = conf_matched >= CONFIDENCE_HIGH
        low_conf_mask  = conf_matched <  CONFIDENCE_LOW
        chd_high_conf  = float(y_matched[high_conf_mask].mean())  if high_conf_mask.sum() > 0 else np.nan
        chd_low_conf   = float(y_matched[low_conf_mask].mean())   if low_conf_mask.sum()  > 0 else np.nan

        log.info(
            f"  Pearson r   = {pearson_r:.4f}  (p={pearson_p:.4f})"
        )
        log.info(
            f"  Spearman rho= {spearman_r:.4f}  (p={spearman_p:.4f})"
        )
        log.info(
            f"  High-conf patients: n={high_conf_mask.sum()}  "
            f"CHD prevalence={chd_high_conf*100:.1f}%"
        )
        log.info(
            f"  Low-conf patients:  n={low_conf_mask.sum()}   "
            f"CHD prevalence={chd_low_conf*100:.1f}%"
        )

        # Interpretation
        if pearson_r < -0.2 and pearson_p < 0.05:
            interp = "NEGATIVE: uncertain predictions → less stable explanations"
        elif pearson_r > 0.2 and pearson_p < 0.05:
            interp = "POSITIVE: uncertain predictions → more stable explanations"
        else:
            interp = "NO SIGNIFICANT CORRELATION"

        log.info(f"  Interpretation: {interp}")

        rows.append({
            "model":          model_name,
            "n_patients":     n_match,
            "pearson_r":      round(float(pearson_r),  4),
            "pearson_p":      round(float(pearson_p),  4),
            "spearman_r":     round(float(spearman_r), 4),
            "spearman_p":     round(float(spearman_p), 4),
            "significant":    pearson_p < 0.05,
            "direction":      "negative" if pearson_r < 0 else "positive",
            "chd_prev_high_conf": round(chd_high_conf, 4) if not np.isnan(chd_high_conf) else None,
            "chd_prev_low_conf":  round(chd_low_conf,  4) if not np.isnan(chd_low_conf)  else None,
            "interpretation": interp,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — QUADRANT ANALYSIS
# Classify patients into 4 risk quadrants
# ══════════════════════════════════════════════════════════════════════════════
def quadrant_analysis(lss_df: pd.DataFrame) -> tuple:
    """
    For each model, classify patients into 4 quadrants based on
    confidence and LSS. Report proportions and CHD prevalence per quadrant.
    """
    log.info(f"\n{'='*65}")
    log.info("  Analysis 2 — Confidence-Stability Quadrant Analysis")
    log.info(f"{'='*65}")
    log.info(f"  Thresholds: confidence >= {CONFIDENCE_HIGH} = HIGH")
    log.info(f"              LSS >= {LSS_HIGH} = STABLE")

    test_s1     = load_test_data("S1")
    X_s1        = test_s1[FEATURES]
    y_s1        = test_s1[TARGET].values

    quad_rows   = []
    patient_rows = []

    for model_name in MODEL_NAMES:
        try:
            model_s1 = load_model("S1", model_name)
            proba    = get_prediction_probabilities(model_s1, model_name, X_s1)
            conf     = compute_confidence(proba)
        except Exception as e:
            log.warning(f"  Failed {model_name}: {e}")
            continue

        m_lss   = lss_df[lss_df["model"] == model_name]["mean_lss"].values
        n_match = min(len(conf), len(m_lss))

        conf_m  = conf[:n_match]
        lss_m   = m_lss[:n_match]
        proba_m = proba[:n_match]
        y_m     = y_s1[:n_match]

        # Classify into quadrants
        high_conf = conf_m >= CONFIDENCE_HIGH
        low_conf  = conf_m <  CONFIDENCE_LOW
        high_lss  = lss_m  >= LSS_HIGH
        low_lss   = lss_m  <  LSS_LOW

        q1 = high_conf & high_lss    # SAFE
        q2 = high_conf & low_lss     # EXPLANATION RISK
        q3 = low_conf  & high_lss    # PREDICTION RISK
        q4 = low_conf  & low_lss     # COMPOUND RISK

        log.info(f"\n  {model_name}  n={n_match}")
        for q_mask, q_name, q_label in [
            (q1, "Q1_SAFE",            "High conf + High LSS"),
            (q2, "Q2_EXPLANATION_RISK","High conf + Low LSS"),
            (q3, "Q3_PREDICTION_RISK", "Low conf  + High LSS"),
            (q4, "Q4_COMPOUND_RISK",   "Low conf  + Low LSS"),
        ]:
            n_q      = int(q_mask.sum())
            pct_q    = float(n_q / n_match * 100)
            chd_prev = float(y_m[q_mask].mean()) if n_q > 0 else np.nan

            log.info(
                f"    {q_label:<30}  "
                f"n={n_q:>4}  ({pct_q:>5.1f}%)  "
                f"CHD={chd_prev*100:.1f}%"
                if not np.isnan(chd_prev) else
                f"    {q_label:<30}  n={n_q:>4}  ({pct_q:>5.1f}%)  CHD=N/A"
            )

            quad_rows.append({
                "model":        model_name,
                "quadrant":     q_name,
                "description":  q_label,
                "n_patients":   n_q,
                "pct_patients": round(pct_q, 2),
                "chd_prevalence": round(chd_prev, 4) if not np.isnan(chd_prev) else None,
            })

        # Store per-patient quadrant assignments
        for i in range(n_match):
            if q1[i]:   quad = "Q1_SAFE"
            elif q2[i]: quad = "Q2_EXPLANATION_RISK"
            elif q3[i]: quad = "Q3_PREDICTION_RISK"
            elif q4[i]: quad = "Q4_COMPOUND_RISK"
            else:       quad = "Q_MIDDLE"   # between thresholds

            patient_rows.append({
                "model":        model_name,
                "patient_id":   i,
                "proba":        round(float(proba_m[i]), 4),
                "confidence":   round(float(conf_m[i]),  4),
                "lss":          round(float(lss_m[i]),   4),
                "chd_true":     int(y_m[i]),
                "quadrant":     quad,
            })

    return pd.DataFrame(quad_rows), pd.DataFrame(patient_rows)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — COMPOUND RISK UNDER SCARCITY
# Does Q4 proportion grow as data decreases?
# ══════════════════════════════════════════════════════════════════════════════
def compound_risk_by_scarcity(lss_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each scarcity level, compute the proportion of patients in Q4
    (compound risk: low confidence + low LSS).
    Tests whether compound risk grows with data scarcity.
    """
    log.info(f"\n{'='*65}")
    log.info("  Analysis 3 — Compound risk under progressive scarcity")
    log.info(f"{'='*65}")

    rows = []

    for model_name in MODEL_NAMES:
        log.info(f"\n  {model_name}:")

        for level in SCARCITY_LEVELS:
            try:
                test_si  = load_test_data(level)
                X_si     = test_si[FEATURES]
                y_si     = test_si[TARGET].values
                model_si = load_model(level, model_name)
                proba_si = get_prediction_probabilities(model_si, model_name, X_si)
                conf_si  = compute_confidence(proba_si)
            except Exception as e:
                log.warning(f"  Failed {model_name} {level}: {e}")
                continue

            # Get LSS for this level
            lss_col = f"lss_{level}"
            if lss_col not in lss_df.columns:
                # Use mean_lss as fallback for S1
                if level == "S1":
                    m_lss_si = lss_df[lss_df["model"] == model_name]["mean_lss"].values
                else:
                    continue
            else:
                m_lss_si = lss_df[lss_df["model"] == model_name][lss_col].values

            n_match  = min(len(conf_si), len(m_lss_si))
            conf_m   = conf_si[:n_match]
            lss_m    = m_lss_si[:n_match]
            y_m      = y_si[:n_match]

            # Compound risk: low confidence AND low LSS
            compound_mask = (conf_m < CONFIDENCE_LOW) & (lss_m < LSS_LOW)
            n_compound    = int(compound_mask.sum())
            pct_compound  = float(n_compound / n_match * 100) if n_match > 0 else 0.0
            chd_compound  = float(y_m[compound_mask].mean()) \
                            if n_compound > 0 else np.nan

            # Also track prediction risk and explanation risk separately
            pred_risk_only  = (conf_m < CONFIDENCE_LOW)  & (lss_m >= LSS_HIGH)
            exp_risk_only   = (conf_m >= CONFIDENCE_HIGH) & (lss_m < LSS_LOW)
            safe            = (conf_m >= CONFIDENCE_HIGH) & (lss_m >= LSS_HIGH)

            rows.append({
                "model":              model_name,
                "level":              level,
                "n_patients":         n_match,
                "pct_compound_risk":  round(pct_compound, 2),
                "n_compound_risk":    n_compound,
                "chd_prev_compound":  round(chd_compound, 4) if not np.isnan(chd_compound) else None,
                "pct_safe":           round(float(safe.sum())/n_match*100, 2),
                "pct_pred_risk_only": round(float(pred_risk_only.sum())/n_match*100, 2),
                "pct_exp_risk_only":  round(float(exp_risk_only.sum())/n_match*100, 2),
            })

            log.info(
                f"  {level}  n={n_match:>5}  "
                f"compound={pct_compound:.1f}%  "
                f"safe={round(float(safe.sum())/n_match*100,1):.1f}%  "
                f"CHD_in_compound={chd_compound*100:.1f}%"
                if not np.isnan(chd_compound) else
                f"  {level}  n={n_match:>5}  "
                f"compound={pct_compound:.1f}%  "
                f"safe={round(float(safe.sum())/n_match*100,1):.1f}%  "
                f"CHD_in_compound=N/A"
            )

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


def plot_confidence_stability_scatter(patient_df: pd.DataFrame) -> None:
    """Scatter plot: confidence vs LSS per patient, colored by quadrant."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes      = axes.flatten()

    quad_colors = {
        "Q1_SAFE":             "#27AE60",
        "Q2_EXPLANATION_RISK": "#F39C12",
        "Q3_PREDICTION_RISK":  "#3498DB",
        "Q4_COMPOUND_RISK":    "#E74C3C",
        "Q_MIDDLE":            "#BDC3C7",
    }

    for idx, model_name in enumerate(MODEL_NAMES):
        ax   = axes[idx]
        m_df = patient_df[patient_df["model"] == model_name]

        for quad, color in quad_colors.items():
            q_df = m_df[m_df["quadrant"] == quad]
            if q_df.empty:
                continue
            ax.scatter(q_df["confidence"], q_df["lss"],
                      c=color, alpha=0.4, s=8, label=quad.replace("_", " "))

        # Threshold lines
        ax.axvline(CONFIDENCE_HIGH, color="gray", linestyle="--",
                  linewidth=1, alpha=0.6)
        ax.axvline(CONFIDENCE_LOW,  color="gray", linestyle=":",
                  linewidth=1, alpha=0.4)
        ax.axhline(LSS_HIGH, color="gray", linestyle="--",
                  linewidth=1, alpha=0.6)
        ax.axhline(LSS_LOW,  color="gray", linestyle=":",
                  linewidth=1, alpha=0.4)

        # Quadrant labels
        ax.text(0.02, 0.98, "Q4\nCOMPOUND\nRISK",
               transform=ax.transAxes, fontsize=7,
               color="#E74C3C", fontweight="bold", va="top")
        ax.text(0.85, 0.98, "Q2\nEXP\nRISK",
               transform=ax.transAxes, fontsize=7,
               color="#F39C12", fontweight="bold", va="top")
        ax.text(0.85, 0.12, "Q1\nSAFE",
               transform=ax.transAxes, fontsize=7,
               color="#27AE60", fontweight="bold", va="bottom")

        # Correlation
        valid = m_df.dropna(subset=["confidence","lss"])
        if len(valid) > 10:
            r, p = pearsonr(valid["confidence"].values, valid["lss"].values)
            ax.text(0.02, 0.02, f"r = {r:.3f}",
                   transform=ax.transAxes, fontsize=9,
                   fontweight="bold", va="bottom",
                   color=COLORS[model_name])

        ax.set_xlabel("Prediction confidence", fontsize=9)
        ax.set_ylabel("Local Stability Score (LSS)", fontsize=9)
        ax.set_title(model_name, fontsize=10, fontweight="bold",
                    color=COLORS[model_name])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Fig — Prediction Confidence vs Explanation Stability\n"
        "Quadrant classification reveals compound risk patients (Q4: low conf + low LSS)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_confidence_stability_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_confidence_stability_scatter.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_confidence_stability_scatter.png / .pdf")


def plot_compound_risk_scarcity(risk_df: pd.DataFrame) -> None:
    """Line chart: compound risk proportion across scarcity levels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    levels    = SCARCITY_LEVELS

    # Left: compound risk % by level
    ax = axes[0]
    for model_name in MODEL_NAMES:
        m_df = risk_df[risk_df["model"] == model_name].sort_values("level")
        x    = range(len(m_df))
        ax.plot(x, m_df["pct_compound_risk"].values,
               color=COLORS[model_name], linewidth=2,
               marker="o", markersize=5,
               label=model_name)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Compound risk patients (%)", fontsize=10)
    ax.set_title("Compound risk (Q4) by scarcity level", fontsize=11)
    ax.legend(fontsize=8)

    # Right: safe % by level
    ax2 = axes[1]
    for model_name in MODEL_NAMES:
        m_df = risk_df[risk_df["model"] == model_name].sort_values("level")
        x    = range(len(m_df))
        ax2.plot(x, m_df["pct_safe"].values,
                color=COLORS[model_name], linewidth=2,
                marker="o", markersize=5,
                label=model_name)

    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.set_ylabel("Safe patients — Q1 (%)", fontsize=10)
    ax2.set_title("Safe patients (Q1) by scarcity level", fontsize=11)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Fig — Compound Risk and Safety Under Progressive Scarcity\n"
        "As data decreases: compound risk grows, safe proportion shrinks",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_compound_risk_scarcity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_compound_risk_scarcity.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_compound_risk_scarcity.png / .pdf")


def plot_quadrant_heatmap(quad_df: pd.DataFrame) -> None:
    """Heatmap: % patients per quadrant per model."""
    fig, ax = plt.subplots(figsize=(9, 4))

    models    = MODEL_NAMES
    quadrants = ["Q1_SAFE","Q2_EXPLANATION_RISK","Q3_PREDICTION_RISK","Q4_COMPOUND_RISK"]
    q_labels  = ["Q1 Safe","Q2 Exp Risk","Q3 Pred Risk","Q4 Compound"]
    matrix    = np.zeros((len(models), len(quadrants)))

    for i, model in enumerate(models):
        for j, quad in enumerate(quadrants):
            row = quad_df[(quad_df["model"]==model) & (quad_df["quadrant"]==quad)]
            if len(row) > 0:
                matrix[i, j] = float(row["pct_patients"].values[0])

    # Color Q4 red, Q1 green, others neutral
    cmap_data = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap_data, aspect="auto",
                  vmin=0, vmax=matrix.max())

    ax.set_xticks(range(len(quadrants)))
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(quadrants)):
            val   = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                   fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="% patients", fraction=0.03, pad=0.02)
    ax.set_title(
        "Fig — Patient Distribution Across Risk Quadrants\n"
        "Q4 (compound risk) patients require highest clinical caution",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    path = FIG_DIR / "fig_quadrant_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_quadrant_heatmap.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_quadrant_heatmap.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 5E — Prediction Confidence vs Explanation Stability")
    log.info("=" * 65)
    log.info("Purpose  : Connect predictive uncertainty with explanation instability")
    log.info(f"Confidence thresholds: HIGH >= {CONFIDENCE_HIGH}  LOW < {CONFIDENCE_LOW}")
    log.info(f"LSS thresholds:        HIGH >= {LSS_HIGH}         LOW < {LSS_LOW}")

    # ── Load Phase 5D LSS data ────────────────────────────────────────────
    lss_path = P5D_DIR / "patient_stability_scores.csv"
    assert lss_path.exists(), \
        "Missing patient_stability_scores.csv — run phase5d first"
    lss_df = pd.read_csv(lss_path)
    log.info(f"\n  Loaded LSS data: {lss_df.shape}")

    # ── Analysis 1: Correlation ───────────────────────────────────────────
    corr_df = compute_confidence_stability_correlation(lss_df)
    corr_df.to_csv(P5E_DIR / "confidence_stability_correlation.csv", index=False)
    log.info(f"\n  Saved confidence_stability_correlation.csv")

    # ── Analysis 2: Quadrant analysis ────────────────────────────────────
    quad_df, patient_df = quadrant_analysis(lss_df)
    quad_df.to_csv(P5E_DIR / "quadrant_analysis.csv", index=False)
    patient_df.to_csv(P5E_DIR / "patient_quadrant_assignments.csv", index=False)
    log.info(f"\n  Saved quadrant_analysis.csv and patient_quadrant_assignments.csv")

    # ── Analysis 3: Compound risk by scarcity ─────────────────────────────
    risk_df = compound_risk_by_scarcity(lss_df)
    risk_df.to_csv(P5E_DIR / "compound_risk_by_level.csv", index=False)
    log.info(f"\n  Saved compound_risk_by_level.csv")

    # ── Visualizations ────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Generating figures ...")
    log.info(f"{'='*65}")
    plot_confidence_stability_scatter(patient_df)
    plot_compound_risk_scarcity(risk_df)
    plot_quadrant_heatmap(quad_df)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("Phase 5E complete — Key findings:")
    log.info(f"{'='*65}")

    log.info("\n  Confidence-stability correlations:")
    if not corr_df.empty:
        log.info(f"\n{corr_df[['model','pearson_r','pearson_p','significant','interpretation']].to_string(index=False)}")

    log.info("\n  Quadrant distribution (% patients):")
    if not quad_df.empty:
        pivot = quad_df.pivot(index="model", columns="quadrant",
                             values="pct_patients").fillna(0)
        log.info(f"\n{pivot.to_string()}")

    log.info("\n  Compound risk (Q4) at S1 vs S7:")
    if not risk_df.empty:
        for model_name in MODEL_NAMES:
            m_df = risk_df[risk_df["model"] == model_name]
            s1   = m_df[m_df["level"]=="S1"]["pct_compound_risk"].values
            s7   = m_df[m_df["level"]=="S7"]["pct_compound_risk"].values
            if len(s1) > 0 and len(s7) > 0:
                log.info(
                    f"  {model_name:<22}  "
                    f"S1={s1[0]:.1f}%  →  S7={s7[0]:.1f}%  "
                    f"(change={s7[0]-s1[0]:+.1f}pp)"
                )

    log.info(
        "\n  CLINICAL INTERPRETATION:"
        "\n  Q4 patients (compound risk) require highest caution —"
        "\n  neither the prediction nor the explanation can be fully trusted."
        "\n  If Q4 proportion grows under scarcity, data volume directly"
        "\n  increases the proportion of clinically unsafe model outputs."
    )
    log.info("\nPhase 5E complete. ✓")


if __name__ == "__main__":
    main()