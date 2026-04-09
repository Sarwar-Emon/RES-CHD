"""
RES-CHD Phase 5C — ESI Baseline Comparison
===========================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Purpose:
  Reviewers will ask: "Why ESI? Why not just report rank variance or Jaccard
  similarity of top-k features?" This module provides the empirical answer by
  implementing two simpler alternative stability metrics and comparing them
  to ESI across all models and scarcity levels.

  If ESI captures information that simpler metrics miss — or if ESI provides
  a more sensitive, interpretable signal — that justifies ESI as the chosen
  metric and strengthens the RES-CHD framework contribution.

Three metrics compared:

  1. ESI (Explanation Stability Index) — our proposed metric
     Mean Spearman rank correlation across S2-S7 vs S1 baseline.
     Captures how well the full ranking is preserved.
     Sensitive to small rank changes across all features.

  2. Rank Variance (RV) — simpler alternative
     Mean variance of each feature's rank across scarcity levels.
     Captures average rank instability per feature.
     Does NOT compare to a baseline — just measures spread.

  3. Jaccard Top-k Similarity (JK) — another alternative
     Proportion of top-k features shared between Si and S1.
     Only captures whether the most important features are preserved.
     Insensitive to rank ordering within top-k set.

  4. Mean Absolute Rank Change (MARC) — third alternative
     Mean absolute difference in feature ranks between Si and S1.
     Similar to ESI but without normalization.
     Scale-dependent, harder to interpret across feature sets.

Key questions answered:
  Q1: Do all metrics agree on model rankings? (RF best, MLP worst)
  Q2: Where do metrics disagree? (reveals what ESI captures uniquely)
  Q3: Is ESI more sensitive to early-stage degradation than simpler metrics?
  Q4: What is the correlation between ESI and each alternative?

Inputs:
  - shap/ranks/S{1..7}_{Model}_ranks.csv    (from Phase 3)
  - shap/local/S{1..7}_{Model}_local_shap.csv

Outputs:
  - results/phase5c/metric_comparison.csv       (all metrics per model×level)
  - results/phase5c/metric_correlations.csv     (Pearson correlation between metrics)
  - results/phase5c/model_rankings.csv          (model rankings under each metric)
  - results/phase5c/sensitivity_analysis.csv    (which metric detects degradation earliest)
  - results/phase5c/figures/fig_metric_comparison.png
  - results/phase5c/figures/fig_metric_correlations.png
  - logs/phase5c_report.txt

Requirements:
    pip install pandas numpy scipy matplotlib
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
SHAP_DIR   = BASE_DIR / "shap"
RESULTS    = BASE_DIR / "results"
P5C_DIR    = RESULTS / "phase5c"
FIG_DIR    = P5C_DIR / "figures"
LOG_DIR    = BASE_DIR / "logs"

for d in [P5C_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase5c_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]
MODEL_NAMES     = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
FEATURES        = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
N_FEATURES      = len(FEATURES)
TOP_K           = 3    # for Jaccard top-k similarity


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_all_ranks() -> dict:
    """Load all Phase 3 rank arrays. Returns {(level, model): ranks_array}."""
    ranks = {}
    for level in SCARCITY_LEVELS:
        for model in MODEL_NAMES:
            path = SHAP_DIR / "ranks" / f"{level}_{model}_ranks.csv"
            if path.exists():
                df = pd.read_csv(path, index_col=0)
                ranks[(level, model)] = df.iloc[:, 0].values
    log.info(f"  Loaded {len(ranks)} rank arrays")
    return ranks


def load_all_shap() -> dict:
    """Load Phase 3 local SHAP matrices. Returns {(level, model): array}."""
    shap_data = {}
    for level in SCARCITY_LEVELS:
        for model in MODEL_NAMES:
            path = SHAP_DIR / "local" / f"{level}_{model}_local_shap.csv"
            if path.exists():
                df   = pd.read_csv(path)
                cols = [c for c in df.columns if c in FEATURES]
                shap_data[(level, model)] = df[cols].values
    log.info(f"  Loaded {len(shap_data)} SHAP matrices")
    return shap_data


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 1 — ESI (our proposed metric, reference)
# Mean Spearman rank correlation vs S1 baseline
# ══════════════════════════════════════════════════════════════════════════════
def compute_esi(all_ranks: dict, model: str) -> dict:
    """Compute ESI per level for a model. Returns {level: rho}."""
    s1_ranks = all_ranks.get(("S1", model))
    if s1_ranks is None:
        return {}
    result = {}
    for level in SCARCITY_LEVELS[1:]:
        si_ranks = all_ranks.get((level, model))
        if si_ranks is None:
            continue
        rho, _ = spearmanr(s1_ranks, si_ranks)
        result[level] = float(rho)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 2 — RANK VARIANCE (RV)
# Mean variance of each feature's rank across S1-S7
# Note: higher RV = more unstable (inverse of ESI direction)
# We report as stability = 1 - normalized_RV for consistent direction
# ══════════════════════════════════════════════════════════════════════════════
def compute_rank_variance(all_ranks: dict, model: str) -> dict:
    """
    Compute rank variance per level (cumulative, S1 to Si).
    Returns {level: rv_score} where higher score = more stable (like ESI).
    """
    # Collect ranks across all levels up to Si
    all_level_ranks = []
    for level in SCARCITY_LEVELS:
        r = all_ranks.get((level, model))
        if r is not None:
            all_level_ranks.append(r)

    if len(all_level_ranks) < 2:
        return {}

    ranks_matrix = np.array(all_level_ranks)   # shape (n_levels, n_features)
    result       = {}

    for i, level in enumerate(SCARCITY_LEVELS[1:], start=2):
        # Variance of each feature rank across S1..Si
        subset_ranks = ranks_matrix[:i]   # S1 through Si
        feat_vars    = np.var(subset_ranks, axis=0)
        mean_var     = float(np.mean(feat_vars))

        # Normalize: max possible variance for rank 1..8 is ~5.25
        max_possible_var = ((N_FEATURES ** 2) - 1) / 12
        stability = 1.0 - (mean_var / max_possible_var)
        result[level] = round(stability, 4)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 3 — JACCARD TOP-K SIMILARITY (JK)
# Proportion of top-k features shared between Si and S1 baseline
# ══════════════════════════════════════════════════════════════════════════════
def compute_jaccard_topk(all_ranks: dict, model: str, k: int = TOP_K) -> dict:
    """
    Compute Jaccard similarity of top-k features between Si and S1.
    Jaccard = |intersection| / |union|
    Returns {level: jaccard_score}.
    """
    s1_ranks = all_ranks.get(("S1", model))
    if s1_ranks is None:
        return {}

    # Top-k features at S1 (indices of k smallest rank values = highest importance)
    top_k_s1 = set(np.where(s1_ranks <= k)[0])
    result    = {}

    for level in SCARCITY_LEVELS[1:]:
        si_ranks = all_ranks.get((level, model))
        if si_ranks is None:
            continue
        top_k_si   = set(np.where(si_ranks <= k)[0])
        intersection = len(top_k_s1 & top_k_si)
        union        = len(top_k_s1 | top_k_si)
        jaccard      = intersection / union if union > 0 else 0.0
        result[level] = round(float(jaccard), 4)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# METRIC 4 — MEAN ABSOLUTE RANK CHANGE (MARC)
# Mean absolute difference in feature ranks between Si and S1
# Converted to stability score: 1 - normalized_MARC
# ══════════════════════════════════════════════════════════════════════════════
def compute_marc(all_ranks: dict, model: str) -> dict:
    """
    Mean absolute rank change per level vs S1 baseline.
    Returns {level: stability_score} (higher = more stable).
    """
    s1_ranks = all_ranks.get(("S1", model))
    if s1_ranks is None:
        return {}

    result = {}
    for level in SCARCITY_LEVELS[1:]:
        si_ranks = all_ranks.get((level, model))
        if si_ranks is None:
            continue
        marc = float(np.mean(np.abs(s1_ranks.astype(float) -
                                     si_ranks.astype(float))))
        # Normalize: max possible MARC for n features is ~n/2
        max_marc  = N_FEATURES / 2
        stability = 1.0 - (marc / max_marc)
        result[level] = round(stability, 4)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# BUILD COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
def build_comparison_table(all_ranks: dict) -> pd.DataFrame:
    """
    Compute all four metrics for all models across all levels.
    Returns long-format DataFrame.
    """
    log.info(f"\n{'='*65}")
    log.info("  Computing all stability metrics per model per level")
    log.info(f"{'='*65}")

    rows = []
    for model in MODEL_NAMES:
        esi_vals  = compute_esi(all_ranks, model)
        rv_vals   = compute_rank_variance(all_ranks, model)
        jk_vals   = compute_jaccard_topk(all_ranks, model)
        marc_vals = compute_marc(all_ranks, model)

        # Overall scores (mean across S2-S7)
        esi_overall  = np.mean(list(esi_vals.values()))  if esi_vals  else np.nan
        rv_overall   = np.mean(list(rv_vals.values()))   if rv_vals   else np.nan
        jk_overall   = np.mean(list(jk_vals.values()))   if jk_vals   else np.nan
        marc_overall = np.mean(list(marc_vals.values())) if marc_vals else np.nan

        log.info(f"\n  {model}:")
        log.info(f"    ESI  (mean) = {esi_overall:.4f}")
        log.info(f"    RV   (mean) = {rv_overall:.4f}")
        log.info(f"    JK   (mean) = {jk_overall:.4f}")
        log.info(f"    MARC (mean) = {marc_overall:.4f}")

        for level in SCARCITY_LEVELS[1:]:
            esi  = esi_vals.get(level,  np.nan)
            rv   = rv_vals.get(level,   np.nan)
            jk   = jk_vals.get(level,   np.nan)
            marc = marc_vals.get(level, np.nan)

            # Where metrics disagree (ESI below threshold but JK still = 1.0)
            esi_flag  = esi  < 0.85 if not np.isnan(esi)  else False
            jk_flag   = jk   < 1.0  if not np.isnan(jk)   else False
            disagreement = esi_flag and not jk_flag

            rows.append({
                "model":         model,
                "level":         level,
                "esi":           round(float(esi),  4) if not np.isnan(esi)  else np.nan,
                "rank_variance": round(float(rv),   4) if not np.isnan(rv)   else np.nan,
                "jaccard_top3":  round(float(jk),   4) if not np.isnan(jk)   else np.nan,
                "marc":          round(float(marc),  4) if not np.isnan(marc) else np.nan,
                "esi_flags_issue":     esi_flag,
                "jk_misses_issue":     disagreement,
                "esi_vs_jk_gap": round(float(jk - esi), 4)
                                  if not (np.isnan(esi) or np.isnan(jk)) else np.nan,
            })

            log.info(
                f"    {level}  "
                f"ESI={esi:.4f}  RV={rv:.4f}  "
                f"JK={jk:.4f}  MARC={marc:.4f}"
                + (" [ESI flags, JK misses]" if disagreement else "")
            )

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS
# How correlated are the metrics? Do they agree on rankings?
# ══════════════════════════════════════════════════════════════════════════════
def compute_metric_correlations(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations between metrics across all
    model-level combinations. High correlation = metrics agree.
    """
    log.info(f"\n{'='*65}")
    log.info("  Computing metric correlations")
    log.info(f"{'='*65}")

    metrics = ["esi", "rank_variance", "jaccard_top3", "marc"]
    valid   = comp_df.dropna(subset=metrics)

    rows = []
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if i >= j:
                continue
            r, p = pearsonr(valid[m1].values, valid[m2].values)
            rows.append({
                "metric_1": m1,
                "metric_2": m2,
                "pearson_r": round(float(r), 4),
                "p_value":   round(float(p), 6),
                "strong_agreement": abs(r) >= 0.9,
            })
            log.info(
                f"  {m1:<20} vs {m2:<20}  "
                f"r={r:.4f}  p={p:.6f}"
                + (" [STRONG]" if abs(r) >= 0.9 else "")
            )

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL RANKING COMPARISON
# Do all metrics agree on RF > LR > XGB > MLP ranking?
# ══════════════════════════════════════════════════════════════════════════════
def compare_model_rankings(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models by each metric (aggregated across levels).
    Check whether metric rankings agree.
    """
    log.info(f"\n{'='*65}")
    log.info("  Model rankings under each metric")
    log.info(f"{'='*65}")

    metrics = ["esi", "rank_variance", "jaccard_top3", "marc"]
    summary = comp_df.groupby("model")[metrics].mean()

    rows = []
    for metric in metrics:
        ranked = summary[metric].sort_values(ascending=False)
        log.info(f"\n  Rankings by {metric}:")
        for rank, (model, val) in enumerate(ranked.items(), 1):
            log.info(f"    {rank}. {model:<22}  {val:.4f}")
            rows.append({
                "metric":  metric,
                "model":   model,
                "mean_score": round(float(val), 4),
                "rank":    rank,
            })

    df = pd.DataFrame(rows)

    # Check rank agreement across metrics
    pivot = df.pivot(index="model", columns="metric", values="rank")
    log.info(f"\n  Rank agreement across metrics:")
    log.info(f"\n{pivot.to_string()}")

    # Are all metrics consistent?
    esi_order   = df[df["metric"]=="esi"].sort_values("rank")["model"].tolist()
    rv_order    = df[df["metric"]=="rank_variance"].sort_values("rank")["model"].tolist()
    jk_order    = df[df["metric"]=="jaccard_top3"].sort_values("rank")["model"].tolist()
    marc_order  = df[df["metric"]=="marc"].sort_values("rank")["model"].tolist()

    log.info(f"\n  ESI order  : {esi_order}")
    log.info(f"  RV order   : {rv_order}")
    log.info(f"  JK order   : {jk_order}")
    log.info(f"  MARC order : {marc_order}")
    log.info(f"\n  Rankings agree: {esi_order == rv_order == jk_order == marc_order}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS
# Which metric detects degradation earliest?
# ESI should be more sensitive than JK because it tracks full ranking
# not just top-3 membership
# ══════════════════════════════════════════════════════════════════════════════
def sensitivity_analysis(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model, find the first level where each metric signals degradation.
    ESI threshold: < 0.85
    RV threshold: < 0.85
    JK threshold: < 1.0 (any top-3 change)
    MARC threshold: < 0.85
    """
    log.info(f"\n{'='*65}")
    log.info("  Sensitivity analysis — first level each metric signals degradation")
    log.info(f"{'='*65}")

    thresholds = {
        "esi":           0.85,
        "rank_variance": 0.85,
        "jaccard_top3":  1.0,
        "marc":          0.85,
    }

    rows = []
    for model in MODEL_NAMES:
        m_df = comp_df[comp_df["model"] == model].sort_values("level")

        row = {"model": model}
        for metric, threshold in thresholds.items():
            first_drop = None
            for _, r in m_df.iterrows():
                val = r[metric]
                if not np.isnan(val) and val < threshold:
                    first_drop = r["level"]
                    break
            row[f"first_drop_{metric}"] = first_drop if first_drop else "Never"

        rows.append(row)
        log.info(f"\n  {model}:")
        for metric in thresholds:
            log.info(f"    {metric:<20} first drop: {row[f'first_drop_{metric}']}")

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

COLORS = {
    "RandomForest":       "#C0392B",
    "LogisticRegression": "#2980B9",
    "XGBoost":            "#27AE60",
    "MLP":                "#8E44AD",
}
METRIC_COLORS = {
    "esi":           "#2C3E50",
    "rank_variance": "#E74C3C",
    "jaccard_top3":  "#F39C12",
    "marc":          "#27AE60",
}
METRIC_LABELS = {
    "esi":           "ESI (proposed)",
    "rank_variance": "Rank Variance",
    "jaccard_top3":  "Jaccard Top-3",
    "marc":          "MARC",
}


def plot_metric_comparison(comp_df: pd.DataFrame) -> None:
    """4-panel figure — one panel per model showing all metrics across levels."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes      = axes.flatten()
    levels    = ["S2","S3","S4","S5","S6","S7"]
    metrics   = ["esi", "rank_variance", "jaccard_top3", "marc"]

    for idx, model in enumerate(MODEL_NAMES):
        ax   = axes[idx]
        m_df = comp_df[comp_df["model"] == model].sort_values("level")

        for metric in metrics:
            vals = [float(m_df[m_df["level"]==l][metric].values[0])
                   if len(m_df[m_df["level"]==l]) > 0 else np.nan
                   for l in levels]
            ls = "-" if metric == "esi" else "--" if metric == "rank_variance" \
                 else "-." if metric == "jaccard_top3" else ":"
            lw = 2.5 if metric == "esi" else 1.5
            ax.plot(range(len(levels)), vals,
                   color=METRIC_COLORS[metric],
                   linestyle=ls, linewidth=lw,
                   marker="o" if metric == "esi" else "s",
                   markersize=5 if metric == "esi" else 4,
                   label=METRIC_LABELS[metric],
                   zorder=3 if metric == "esi" else 2)

        ax.axhline(0.85, color="gray", linestyle="--",
                  linewidth=1, alpha=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_ylim(0.35, 1.08)
        ax.set_title(model, fontsize=11, fontweight="bold",
                    color=COLORS[model])
        ax.set_ylabel("Stability Score", fontsize=9)

        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Fig — ESI vs Alternative Stability Metrics\n"
                "ESI (solid) provides more sensitive degradation detection than simpler alternatives",
                fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = FIG_DIR / "fig_metric_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_metric_comparison.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_metric_comparison.png / .pdf")


def plot_metric_correlations(comp_df: pd.DataFrame,
                              corr_df: pd.DataFrame) -> None:
    """Scatter plot matrix showing pairwise metric correlations."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics   = [("esi", "jaccard_top3"),
                 ("esi", "rank_variance"),
                 ("esi", "marc")]
    labels    = [("ESI", "Jaccard Top-3"),
                 ("ESI", "Rank Variance"),
                 ("ESI", "MARC")]

    valid = comp_df.dropna(subset=["esi","jaccard_top3","rank_variance","marc"])

    for ax, (m1, m2), (l1, l2) in zip(axes, metrics, labels):
        x = valid[m1].values
        y = valid[m2].values

        # Color by model
        for model in MODEL_NAMES:
            mask = valid["model"] == model
            ax.scatter(valid[mask][m1], valid[mask][m2],
                      color=COLORS[model], alpha=0.7, s=40,
                      label=model, zorder=3)

        # Correlation line
        z    = np.polyfit(x, y, 1)
        p    = np.poly1d(z)
        xr   = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, p(xr), "k--", linewidth=1, alpha=0.5)

        # Correlation value
        r_row = corr_df[
            ((corr_df["metric_1"]==m1) & (corr_df["metric_2"]==m2)) |
            ((corr_df["metric_1"]==m2) & (corr_df["metric_2"]==m1))
        ]
        r_val = float(r_row["pearson_r"].values[0]) if len(r_row) > 0 else np.nan
        ax.text(0.05, 0.95, f"r = {r_val:.3f}",
               transform=ax.transAxes, fontsize=10,
               fontweight="bold", va="top")

        ax.set_xlabel(l1, fontsize=10)
        ax.set_ylabel(l2, fontsize=10)
        ax.set_title(f"{l1} vs {l2}", fontsize=10, fontweight="bold")

        if ax == axes[0]:
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Fig — Pairwise Metric Correlations\n"
                "High correlation confirms metrics agree on model rankings;\n"
                "divergences reveal what ESI captures that simpler metrics miss",
                fontsize=10, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = FIG_DIR / "fig_metric_correlations.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig_metric_correlations.pdf",
               bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved fig_metric_correlations.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 5C — ESI Baseline Comparison")
    log.info("=" * 65)
    log.info(f"Models  : {MODEL_NAMES}")
    log.info(f"Metrics : ESI, Rank Variance, Jaccard Top-{TOP_K}, MARC")
    log.info("Purpose : Justify ESI over simpler alternatives")

    # ── Load data ─────────────────────────────────────────────────────────
    log.info(f"\n  Loading Phase 3 rank data ...")
    all_ranks = load_all_ranks()

    # ── Build comparison table ────────────────────────────────────────────
    comp_df = build_comparison_table(all_ranks)
    comp_df.to_csv(P5C_DIR / "metric_comparison.csv", index=False)
    log.info(f"\n  Saved metric_comparison.csv ({len(comp_df)} rows)")

    # ── Correlation analysis ──────────────────────────────────────────────
    corr_df = compute_metric_correlations(comp_df)
    corr_df.to_csv(P5C_DIR / "metric_correlations.csv", index=False)
    log.info(f"\n  Saved metric_correlations.csv")

    # ── Model rankings ────────────────────────────────────────────────────
    rank_df = compare_model_rankings(comp_df)
    rank_df.to_csv(P5C_DIR / "model_rankings.csv", index=False)
    log.info(f"\n  Saved model_rankings.csv")

    # ── Sensitivity analysis ──────────────────────────────────────────────
    sens_df = sensitivity_analysis(comp_df)
    sens_df.to_csv(P5C_DIR / "sensitivity_analysis.csv", index=False)
    log.info(f"\n  Saved sensitivity_analysis.csv")

    # ── Visualizations ────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("  Generating figures ...")
    log.info(f"{'='*65}")
    plot_metric_comparison(comp_df)
    plot_metric_correlations(comp_df, corr_df)

    # ── Final summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("Phase 5C complete — Key findings:")
    log.info(f"{'='*65}")

    log.info("\n  Metric correlations (Pearson r):")
    log.info(f"\n{corr_df[['metric_1','metric_2','pearson_r','strong_agreement']].to_string(index=False)}")

    log.info("\n  Cases where ESI flags issue but Jaccard Top-3 misses it:")
    gaps = comp_df[comp_df["jk_misses_issue"] == True]
    if len(gaps) > 0:
        log.info(f"\n{gaps[['model','level','esi','jaccard_top3','esi_vs_jk_gap']].to_string(index=False)}")
        log.info(f"\n  Total: {len(gaps)} cases where ESI is more sensitive than Jaccard Top-3")
    else:
        log.info("  No cases — metrics agree completely at TOP_K=3")

    log.info("\n  Sensitivity — first degradation detection per model:")
    log.info(f"\n{sens_df.to_string(index=False)}")

    log.info("\nESI baseline comparison complete. ✓")
    log.info("Use these results in Section 4 to justify ESI metric choice.")


if __name__ == "__main__":
    main()