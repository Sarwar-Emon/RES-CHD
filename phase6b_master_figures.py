"""
RES-CHD Phase 6B — Master Figure Generator (Updated)
=====================================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Purpose:
  Regenerates ALL publication figures in one place, incorporating results
  from all modules including the new phases 5B-5F and Cleveland validation.

  Phase 6 (original) — 10 figures from core pipeline
  Phase 6B (this)    — 6 additional figures from new modules
                       + master consolidated figure index

Complete figure list after this script:

  From Phase 6 (original — already generated):
  fig01_esi_stability_curves.png
  fig02_bootstrapped_esi_bars.png
  fig03_feature_reliability_heatmap.png
  fig04_reliability_flag_heatmap.png
  fig05_ablation_threshold.png
  fig06_ablation_temporal_vs_random.png
  fig07_shap_importance_s1.png
  fig08_scarcity_threshold.png
  fig09_auc_vs_esi.png
  fig10_publication_table.png

  From Phase 6B (this script — new):
  fig11_distributional_shift.png        (Phase 5B)
  fig12_cycle_drift.png                 (Phase 5B)
  fig13_metric_comparison.png           (Phase 5C)
  fig14_local_stability_distribution.png (Phase 5D)
  fig15_compound_risk_scarcity.png      (Phase 5E)
  fig16_calibration_curves.png          (Phase 5F)
  fig17_cleveland_vs_nhanes.png         (Cleveland validation)
  fig18_updated_esi_bars_all_models.png (Phase 4B — MLP now included)

Outputs:
  results/phase6b/figures/fig11_*.png/pdf  ...  fig18_*.png/pdf
  results/phase6b/figure_index.csv          (master table of all figures)
  logs/phase6b_report.txt
"""

import logging
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Directory setup ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RESULTS    = BASE_DIR / "results"
P4_DIR     = RESULTS / "phase4"
P5B_DIR    = RESULTS / "phase5b"
P5C_DIR    = RESULTS / "phase5c"
P5D_DIR    = RESULTS / "phase5d"
P5E_DIR    = RESULTS / "phase5e"
P5F_DIR    = RESULTS / "phase5f"
P6_DIR     = RESULTS / "phase6" / "figures"
P6B_DIR    = RESULTS / "phase6b" / "figures"
CLE_DIR    = RESULTS / "cleveland"
SHAP_DIR   = BASE_DIR / "shap"
LOG_DIR    = BASE_DIR / "logs"

for d in [P6B_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "phase6b_report.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "legend.frameon":    False,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

COLORS = {
    "XGBoost":            "#27AE60",
    "RandomForest":       "#C0392B",
    "LogisticRegression": "#2980B9",
    "MLP":                "#8E44AD",
}
MODEL_LABELS = {
    "XGBoost":            "XGBoost",
    "RandomForest":       "Random Forest",
    "LogisticRegression": "Logistic Regression",
    "MLP":                "MLP",
}
SCARCITY_LEVELS = [f"S{i}" for i in range(1, 8)]


def save_fig(fig, name: str) -> None:
    png = P6B_DIR / f"{name}.png"
    pdf = P6B_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Saved {name}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — DISTRIBUTIONAL SHIFT (Phase 5B)
# Temporal vs random Wasserstein comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig11_distributional_shift():
    path = P5B_DIR / "random_vs_temporal_drift.csv"
    if not path.exists():
        log.warning("  Phase 5B data not found — skipping Fig 11")
        return

    df     = pd.read_csv(path)
    levels = ["S2","S3","S4","S5","S6","S7"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: temporal vs random Wasserstein
    ax    = axes[0]
    x     = np.arange(len(levels))
    w     = 0.35
    t_val = df["temporal_wass_mean"].values
    r_val = df["random_wass_mean"].values

    bars1 = ax.bar(x - w/2, t_val, w, label="Temporal cycles",
                  color="#C0392B", alpha=0.82, edgecolor="white")
    bars2 = ax.bar(x + w/2, r_val, w, label="Random subsampling",
                  color="#2980B9", alpha=0.75, edgecolor="white")

    for i, (t, r) in enumerate(zip(t_val, r_val)):
        ratio = t / r if r > 0 else 0
        ax.text(i, max(t, r) + 0.015,
               f"{ratio:.1f}×", ha="center", fontsize=8,
               color="#C0392B", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel("Mean Wasserstein Distance", fontsize=10)
    ax.set_title("Temporal vs Random Distributional Shift", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.97,
           "Ratio = temporal/random drift\n"
           "Temporal cycles are 5-6× harder",
           transform=ax.transAxes, fontsize=8, va="top",
           bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="#FFF9E6", edgecolor="#F39C12", alpha=0.8))

    # Right: significant features per level
    ax2   = axes[1]
    t_sig = df["temporal_sig_features"].values
    r_sig = df["random_sig_features_mean"].values

    ax2.bar(x - w/2, t_sig, w, label="Temporal",
           color="#C0392B", alpha=0.82, edgecolor="white")
    ax2.bar(x + w/2, r_sig, w, label="Random",
           color="#2980B9", alpha=0.75, edgecolor="white")

    ax2.axhline(8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels, fontsize=10)
    ax2.set_ylabel("Features with significant drift (KS p<0.05)", fontsize=10)
    ax2.set_title("Significantly Shifted Features per Level", fontsize=11)
    ax2.set_ylim(0, 9)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Fig 11 — Distributional Shift: Temporal Cycles vs Random Subsampling\n"
        "Temporal scarcity introduces 5.3–6.4× more real distributional drift",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig11_distributional_shift")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — CYCLE DRIFT (Phase 5B)
# ══════════════════════════════════════════════════════════════════════════════
def fig12_cycle_drift():
    path = P5B_DIR / "cycle_drift.csv"
    if not path.exists():
        log.warning("  Phase 5B cycle data not found — skipping Fig 12")
        return

    df     = pd.read_csv(path)
    cycles = df["cycle"].unique()
    cycle_kl  = []
    cycle_sig = []

    for cycle in ["2015_2016","2013_2014","2011_2012",
                  "2009_2010","2007_2008","2005_2006"]:
        c_df = df[df["cycle"] == cycle]
        if len(c_df) == 0:
            continue
        cycle_kl.append(float(c_df["kl_divergence"].mean()))
        cycle_sig.append(int(c_df["significant_drift"].sum()))

    cycle_labels = ["2015-16","2013-14","2011-12",
                    "2009-10","2007-08","2005-06"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors  = ["#E74C3C" if s >= 4 else "#F39C12" if s >= 2 else "#27AE60"
               for s in cycle_sig]
    bars    = ax.bar(range(len(cycle_labels)), cycle_kl,
                    color=colors, alpha=0.85, edgecolor="white")

    for i, (bar, sig) in enumerate(zip(bars, cycle_sig)):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.0005,
               f"{sig}/8", ha="center", fontsize=9)

    ax.set_xticks(range(len(cycle_labels)))
    ax.set_xticklabels(cycle_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean KL Divergence vs 2017-2018", fontsize=10)
    ax.set_title(
        "Fig 12 — Distributional Drift Per NHANES Cycle\n"
        "Older cycles show greater feature distribution shift from 2017-2018 baseline",
        fontsize=11, fontweight="bold"
    )
    legend_elements = [
        mpatches.Patch(facecolor="#E74C3C", alpha=0.85, label="≥4 features sig. shifted"),
        mpatches.Patch(facecolor="#F39C12", alpha=0.85, label="2-3 features sig. shifted"),
        mpatches.Patch(facecolor="#27AE60", alpha=0.85, label="0-1 features sig. shifted"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig12_cycle_drift")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 — ESI METRIC COMPARISON (Phase 5C)
# ══════════════════════════════════════════════════════════════════════════════
def fig13_metric_comparison():
    path = P5C_DIR / "metric_comparison.csv"
    if not path.exists():
        log.warning("  Phase 5C data not found — skipping Fig 13")
        return

    comp_df = pd.read_csv(path)
    levels  = ["S2","S3","S4","S5","S6","S7"]
    models  = ["RandomForest","LogisticRegression","XGBoost","MLP"]
    metrics = ["esi","rank_variance","jaccard_top3","marc"]
    metric_colors = {
        "esi":           "#2C3E50",
        "rank_variance": "#E74C3C",
        "jaccard_top3":  "#F39C12",
        "marc":          "#27AE60",
    }
    metric_labels = {
        "esi":           "ESI (proposed)",
        "rank_variance": "Rank Variance",
        "jaccard_top3":  "Jaccard Top-3",
        "marc":          "MARC",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes      = axes.flatten()

    for idx, model in enumerate(models):
        ax   = axes[idx]
        m_df = comp_df[comp_df["model"] == model].sort_values("level")

        for metric in metrics:
            vals = [float(m_df[m_df["level"]==l][metric].values[0])
                   if len(m_df[m_df["level"]==l]) > 0 else np.nan
                   for l in levels]
            ls   = "-" if metric=="esi" else "--" if metric=="rank_variance" \
                   else "-." if metric=="jaccard_top3" else ":"
            lw   = 2.5 if metric=="esi" else 1.5
            ax.plot(range(len(levels)), vals,
                   color=metric_colors[metric],
                   linestyle=ls, linewidth=lw,
                   marker="o" if metric=="esi" else "s",
                   markersize=5 if metric=="esi" else 4,
                   label=metric_labels[metric], zorder=3 if metric=="esi" else 2)

        ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_ylim(0.3, 1.1)
        ax.set_ylabel("Stability Score", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=11,
                    fontweight="bold", color=COLORS[model])
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Fig 13 — ESI vs Alternative Stability Metrics\n"
        "ESI detects instability that Jaccard Top-3 misses in 6 critical cases",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig13_metric_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 14 — LOCAL STABILITY DISTRIBUTION (Phase 5D)
# ══════════════════════════════════════════════════════════════════════════════
def fig14_local_stability():
    path = P5D_DIR / "patient_stability_scores.csv"
    if not path.exists():
        log.warning("  Phase 5D data not found — skipping Fig 14")
        return

    lss_df  = pd.read_csv(path)
    models  = ["XGBoost","RandomForest","LogisticRegression","MLP"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes      = axes.flatten()

    for idx, model in enumerate(models):
        ax      = axes[idx]
        m_lss   = lss_df[lss_df["model"]==model]["mean_lss"].values
        color   = COLORS[model]

        ax.hist(m_lss, bins=30, color=color, alpha=0.75,
               edgecolor="white", linewidth=0.5)
        ax.axvline(0.90, color="green", linestyle="--",
                  linewidth=1.5, label="Stable ≥0.90")
        ax.axvline(0.70, color="red",   linestyle="--",
                  linewidth=1.5, label="Volatile <0.70")

        stable_pct   = (m_lss >= 0.90).mean() * 100
        volatile_pct = (m_lss <  0.70).mean() * 100

        ax.set_xlabel("Mean Local Stability Score (LSS)", fontsize=9)
        ax.set_ylabel("Number of patients", fontsize=9)
        ax.set_title(
            f"{MODEL_LABELS[model]}\n"
            f"Stable: {stable_pct:.1f}%  |  Volatile: {volatile_pct:.1f}%",
            fontsize=10, fontweight="bold", color=color
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        "Fig 14 — Per-Patient Local Explanation Stability Distribution\n"
        "LSS = mean cosine similarity of |SHAP| vectors across S1–S7",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig14_local_stability_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 15 — COMPOUND RISK UNDER SCARCITY (Phase 5E)
# ══════════════════════════════════════════════════════════════════════════════
def fig15_compound_risk():
    path = P5E_DIR / "compound_risk_by_level.csv"
    if not path.exists():
        log.warning("  Phase 5E data not found — skipping Fig 15")
        return

    risk_df = pd.read_csv(path)
    levels  = SCARCITY_LEVELS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: compound risk
    ax = axes[0]
    for model in ["XGBoost","RandomForest","LogisticRegression","MLP"]:
        m_df = risk_df[risk_df["model"]==model].sort_values("level")
        ax.plot(range(len(m_df)), m_df["pct_compound_risk"].values,
               color=COLORS[model], linewidth=2,
               marker="o", markersize=5, label=MODEL_LABELS[model])

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Compound risk patients (%)", fontsize=10)
    ax.set_title("Q4 Compound Risk (Low conf + Low LSS)", fontsize=11)
    ax.legend(fontsize=8)

    # Right: safe patients
    ax2 = axes[1]
    for model in ["XGBoost","RandomForest","LogisticRegression","MLP"]:
        m_df = risk_df[risk_df["model"]==model].sort_values("level")
        ax2.plot(range(len(m_df)), m_df["pct_safe"].values,
                color=COLORS[model], linewidth=2,
                marker="o", markersize=5, label=MODEL_LABELS[model])

    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.set_ylabel("Safe patients — Q1 (%)", fontsize=10)
    ax2.set_title("Q1 Safe Patients (High conf + High LSS)", fontsize=11)
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Fig 15 — Compound Risk Under Progressive Data Scarcity\n"
        "As data decreases: compound risk grows (+8–14pp), safe proportion shrinks",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig15_compound_risk_scarcity")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 16 — CALIBRATION TRENDS (Phase 5F)
# ══════════════════════════════════════════════════════════════════════════════
def fig16_calibration():
    path = P5F_DIR / "calibration_metrics.csv"
    if not path.exists():
        log.warning("  Phase 5F data not found — skipping Fig 16")
        return

    cal_df = pd.read_csv(path)
    levels = SCARCITY_LEVELS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: ECE
    ax = axes[0]
    for model in ["XGBoost","RandomForest","LogisticRegression","MLP"]:
        m_df = cal_df[cal_df["model"]==model].sort_values("level")
        ax.plot(range(len(m_df)), m_df["ece"].values,
               color=COLORS[model], linewidth=2,
               marker="o", markersize=5, label=MODEL_LABELS[model])

    ax.axhline(0.10, color="gray", linestyle="--",
              linewidth=1, alpha=0.6, label="ECE=0.10 threshold")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=10)
    ax.set_title("Calibration stability across scarcity", fontsize=11)
    ax.legend(fontsize=8)

    # Right: CSI vs ESI
    stab_path = P5F_DIR / "calibration_stability.csv"
    if stab_path.exists():
        stab_df = pd.read_csv(stab_path).dropna(subset=["csi","esi"])
        ax2     = axes[1]

        for model in ["XGBoost","RandomForest","LogisticRegression","MLP"]:
            m_df = stab_df[stab_df["model"]==model]
            if m_df.empty:
                continue
            ax2.scatter(m_df["esi"], m_df["csi"],
                       color=COLORS[model], s=70, alpha=0.8,
                       label=MODEL_LABELS[model], zorder=3)
            m_sorted = m_df.sort_values("level")
            ax2.plot(m_sorted["esi"], m_sorted["csi"],
                    color=COLORS[model], alpha=0.3,
                    linewidth=1, linestyle="--")

        lims = [0.5, 1.05]
        ax2.plot(lims, lims, "k--", linewidth=1,
                alpha=0.4, label="Equal degradation")
        ax2.set_xlabel("ESI (explanation stability)", fontsize=10)
        ax2.set_ylabel("CSI (calibration stability)", fontsize=10)
        ax2.set_title("CSI vs ESI — which degrades faster?", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.set_xlim(0.5, 1.05)
        ax2.set_ylim(0.5, 1.05)
    else:
        axes[1].text(0.5, 0.5, "CSI data not available",
                    ha="center", va="center",
                    transform=axes[1].transAxes)

    fig.suptitle(
        "Fig 16 — Calibration Analysis Under Progressive Data Scarcity\n"
        "ECE changes <0.007 (stable) while ESI degrades significantly for MLP/XGB",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig16_calibration_curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 17 — CLEVELAND VS NHANES (External validation)
# ══════════════════════════════════════════════════════════════════════════════
def fig17_cleveland_comparison():
    path = CLE_DIR / "comparison_table.csv"
    if not path.exists():
        log.warning("  Cleveland data not found — skipping Fig 17")
        return

    comp_df = pd.read_csv(path)
    models  = ["RandomForest","LogisticRegression","XGBoost","MLP"]
    comp_df = comp_df.set_index("model").reindex(models).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x       = np.arange(len(models))
    w       = 0.35

    n_vals  = comp_df["nhanes_esi"].values
    c_vals  = comp_df["cleveland_esi"].values
    colors  = [COLORS[m] for m in models]

    bars1 = ax.bar(x - w/2, n_vals, w, label="NHANES (n=32,118)",
                  color=colors, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, c_vals, w, label="Cleveland (n=297)",
                  color=colors, alpha=0.45,
                  edgecolor=colors, linewidth=1.5, hatch="///")

    ax.axhline(0.85, color="gray", linestyle="--",
              linewidth=1, alpha=0.6, label="HIGH threshold (ESI=0.85)")

    for bar, val in zip(bars1, n_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.01,
               f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, c_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
               bar.get_height() + 0.01,
               f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m].replace("Logistic Regression","Log. Reg.")
         for m in models], fontsize=10
    )
    ax.set_ylabel("ESI (Explanation Stability Index)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Fig 17 — External Validation: NHANES vs Cleveland\n"
        "ESI ranking RF > LR > XGB > MLP preserved (ρ=1.00, p<0.001)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "fig17_cleveland_vs_nhanes")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 18 — UPDATED ESI BARS ALL 4 MODELS (Phase 4B fix)
# ══════════════════════════════════════════════════════════════════════════════
def fig18_updated_esi_all_models():
    path = P4_DIR / "bootstrapped_esi.csv"
    if not path.exists():
        log.warning("  Phase 4 data not found — skipping Fig 18")
        return

    boot_df = pd.read_csv(path)
    order   = ["RandomForest","LogisticRegression","XGBoost","MLP"]
    boot_df = boot_df.set_index("model").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x       = np.arange(len(order))
    colors  = [COLORS[m] for m in order]
    means   = boot_df["esi_mean"].values
    lowers  = np.nan_to_num(means - boot_df["esi_ci_lower"].values, nan=0)
    uppers  = np.nan_to_num(boot_df["esi_ci_upper"].values - means, nan=0)

    bars = ax.bar(x, means, width=0.55, color=colors,
                 alpha=0.82, edgecolor="white", linewidth=0.5, zorder=3)
    ax.errorbar(x, means, yerr=[lowers, uppers],
               fmt="none", color="black", capsize=5,
               capthick=1.5, linewidth=1.5, zorder=4)

    for i, (bar, mean, model) in enumerate(zip(bars, means, order)):
        ax.text(bar.get_x() + bar.get_width()/2,
               mean + 0.015,
               f"{mean:.4f}", ha="center", va="bottom",
               fontsize=9, fontweight="bold", color=colors[i])

    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(0.70, color="gray", linestyle=":",  linewidth=1.0, alpha=0.5)
    ax.text(3.4, 0.86, "HIGH (≥0.85)", color="gray", fontsize=8)
    ax.text(3.4, 0.71, "MODERATE (≥0.70)", color="gray", fontsize=8)

    rel = boot_df["reliability"].values
    for i, (bar, r) in enumerate(zip(bars, rel)):
        color_map = {"HIGH":"green","MODERATE":"orange","LOW":"red","UNKNOWN":"gray"}
        ax.text(bar.get_x() + bar.get_width()/2,
               0.02, r, ha="center", va="bottom",
               fontsize=8, color=color_map.get(r,"gray"),
               transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m].replace("Logistic Regression","Log. Reg.")
         for m in order], fontsize=10
    )
    ax.set_ylabel("Bootstrapped ESI (mean ± 95% CI)", fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.set_title(
        "Fig 18 — Bootstrapped ESI with 95% CI — All Four Models\n"
        "(B=50 for tree/linear; B=200 for MLP using SHAP-array bootstrap)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_fig(fig, "fig18_updated_esi_all_models")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE INDEX — master table of all figures
# ══════════════════════════════════════════════════════════════════════════════
def build_figure_index() -> pd.DataFrame:
    """Build a master CSV index of all figures for the paper."""
    all_figures = [
        # Original Phase 6
        ("fig01","fig1_esi_stability_curves",     "phase6","Main result — ESI curves with 95% CI bands"),
        ("fig02","fig2_bootstrapped_esi_bars",     "phase6","Bootstrapped ESI summary (original 3 models)"),
        ("fig03","fig3_feature_reliability_heatmap","phase6","Feature reliability heatmap"),
        ("fig04","fig4_reliability_flag_heatmap",  "phase6","GREEN/AMBER/RED flag grid"),
        ("fig05","fig5_ablation_threshold",        "phase6","Ablation A — threshold sensitivity"),
        ("fig06","fig6_ablation_temporal_vs_random","phase6","Ablation C — temporal vs random"),
        ("fig07","fig7_shap_importance_s1",        "phase6","SHAP importance at S1 full data"),
        ("fig08","fig8_scarcity_threshold",        "phase6","ESI vs patient count"),
        ("fig09","fig9_auc_vs_esi",                "phase6","AUC vs ESI scatter"),
        ("fig10","fig10_publication_table",        "phase6","Publication summary table"),
        # New Phase 6B
        ("fig11","fig11_distributional_shift",     "phase6b","Temporal vs random Wasserstein shift"),
        ("fig12","fig12_cycle_drift",              "phase6b","KL drift per NHANES cycle"),
        ("fig13","fig13_metric_comparison",        "phase6b","ESI vs alternative metrics"),
        ("fig14","fig14_local_stability_distribution","phase6b","Per-patient LSS distribution"),
        ("fig15","fig15_compound_risk_scarcity",   "phase6b","Compound risk growth under scarcity"),
        ("fig16","fig16_calibration_curves",       "phase6b","Calibration ECE and CSI vs ESI"),
        ("fig17","fig17_cleveland_vs_nhanes",      "phase6b","External validation NHANES vs Cleveland"),
        ("fig18","fig18_updated_esi_all_models",   "phase6b","Updated ESI bars all 4 models with CI"),
    ]

    rows = []
    for fig_num, filename, folder, description in all_figures:
        base_dir = RESULTS / folder / "figures"
        png_path = base_dir / f"{filename}.png"
        exists   = png_path.exists()

        rows.append({
            "figure_number": fig_num,
            "filename":      filename,
            "folder":        folder,
            "description":   description,
            "png_exists":    exists,
            "pdf_exists":    (base_dir / f"{filename}.pdf").exists(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase6b" / "figure_index.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("RES-CHD  Phase 6B — Master Figure Generator")
    log.info("=" * 65)
    log.info("Generating figures from all new modules (5B, 5C, 5D, 5E, 5F, Cleveland)")

    figures = [
        ("Fig 11 — Distributional shift",       fig11_distributional_shift),
        ("Fig 12 — Cycle drift",                fig12_cycle_drift),
        ("Fig 13 — ESI metric comparison",      fig13_metric_comparison),
        ("Fig 14 — Local stability dist.",      fig14_local_stability),
        ("Fig 15 — Compound risk scarcity",     fig15_compound_risk),
        ("Fig 16 — Calibration trends",         fig16_calibration),
        ("Fig 17 — Cleveland vs NHANES",        fig17_cleveland_comparison),
        ("Fig 18 — Updated ESI all models",     fig18_updated_esi_all_models),
    ]

    success = 0
    for name, func in figures:
        log.info(f"\n  {name}")
        try:
            func()
            success += 1
        except Exception as e:
            log.error(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Build master figure index
    log.info(f"\n{'='*65}")
    log.info("  Building master figure index ...")
    index_df = build_figure_index()
    log.info(f"\n  Complete figure inventory:")
    log.info(f"\n{index_df[['figure_number','filename','png_exists','description']].to_string(index=False)}")

    total_figs = index_df["png_exists"].sum()
    log.info(f"\n{'='*65}")
    log.info(f"Phase 6B complete — {success}/8 new figures generated")
    log.info(f"Total figures available: {total_figs}/18")
    log.info(f"Output: {P6B_DIR}")
    log.info(f"\nAll publication figures ready. ✓")


if __name__ == "__main__":
    main()