"""
RES-CHD Phase 6 — Publication Quality Visualizations
=====================================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Generates all publication-quality figures for thesis and journal paper.

Figures produced:
  Fig 1 — ESI stability curves with 95% CI bands (main result)
  Fig 2 — Bootstrapped ESI bar chart with error bars (summary)
  Fig 3 — Feature reliability heatmap (per model × feature)
  Fig 4 — Reliability flag heatmap (GREEN/AMBER/RED grid)
  Fig 5 — Ablation A: threshold sensitivity
  Fig 6 — Ablation C: temporal vs random scarcity delta
  Fig 7 — Feature importance at S1 (SHAP bar chart, 4 models)
  Fig 8 — Scarcity threshold visualization (patient count vs ESI)
  Fig 9 — AUC vs ESI scatter (predictive vs explanatory reliability)
  Fig 10 — Publication summary table (formatted)

All figures saved as:
  - PNG (300 DPI for print)
  - PDF (vector, for journal submission)

Outputs: results/phase6/figures/
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
SHAP_DIR   = BASE_DIR / "shap"
RESULTS    = BASE_DIR / "results"
P4_DIR     = RESULTS / "phase4"
P5_DIR     = RESULTS / "phase5"
FIG_DIR    = RESULTS / "phase6" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
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

# ── Color palette (accessible, publication-safe) ───────────────────────────────
COLORS = {
    "RandomForest":      "#C0392B",   # red
    "LogisticRegression":"#2980B9",   # blue
    "XGBoost":           "#27AE60",   # green
    "MLP":               "#8E44AD",   # purple
}
LINESTYLES = {
    "RandomForest":      "-",
    "LogisticRegression":"--",
    "XGBoost":           "-.",
    "MLP":               ":",
}
MARKERS = {
    "RandomForest":      "o",
    "LogisticRegression":"s",
    "XGBoost":           "^",
    "MLP":               "D",
}
MODEL_LABELS = {
    "RandomForest":      "Random Forest",
    "LogisticRegression":"Logistic Regression",
    "XGBoost":           "XGBoost",
    "MLP":               "MLP",
}

LEVELS     = ["S1","S2","S3","S4","S5","S6","S7"]
LEVEL_XPOS = list(range(len(LEVELS)))

def save_fig(fig, name: str) -> None:
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — ESI STABILITY CURVES WITH 95% CI BANDS
# ══════════════════════════════════════════════════════════════════════════════
def fig1_esi_stability_curves():
    """Main result figure — Spearman rho vs S1 across scarcity levels."""

    # Phase 3 point estimates
    rank_stab = pd.read_csv(SHAP_DIR / "stability" / "rank_stability.csv")

    # Bootstrap distributions for CI bands
    boot_dir  = P4_DIR / "bootstrap_distributions"

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in ["RandomForest", "LogisticRegression", "XGBoost", "MLP"]:
        m_df = rank_stab[rank_stab["model"] == model].sort_values("comparison")
        x    = [LEVELS.index(l) + 1 for l in m_df["comparison"]]   # S2=1..S7=6
        y    = m_df["spearman_rho"].values
        color = COLORS[model]

        # Point estimates
        ax.plot(x, y,
                color=color,
                linestyle=LINESTYLES[model],
                marker=MARKERS[model],
                markersize=7,
                linewidth=2,
                label=MODEL_LABELS[model],
                zorder=3)

        # Bootstrap CI band (if available)
        ci_lowers, ci_uppers = [], []
        for level in ["S2","S3","S4","S5","S6","S7"]:
            boot_path = boot_dir / f"{model}_{level}_bootstrap.csv"
            if boot_path.exists():
                boot_df = pd.read_csv(boot_path)
                rhos    = boot_df["spearman_rho"].values
                ci_lowers.append(float(np.percentile(rhos, 2.5)))
                ci_uppers.append(float(np.percentile(rhos, 97.5)))
            else:
                ci_lowers.append(y[["S2","S3","S4","S5","S6","S7"].index(level)]
                                  if level in ["S2","S3","S4","S5","S6","S7"] else np.nan)
                ci_uppers.append(ci_lowers[-1])

        if any(not np.isnan(v) for v in ci_lowers):
            ax.fill_between(x, ci_lowers, ci_uppers,
                           color=color, alpha=0.12, zorder=2)

    # Threshold lines
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1.2,
               alpha=0.7, label="ESI=0.85 (HIGH threshold)")
    ax.axhline(0.70, color="gray", linestyle=":",  linewidth=1.0,
               alpha=0.5, label="ESI=0.70 (MODERATE threshold)")

    ax.set_xlabel("Scarcity Level (S2 = 6 cycles → S7 = 1 cycle)", fontsize=11)
    ax.set_ylabel("Spearman ρ vs S1 Baseline", fontsize=11)
    ax.set_title("Fig 1 — SHAP Explanation Stability Index (ESI)\nacross Progressive Data Scarcity Levels",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"S{i}\n({['28K','23K','18K','14K','9K','4K'][i-2]})"
                        for i in range(2, 8)], fontsize=9)
    ax.set_ylim(0.35, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0.5, 6.5)

    # Shade unstable zone
    ax.axhspan(0.35, 0.70, alpha=0.05, color="red", zorder=0)
    ax.text(6.4, 0.52, "LOW\nstability", color="red", alpha=0.6,
            fontsize=8, ha="right", va="center")

    save_fig(fig, "fig1_esi_stability_curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — BOOTSTRAPPED ESI BAR CHART WITH ERROR BARS
# ══════════════════════════════════════════════════════════════════════════════
def fig2_bootstrapped_esi_bars():
    boot_df = pd.read_csv(P4_DIR / "bootstrapped_esi.csv")

    # Add MLP point estimate (no bootstrap)
    mlp_esi = 0.6587
    mlp_row = pd.DataFrame([{
        "model": "MLP", "esi_mean": mlp_esi, "esi_std": np.nan,
        "esi_ci_lower": np.nan, "esi_ci_upper": np.nan, "reliability": "LOW (point estimate)"
    }])
    boot_df = pd.concat([boot_df, mlp_row], ignore_index=True)

    # Sort by ESI
    order   = ["RandomForest","LogisticRegression","XGBoost","MLP"]
    boot_df = boot_df.set_index("model").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x      = np.arange(len(order))
    colors = [COLORS[m] for m in order]
    means  = boot_df["esi_mean"].values
    lowers = (means - boot_df["esi_ci_lower"].values)
    uppers = (boot_df["esi_ci_upper"].values - means)

    # Replace nan errors with 0 for MLP
    lowers = np.nan_to_num(lowers, nan=0)
    uppers = np.nan_to_num(uppers, nan=0)

    bars = ax.bar(x, means, width=0.55, color=colors, alpha=0.82,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # Error bars
    ax.errorbar(x, means, yerr=[lowers, uppers],
                fmt="none", color="black", capsize=5,
                capthick=1.5, linewidth=1.5, zorder=4)

    # Value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.015,
                f"{mean:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=colors[i])

    # Threshold lines
    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(0.70, color="gray", linestyle=":",  linewidth=1.0, alpha=0.5)
    ax.text(3.4, 0.86, "HIGH (≥0.85)", color="gray", fontsize=8, va="bottom")
    ax.text(3.4, 0.71, "MODERATE (≥0.70)", color="gray", fontsize=8, va="bottom")

    # MLP annotation
    ax.annotate("*Point estimate only\n(no bootstrap CI)",
                xy=(3, mlp_esi), xytext=(2.5, mlp_esi - 0.08),
                fontsize=8, color="#8E44AD",
                arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order], fontsize=10)
    ax.set_ylabel("Bootstrapped ESI (mean ± 95% CI)", fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.set_title("Fig 2 — Bootstrapped Explanation Stability Index\nwith 95% Confidence Intervals (B=50)",
                 fontsize=12, fontweight="bold")

    save_fig(fig, "fig2_bootstrapped_esi_bars")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — FEATURE RELIABILITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def fig3_feature_reliability_heatmap():
    feat_df = pd.read_csv(P4_DIR / "feature_reliability.csv")
    features = ["age","total_chol","sex","smoking","dbp","hdl","sbp","bmi"]
    models   = ["RandomForest","LogisticRegression","XGBoost","MLP"]

    matrix = np.zeros((len(models), len(features)))
    for i, model in enumerate(models):
        for j, feat in enumerate(features):
            row = feat_df[(feat_df["model"]==model) & (feat_df["feature"]==feat)]
            if len(row) > 0:
                matrix[i, j] = float(row["reliability_score"].values[0])

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=10)

    # Cell annotations
    for i in range(len(models)):
        for j in range(len(features)):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            anchor = "★" if val == 1.0 else f"{val:.2f}"
            ax.text(j, i, anchor, ha="center", va="center",
                   fontsize=9, color=color, fontweight="bold" if val==1.0 else "normal")

    plt.colorbar(im, ax=ax, label="Feature Reliability Score (0=volatile, 1=anchor)",
                 fraction=0.03, pad=0.02)
    ax.set_title("Fig 3 — Feature-Level Reliability Scores\n★ = Anchor feature (rank never changes)",
                 fontsize=12, fontweight="bold")

    save_fig(fig, "fig3_feature_reliability_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — RELIABILITY FLAG HEATMAP (GREEN/AMBER/RED)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_reliability_flag_heatmap():
    flags_df = pd.read_csv(P4_DIR / "reliability_flags.csv")
    models   = ["RandomForest","LogisticRegression","XGBoost","MLP"]
    levels   = ["S1","S2","S3","S4","S5","S6","S7"]

    flag_map = {"GREEN": 1.0, "AMBER": 0.5, "RED": 0.0, "UNKNOWN": -1.0}
    cmap     = matplotlib.colors.ListedColormap(["#cccccc","#e74c3c","#f39c12","#27ae60"])
    bounds   = [-1.5, -0.5, 0.25, 0.75, 1.5]
    norm     = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    matrix = np.full((len(models), len(levels)), -1.0)
    for i, model in enumerate(models):
        for j, level in enumerate(levels):
            row = flags_df[(flags_df["model"]==model) & (flags_df["level"]==level)]
            if len(row) > 0:
                matrix[i, j] = flag_map.get(row["flag"].values[0], -1.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(
        [f"{l}\n({'32K' if l=='S1' else '28K' if l=='S2' else '23K' if l=='S3' else '18K' if l=='S4' else '14K' if l=='S5' else '9K' if l=='S6' else '4K'})"
         for l in levels], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=10)

    # Text annotations
    flag_text = {"GREEN":"G","AMBER":"A","RED":"R","UNKNOWN":"?"}
    for i, model in enumerate(models):
        for j, level in enumerate(levels):
            row = flags_df[(flags_df["model"]==model) & (flags_df["level"]==level)]
            if len(row) > 0:
                flag = row["flag"].values[0]
                color = "white" if flag in ["RED","UNKNOWN"] else "black"
                ax.text(j, i, flag_text.get(flag,"?"),
                       ha="center", va="center",
                       fontsize=10, fontweight="bold", color=color)

    legend_elements = [
        mpatches.Patch(facecolor="#27ae60", label="GREEN — High reliability (ESI ≥ 0.85)"),
        mpatches.Patch(facecolor="#f39c12", label="AMBER — Moderate (0.70 ≤ ESI < 0.85)"),
        mpatches.Patch(facecolor="#e74c3c", label="RED — Low reliability (ESI < 0.70)"),
        mpatches.Patch(facecolor="#cccccc", label="UNKNOWN — No bootstrap (MLP)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
             fontsize=8, bbox_to_anchor=(1.0, -0.35), ncol=2)

    ax.set_title("Fig 4 — RES-CHD Reliability Flag System\nClinical Deployment Guidance per Model × Scarcity Level",
                 fontsize=12, fontweight="bold")

    save_fig(fig, "fig4_reliability_flag_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — ABLATION A: THRESHOLD SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
def fig5_ablation_threshold():
    abl_df = pd.read_csv(P5_DIR / "ablation_esi_thresholds.csv")
    thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    models     = ["RandomForest","LogisticRegression","XGBoost","MLP"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model in models:
        m_df = abl_df[abl_df["model"]==model].sort_values("threshold")
        esi  = m_df["esi"].values[0]   # ESI is same across thresholds
        ax.axhline(esi, color=COLORS[model],
                  linestyle=LINESTYLES[model], linewidth=2,
                  label=f"{MODEL_LABELS[model]} (ESI={esi:.4f})")

    for t in thresholds:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(t, 0.37, str(t), ha="center", fontsize=8, color="gray")

    ax.set_xlabel("ESI Reliability Threshold", fontsize=11)
    ax.set_ylabel("ESI Score", fontsize=11)
    ax.set_ylim(0.35, 1.02)
    ax.set_xlim(0.73, 0.97)
    ax.set_title("Fig 5 — Ablation A: Threshold Sensitivity\nESI Rankings Preserved Across All Thresholds",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="center right", fontsize=9)

    save_fig(fig, "fig5_ablation_threshold")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — ABLATION C: TEMPORAL VS RANDOM SCARCITY
# ══════════════════════════════════════════════════════════════════════════════
def fig6_ablation_temporal_vs_random():
    abl_df = pd.read_csv(P5_DIR / "ablation_random_scarcity.csv")
    levels = ["S2","S3","S4","S5","S6","S7"]
    models = ["RandomForest","LogisticRegression","XGBoost","MLP"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: temporal vs random rho per model
    ax = axes[0]
    for model in models:
        m_df = abl_df[abl_df["model"]==model].sort_values("level")
        x    = range(len(m_df))
        ax.plot(x, m_df["rho_temporal"].values,
               color=COLORS[model], linestyle="-",
               marker=MARKERS[model], markersize=6,
               linewidth=2, label=MODEL_LABELS[model])
        ax.plot(x, m_df["rho_random_mean"].values,
               color=COLORS[model], linestyle="--",
               marker=MARKERS[model], markersize=5,
               linewidth=1.5, alpha=0.5)

    ax.set_xticks(range(6))
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel("Spearman ρ vs S1 baseline", fontsize=10)
    ax.set_title("Temporal (solid) vs Random (dashed)", fontsize=11)
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=8)

    # Right: delta (temporal - random) — always negative
    ax2 = axes[1]
    x    = np.arange(6)
    width = 0.2
    for i, model in enumerate(models):
        m_df  = abl_df[abl_df["model"]==model].sort_values("level")
        delta = m_df["delta_temporal_minus_random"].values
        ax2.bar(x + i*width - 0.3, delta, width=width,
               color=COLORS[model], alpha=0.8, label=MODEL_LABELS[model])

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels, fontsize=10)
    ax2.set_ylabel("Δ (Temporal − Random)", fontsize=10)
    ax2.set_title("ESI Gap: Temporal harder than Random", fontsize=11)
    ax2.legend(fontsize=8)

    fig.suptitle("Fig 6 — Ablation C: Temporal Cycle vs Random Subsampling\nTemporal scarcity reflects real distributional shift (lower ESI = harder, more realistic)",
                fontsize=11, fontweight="bold", y=1.02)

    save_fig(fig, "fig6_ablation_temporal_vs_random")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — SHAP FEATURE IMPORTANCE AT S1 (ALL 4 MODELS)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_shap_importance_s1():
    features   = ["age","total_chol","sex","smoking","dbp","hdl","sbp","bmi"]
    models     = ["RandomForest","LogisticRegression","XGBoost","MLP"]

    # Load S1 global SHAP importance
    global_df = pd.read_csv(SHAP_DIR / "global" / "all_global_shap.csv", index_col=0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax    = axes[idx]
        col   = f"S1_{model}"
        if col not in global_df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes)
            continue

        vals  = global_df[col].reindex(features).values
        # Normalize to [0,1] for visual comparison
        vals_norm = vals / vals.max() if vals.max() > 0 else vals
        colors_bar = [COLORS[model]] * len(features)
        # Highlight top feature
        colors_bar[np.argmax(vals)] = "#F39C12"

        bars = ax.barh(features[::-1], vals_norm[::-1],
                      color=colors_bar[::-1], alpha=0.85, edgecolor="white")

        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Normalized mean |SHAP|", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=11,
                    fontweight="bold", color=COLORS[model])
        ax.tick_params(axis="y", labelsize=9)

        # Value annotations
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Fig 7 — Global SHAP Feature Importance at S1 (Full Data Baseline)\nNormalized mean |SHAP| — gold bar = highest importance feature",
                fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_fig(fig, "fig7_shap_importance_s1")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — SCARCITY THRESHOLD VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def fig8_scarcity_threshold():
    """ESI vs patient count — shows where each model's explanation breaks."""

    # Patient counts per level
    level_n = {
        "S1": 32118, "S2": 28289, "S3": 23434,
        "S4": 18276, "S5": 13846, "S6": 9047, "S7": 4281
    }

    rank_stab = pd.read_csv(SHAP_DIR / "stability" / "rank_stability.csv")
    models    = ["RandomForest","LogisticRegression","XGBoost","MLP"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        m_df = rank_stab[rank_stab["model"]==model].sort_values("comparison")
        # Include S1 as baseline (rho=1.0)
        ns   = [level_n["S1"]] + [level_n[l] for l in m_df["comparison"]]
        rhos = [1.0] + list(m_df["spearman_rho"].values)

        ax.plot(ns, rhos,
               color=COLORS[model],
               linestyle=LINESTYLES[model],
               marker=MARKERS[model],
               markersize=7, linewidth=2,
               label=MODEL_LABELS[model],
               zorder=3)

    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1.2,
              alpha=0.7, label="HIGH threshold (ESI=0.85)")
    ax.axhline(0.70, color="gray", linestyle=":",  linewidth=1.0,
              alpha=0.5, label="MODERATE threshold (ESI=0.70)")

    # Shade regions
    ax.axhspan(0.85, 1.05, alpha=0.05, color="green", zorder=0)
    ax.axhspan(0.70, 0.85, alpha=0.05, color="orange", zorder=0)
    ax.axhspan(0.30, 0.70, alpha=0.05, color="red",    zorder=0)

    ax.set_xlabel("Training Set Size (N patients)", fontsize=11)
    ax.set_ylabel("Spearman ρ vs S1 baseline", fontsize=11)
    ax.set_title("Fig 8 — ESI vs Data Availability\nClinical Scarcity Threshold Identification",
                fontsize=12, fontweight="bold")
    ax.set_ylim(0.30, 1.05)
    ax.invert_xaxis()   # more data on left = S1, less on right = S7
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K")
    )
    ax.legend(loc="lower right", fontsize=9)

    # Annotate threshold crossings
    ax.annotate("XGB drops\nbelow 0.85\n~13.8K pts",
               xy=(13846, 0.8333), xytext=(18000, 0.72),
               fontsize=8, color=COLORS["XGBoost"],
               arrowprops=dict(arrowstyle="->", color=COLORS["XGBoost"], lw=1))

    save_fig(fig, "fig8_scarcity_threshold")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — AUC vs ESI SCATTER (predictive vs explanatory reliability)
# ══════════════════════════════════════════════════════════════════════════════
def fig9_auc_vs_esi():
    """Shows that high AUC does not guarantee high ESI — key thesis argument."""

    consolidated = pd.read_csv(P5_DIR / "final_consolidated_results.csv")
    esi_df       = pd.read_csv(SHAP_DIR / "stability" / "esi_scores.csv")
    esi_map      = dict(zip(esi_df["model"], esi_df["esi_score"]))

    fig, ax = plt.subplots(figsize=(7, 5))

    for model in ["RandomForest","LogisticRegression","XGBoost","MLP"]:
        m_df = consolidated[consolidated["model"]==model]
        esi  = esi_map.get(model, np.nan)
        aucs = m_df["auc"].values

        # Plot each level as a dot
        for i, (level, auc) in enumerate(zip(LEVELS, aucs)):
            ax.scatter(auc, esi,
                      color=COLORS[model],
                      marker=MARKERS[model],
                      s=80, alpha=0.7 + 0.03*i,
                      zorder=3)

        # Model label
        mean_auc = np.mean(aucs)
        ax.annotate(MODEL_LABELS[model],
                   xy=(mean_auc, esi),
                   xytext=(mean_auc + 0.003, esi + 0.015),
                   fontsize=9, color=COLORS[model], fontweight="bold")

    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(0.82, color="gray", linestyle=":",  linewidth=1, alpha=0.5)

    ax.set_xlabel("ROC-AUC (predictive performance)", fontsize=11)
    ax.set_ylabel("ESI (explanation stability)", fontsize=11)
    ax.set_title("Fig 9 — Predictive Performance vs Explanation Stability\nHigh AUC does not guarantee reliable explanations",
                fontsize=12, fontweight="bold")

    # Quadrant labels
    ax.text(0.807, 0.60, "Low AUC\nLow ESI", fontsize=8, color="gray",
           ha="center", va="center", alpha=0.6)
    ax.text(0.862, 0.60, "High AUC\nLow ESI", fontsize=8, color="red",
           ha="center", va="center", alpha=0.7)
    ax.text(0.862, 0.92, "High AUC\nHigh ESI\n(ideal)", fontsize=8, color="green",
           ha="center", va="center", alpha=0.7)

    save_fig(fig, "fig9_auc_vs_esi")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — PUBLICATION SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def fig10_publication_table():
    pub_df = pd.read_csv(P5_DIR / "publication_summary.csv")

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    cols = ["model","auc_at_s1","prauc_at_s1",
            "esi_bootstrapped","esi_95ci",
            "overall_reliability","scarcity_threshold",
            "green_levels","amber_levels","red_levels"]

    col_labels = ["Model","AUC\n@S1","PR-AUC\n@S1",
                  "ESI\n(Boot.)","95% CI",
                  "Overall\nReliability","Scarcity\nThreshold",
                  "GREEN\nLevels","AMBER\nLevels","RED\nLevels"]

    table_data = []
    for _, row in pub_df.iterrows():
        table_data.append([
            MODEL_LABELS.get(row["model"], row["model"]),
            f"{row['auc_at_s1']:.4f}",
            f"{row['prauc_at_s1']:.4f}",
            f"{row['esi_bootstrapped']:.4f}" if not pd.isna(row['esi_bootstrapped']) else "N/A",
            str(row['esi_95ci']),
            str(row['overall_reliability']),
            str(row['scarcity_threshold']),
            str(int(row['green_levels'])),
            str(int(row['amber_levels'])),
            str(int(row['red_levels'])),
        ])

    table = ax.table(cellText=table_data,
                    colLabels=col_labels,
                    cellLoc="center",
                    loc="center",
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C4E7E")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by reliability
    row_colors = {"HIGH": "#EAF5EA", "LOW": "#FDE9E9", "UNKNOWN": "#F5F5F5"}
    for i, row in pub_df.iterrows():
        color = row_colors.get(row["overall_reliability"], "white")
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(color)

    ax.set_title("Fig 10 — RES-CHD Publication Summary Table\nComplete results across all models",
                fontsize=12, fontweight="bold", pad=20)

    save_fig(fig, "fig10_publication_table")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("RES-CHD  Phase 6 — Publication Quality Visualizations")
    print("=" * 60)

    figures = [
        ("Fig 1  — ESI stability curves",         fig1_esi_stability_curves),
        ("Fig 2  — Bootstrapped ESI bar chart",    fig2_bootstrapped_esi_bars),
        ("Fig 3  — Feature reliability heatmap",   fig3_feature_reliability_heatmap),
        ("Fig 4  — Reliability flag heatmap",      fig4_reliability_flag_heatmap),
        ("Fig 5  — Ablation A: threshold",         fig5_ablation_threshold),
        ("Fig 6  — Ablation C: temporal vs random",fig6_ablation_temporal_vs_random),
        ("Fig 7  — SHAP importance at S1",         fig7_shap_importance_s1),
        ("Fig 8  — Scarcity threshold plot",       fig8_scarcity_threshold),
        ("Fig 9  — AUC vs ESI scatter",            fig9_auc_vs_esi),
        ("Fig 10 — Publication summary table",     fig10_publication_table),
    ]

    success = 0
    for name, func in figures:
        print(f"\n  {name}")
        try:
            func()
            success += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Phase 6 complete — {success}/{len(figures)} figures generated")
    print(f"Output directory: {FIG_DIR}")
    print(f"\nFiles saved:")
    for f in sorted(FIG_DIR.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")
    print("\nAll figures ready for thesis and journal submission. ✓")


if __name__ == "__main__":
    main()