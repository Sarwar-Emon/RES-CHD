import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["RandomForest", "LogisticRegression", "XGBoost", "MLP"]
LABEL = {"RandomForest": "Random Forest", "LogisticRegression": "Logistic Regression", "XGBoost": "XGBoost", "MLP": "MLP"}
STYLE = {"RandomForest": ("#1F3864", "o"), "LogisticRegression": ("#2E75B6", "s"), "XGBoost": ("#F5A623", "^"), "MLP": ("#C0392B", "D")}
LEVELS = [f"S{k}" for k in range(1, 8)]
THESIS_NTRAIN = [25694, 22631, 18747, 14620, 11076, 7237, 3424]   # Table 3.2, used only if the train files are missing

def main(a):
    cal = pd.read_csv(os.path.join(a.base, "results/phase5f/calibration_stability.csv"))
    ci = pd.read_csv(os.path.join(a.base, "results/final/cleveland_esi_with_ci.csv")).set_index("model")
    try:
        ntrain = [len(pd.read_csv(os.path.join(a.base, f"data/scarcity_levels/{lv}_train.csv"), usecols=[0])) for lv in LEVELS]
        print("  training N read from data/scarcity_levels/*_train.csv")
    except Exception as e:
        ntrain = THESIS_NTRAIN; print("  WARNING: could not read the train files (", e, ") - using Training N from thesis Table 3.2")
    rho = {m: [1.0] + [float(cal[(cal.model == m) & (cal.level == lv)]["esi"].iloc[0]) for lv in LEVELS[1:]] for m in MODELS}
    ps = os.path.join(a.base, "results/phase5/publication_summary.csv")
    if os.path.exists(ps):
        pt = pd.read_csv(ps).set_index("model")["esi_point"]
        for m in MODELS:
            mean = float(np.mean(rho[m][1:])); flag = "OK" if abs(mean - float(pt[m])) <= 0.002 else "WARNING: differs from esi_point"
            print(f"  check {m:20s} mean of S2-S7 = {mean:.4f} | esi_point = {float(pt[m]):.4f}  {flag}")
    print("  training N per level:", dict(zip(LEVELS, ntrain)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(7)
    ax1.axhspan(0.85, 1.05, color="#2E8B57", alpha=0.10, lw=0); ax1.axhspan(0.70, 0.85, color="#F5A623", alpha=0.12, lw=0); ax1.axhspan(0.30, 0.70, color="#C0392B", alpha=0.09, lw=0)
    ax1.axhline(0.85, color="#2E8B57", ls="--", lw=1.3, label="HIGH threshold (0.85)"); ax1.axhline(0.70, color="#E0A000", ls="--", lw=1.3, label="MODERATE threshold (0.70)")
    for m in MODELS:
        c, mk = STYLE[m]; ax1.plot(x, rho[m], color=c, marker=mk, lw=2.6, ms=8, label=LABEL[m], zorder=3)
    for m in ("RandomForest", "MLP"):
        ax1.annotate(f"{rho[m][-1]:.3f}", (6, rho[m][-1]), xytext=(9, 0), textcoords="offset points", fontsize=9, color=STYLE[m][0], va="center")
    ax1.annotate(f"{rho['LogisticRegression'][-1]:.3f}\n(LR and XGB)", (6, rho["LogisticRegression"][-1]), xytext=(9, 0), textcoords="offset points", fontsize=9, color="#2E75B6", va="center")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{lv}\n({n:,})" for lv, n in zip(LEVELS, ntrain)], fontsize=9.5)
    ax1.set_xlim(-0.3, 6.9); ax1.set_ylim(0.34, 1.05); ax1.set_ylabel("Spearman rho between S1 ranking and level ranking"); ax1.set_xlabel("Scarcity level (number of training patients)")
    ax1.set_title("(a) Per-level rank stability", fontsize=12, fontweight="bold", loc="left"); ax1.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.95); ax1.grid(alpha=0.25, ls=":")
    mean = [float(ci.loc[m, "nhanes_esi"]) for m in MODELS]; lo = [float(ci.loc[m, "nhanes_ci_lower"]) for m in MODELS]; hi = [float(ci.loc[m, "nhanes_ci_upper"]) for m in MODELS]
    xb = np.arange(4); cols = [STYLE[m][0] for m in MODELS]
    ax2.bar(xb, mean, color=cols, width=0.62, yerr=[np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)], capsize=5, ecolor="#333333", zorder=3)
    ax2.axhline(0.85, color="#2E8B57", ls="--", lw=1.3); ax2.axhline(0.70, color="#E0A000", ls="--", lw=1.3)
    for xi, mv, h_ in zip(xb, mean, hi):
        ax2.text(xi, h_ + 0.015, f"{mv:.4f}", ha="center", fontsize=9.5, fontweight="bold"); ax2.text(xi, 0.04, "HIGH" if mv >= 0.85 else ("MODERATE" if mv >= 0.70 else "LOW"), ha="center", fontsize=8.5, color="white", fontweight="bold")
    ax2.set_xticks(xb); ax2.set_xticklabels([LABEL[m].replace(" ", "\n") for m in MODELS], fontsize=9.5); ax2.set_ylim(0, 1.08); ax2.set_ylabel("Mean ESI over S2-S7 (95% bootstrap interval)")
    ax2.set_title("(b) Mean ESI", fontsize=12, fontweight="bold", loc="left"); ax2.grid(axis="y", alpha=0.25, ls=":", zorder=0)
    plt.tight_layout(); os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "Fig4_1_ESI_Trajectory.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(a.out, "Fig4_1_ESI_Trajectory.pdf"), bbox_inches="tight")
    print("  saved", os.path.join(a.out, "Fig4_1_ESI_Trajectory.png"), "and .pdf")
    print("\n  per-level rho used in panel (a):"); print(pd.DataFrame(rho, index=LEVELS).T.round(4).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--out", default="outputs/figures"); main(ap.parse_args())
