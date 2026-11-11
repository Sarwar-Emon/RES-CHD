import argparse, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "MLP"]; SHORT = {"LogisticRegression": "LR", "RandomForest": "RF", "XGBoost": "XGB", "MLP": "MLP"}
COL = {"LogisticRegression": "#2B7BD9", "RandomForest": "#F26A3B", "XGBoost": "#1FB58A", "MLP": "#F2A81D"}; MK = {"LogisticRegression": "s", "RandomForest": "o", "XGBoost": "^", "MLP": "D"}
LS = {"LogisticRegression": "-", "RandomForest": "--", "XGBoost": "-.", "MLP": ":"}; LV = [f"S{k}" for k in range(1, 8)]

def main(a):
    cal = pd.read_csv(os.path.join(a.base, "results/phase5f/calibration_stability.csv"))
    rho = {m: [1.0] + [float(cal[(cal.model == m) & (cal.level == lv)]["esi"].iloc[0]) for lv in LV[1:]] for m in ORDER}
    raw = pd.concat([pd.read_csv(os.path.join(a.base, f"outputs/{d}/split_robustness_raw.csv")) for d in ("sr_mlp10", "sr_rf10", "sr_fast300")]); raw = raw[raw.level != "S1"]
    per = raw.groupby(["model", "seed"])["rho"].mean(); stats = {m: (per[m].mean(), per[m].std(ddof=1), len(per[m])) for m in ORDER}
    auc = pd.read_csv(os.path.join(a.base, "results/phase5/publication_summary.csv")).set_index("model")["auc_at_s1"]
    print("  exact means:", {SHORT[m]: round(float(stats[m][0]), 6) for m in ORDER}); print("  per-level rho (panel a):"); print(pd.DataFrame(rho, index=LV).T.round(4).to_string())
    print("  repeated runs (panel b):"); [print(f"    {m:20s} n={stats[m][2]:2d} mean {stats[m][0]:.4f} SD {stats[m][1]:.4f} | AUC at S1 {auc[m]:.4f}") for m in ORDER]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.56), gridspec_kw={"width_ratios": [1.4, 1]})
    lab = (lambda v: str(Decimal(str(round(v, 4))).quantize(Decimal("0.001"), ROUND_HALF_UP))) if a.round4 else (lambda v: f"{v:.3f}")
    x = np.arange(7)
    for m in ORDER: ax1.plot(x, rho[m], color=COL[m], ls=LS[m], marker=MK[m], lw=2.0, ms=5.5, mec="white", mew=0.6, label=SHORT[m])
    for y in (0.85, 0.70): ax1.axhline(y, color="#7f7f7f", ls=":", lw=1.0); ax1.text(-0.35, y + 0.006, f"{y:.2f}", fontsize=7.5, color="#555555", va="bottom")
    ax1.set_xticks(x); ax1.set_xticklabels(LV); ax1.set_ylim(0.38, 1.05); ax1.set_xlim(-0.4, 6.4)
    ax1.set_ylabel("Spearman $\\rho$ vs. S1 ranking"); ax1.set_xlabel("Scarcity level (older cycles removed)"); ax1.set_title("(a) Per-level stability, primary split", fontsize=10)
    ax1.grid(axis="y", alpha=0.35); ax1.legend(loc="lower left", ncol=4, frameon=False, fontsize=8, handlelength=2.2, columnspacing=1.0)
    xb = np.arange(4)
    for i, m in enumerate(ORDER):
        mu, sd, n = stats[m]; ax2.errorbar(i, mu, yerr=sd, fmt=MK[m], color=COL[m], ms=7, mec="white", mew=0.6, ecolor="#333333", elinewidth=1.3, capsize=4, zorder=3)
        ax2.text(i + 0.17, mu, lab(mu), fontsize=8, va="center")
    ax2.set_xticks(xb); ax2.set_xticklabels([f"{SHORT[m]}\nAUC {auc[m]:.2f}" for m in ORDER], fontsize=7); ax2.set_ylim(0.6, 1.0); ax2.set_xlim(-0.5, 3.95)
    ax2.set_ylabel("Mean ESI ($\\pm$ SD)"); ax2.set_title("(b) ESI over independent splits", fontsize=10); ax2.grid(axis="y", alpha=0.35)
    plt.tight_layout(); os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "Fig1_paper.png"), dpi=275, bbox_inches="tight"); fig.savefig(os.path.join(a.out, "Fig1_paper.pdf"), bbox_inches="tight")
    print("  saved", os.path.join(a.out, "Fig1_paper.png"), "and .pdf")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--out", default="outputs/figures"); ap.add_argument("--round4", action="store_true"); main(ap.parse_args())
