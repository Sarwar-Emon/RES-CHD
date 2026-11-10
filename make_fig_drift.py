import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(a):
    d = pd.read_csv(os.path.join(a.base, "results/phase5b/random_vs_temporal_drift.csv"))
    x = np.arange(len(d)); lab = [f"{l}\n({int(n):,})" for l, n in zip(d.level, d.n_patients)]
    T, R = "#C0392B", "#7F8C8D"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
    ax = axs[0]; ax.bar(x, d.wass_ratio_t_over_r, color=T, width=0.62, zorder=3)
    for xi, v in zip(x, d.wass_ratio_t_over_r): ax.text(xi, v + 0.12, f"{v:.1f}x", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1, color="#333333", lw=1, ls=":"); ax.text(-0.45, 1.12, "equal to random (1x)", fontsize=8.5, ha="left", color="#333333")
    ax.set_ylim(0, 7.6); ax.set_ylabel("Wasserstein distance: temporal / random"); ax.set_title("(a) Drift ratio", fontweight="bold", loc="left")
    ax = axs[1]; w = 0.36
    ax.bar(x - w / 2, d.temporal_sig_features, w, color=T, label="Temporal removal", zorder=3); ax.bar(x + w / 2, d.random_sig_features_mean, w, color=R, label="Random subsampling", zorder=3)
    for xi, v in zip(x, d.temporal_sig_features): ax.text(xi - w / 2, v + 0.15, f"{int(v)}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 8.6); ax.set_yticks(range(0, 9)); ax.set_ylabel("Features with significant KS drift (of 8)"); ax.set_title("(b) Significant drift", fontweight="bold", loc="left"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axs[2]
    ax.bar(x - w / 2, d.temporal_psi_mean, w, color=T, zorder=3); ax.bar(x + w / 2, d.random_psi_mean, w, color=R, zorder=3)
    ax.axhline(0.10, color="#E0A000", ls="--", lw=1.4); ax.text(len(d) - 0.5, 0.104, "0.10: limit for a negligible shift", fontsize=8.5, ha="right", color="#B07800")
    for xi, v in zip(x, d.temporal_psi_mean): ax.text(xi - w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8.5, rotation=90, va="bottom")
    ax.set_ylim(0, 0.125); ax.set_ylabel("Mean PSI over the 8 features"); ax.set_title("(c) Size of the shift (PSI)", fontweight="bold", loc="left")
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9); ax.set_xlabel("Scarcity level (patients in the level)"); ax.grid(axis="y", alpha=0.25, ls=":", zorder=0)
    plt.tight_layout(); os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "Fig4_2_Drift.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(a.out, "Fig4_2_Drift.pdf"), bbox_inches="tight")
    print("saved", os.path.join(a.out, "Fig4_2_Drift.png"), "and .pdf"); print(d[["level", "n_patients", "wass_ratio_t_over_r", "temporal_sig_features", "random_sig_features_mean", "temporal_psi_mean"]].round(4).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--out", default="outputs/figures"); main(ap.parse_args())
