"""
RES-CHD — Master Consolidated Results Table
============================================
Builds a single comprehensive CSV joining all results from all phases.
Useful for writing the results section and for supplementary materials.

Outputs:
  results/master_results_table.csv   — all metrics per model per level
  results/master_summary_table.csv   — one row per model (aggregated)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
RESULTS  = BASE_DIR / "results"
SHAP_DIR = BASE_DIR / "shap"

MODELS  = ["XGBoost","RandomForest","LogisticRegression","MLP"]
LEVELS  = [f"S{i}" for i in range(1, 8)]


def safe_load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    print("Building master consolidated results table ...")

    # ── Load all source files ──────────────────────────────────────────────
    metrics_df  = safe_load(RESULTS / "metrics_per_level.csv")
    esi_df      = safe_load(SHAP_DIR / "stability" / "esi_scores.csv")
    boot_df     = safe_load(RESULTS / "phase4" / "bootstrapped_esi.csv")
    flags_df    = safe_load(RESULTS / "phase4" / "reliability_flags.csv")
    thresh_df   = safe_load(RESULTS / "phase4" / "scarcity_thresholds.csv")
    feat_df     = safe_load(RESULTS / "phase4" / "feature_reliability.csv")
    cal_df      = safe_load(RESULTS / "phase5f" / "calibration_metrics.csv")
    csi_df      = safe_load(RESULTS / "phase5f" / "calibration_stability.csv")
    lss_df      = safe_load(RESULTS / "phase5d" / "model_local_esi.csv")
    risk_df     = safe_load(RESULTS / "phase5e" / "compound_risk_by_level.csv")
    clev_df     = safe_load(RESULTS / "cleveland" / "esi_scores.csv")

    # ── Build per-level table ──────────────────────────────────────────────
    rows = []
    for model in MODELS:
        for level in LEVELS:
            row = {"model": model, "level": level}

            # Predictive performance
            if not metrics_df.empty:
                m = metrics_df[(metrics_df["model"]==model) &
                               (metrics_df["level"]==level)]
                row["auc"]    = float(m["auc"].values[0])    if len(m)>0 else None
                row["pr_auc"] = float(m["pr_auc"].values[0]) if len(m)>0 else None
                row["f1"]     = float(m["f1"].values[0])     if len(m)>0 else None

            # Global ESI (point estimate)
            if not esi_df.empty:
                e = esi_df[esi_df["model"]==model]
                row["esi_point"] = float(e["esi_score"].values[0]) if len(e)>0 else None

            # Bootstrapped ESI
            if not boot_df.empty:
                b = boot_df[boot_df["model"]==model]
                row["esi_boot"]       = float(b["esi_mean"].values[0])     if len(b)>0 else None
                row["esi_ci_lower"]   = float(b["esi_ci_lower"].values[0]) if len(b)>0 else None
                row["esi_ci_upper"]   = float(b["esi_ci_upper"].values[0]) if len(b)>0 else None
                row["reliability"]    = b["reliability"].values[0]          if len(b)>0 else None

            # Reliability flag
            if not flags_df.empty:
                f = flags_df[(flags_df["model"]==model) & (flags_df["level"]==level)]
                row["flag"] = f["flag"].values[0] if len(f)>0 else None

            # Calibration
            if not cal_df.empty:
                c = cal_df[(cal_df["model"]==model) & (cal_df["level"]==level)]
                row["brier_score"] = float(c["brier_score"].values[0]) if len(c)>0 else None
                row["ece"]         = float(c["ece"].values[0])          if len(c)>0 else None
                row["cal_quality"] = c["cal_quality"].values[0]          if len(c)>0 else None

            # Calibration stability
            if not csi_df.empty and level != "S1":
                cs = csi_df[(csi_df["model"]==model) & (csi_df["level"]==level)]
                row["csi"] = float(cs["csi"].values[0]) if len(cs)>0 else None

            # Compound risk
            if not risk_df.empty:
                r = risk_df[(risk_df["model"]==model) & (risk_df["level"]==level)]
                row["pct_compound_risk"] = float(r["pct_compound_risk"].values[0]) if len(r)>0 else None
                row["pct_safe"]          = float(r["pct_safe"].values[0])           if len(r)>0 else None

            rows.append(row)

    master_df = pd.DataFrame(rows)
    master_df.to_csv(RESULTS / "master_results_table.csv", index=False)
    print(f"  Saved master_results_table.csv ({len(master_df)} rows)")

    # ── Build one-row-per-model summary ────────────────────────────────────
    summary_rows = []
    for model in MODELS:
        m_df = master_df[master_df["model"]==model]

        # Cleveland ESI
        clev_esi = None
        if not clev_df.empty:
            c = clev_df[clev_df["model"]==model]
            clev_esi = float(c["esi"].values[0]) if len(c)>0 else None

        # Threshold
        thresh_level = thresh_n = "N/A"
        if not thresh_df.empty:
            t = thresh_df[thresh_df["model"]==model]
            if len(t)>0:
                thresh_level = t["first_drop_below_level"].values[0]
                thresh_n     = t["first_drop_below_n"].values[0]

        # Local stability
        lss_mean = lss_pct_stable = lss_pct_volatile = None
        if not lss_df.empty:
            l = lss_df[lss_df["model"]==model]
            if len(l)>0:
                lss_mean        = float(l["mean_lss"].values[0])
                lss_pct_stable  = float(l["pct_stable"].values[0])
                lss_pct_volatile= float(l["pct_volatile"].values[0])

        summary_rows.append({
            "model":                  model,
            "auc_at_s1":              m_df[m_df["level"]=="S1"]["auc"].values[0]
                                      if len(m_df[m_df["level"]=="S1"])>0 else None,
            "prauc_at_s1":            m_df[m_df["level"]=="S1"]["pr_auc"].values[0]
                                      if len(m_df[m_df["level"]=="S1"])>0 else None,
            "esi_point":              m_df["esi_point"].iloc[0],
            "esi_bootstrapped":       m_df["esi_boot"].iloc[0],
            "esi_ci_lower":           m_df["esi_ci_lower"].iloc[0],
            "esi_ci_upper":           m_df["esi_ci_upper"].iloc[0],
            "overall_reliability":    m_df["reliability"].iloc[0],
            "scarcity_threshold":     f"{thresh_level} (n={int(thresh_n):,})"
                                      if thresh_n not in ["N/A", None] and
                                         not (isinstance(thresh_n, float) and
                                              np.isnan(thresh_n))
                                      else thresh_level,
            "green_levels":           int((m_df["flag"]=="GREEN").sum()),
            "amber_levels":           int((m_df["flag"]=="AMBER").sum()),
            "red_levels":             int((m_df["flag"]=="RED").sum()),
            "ece_at_s1":              m_df[m_df["level"]=="S1"]["ece"].values[0]
                                      if len(m_df[m_df["level"]=="S1"])>0 else None,
            "ece_at_s7":              m_df[m_df["level"]=="S7"]["ece"].values[0]
                                      if len(m_df[m_df["level"]=="S7"])>0 else None,
            "local_lss_mean":         lss_mean,
            "local_pct_stable":       lss_pct_stable,
            "local_pct_volatile":     lss_pct_volatile,
            "compound_risk_s1_pct":   m_df[m_df["level"]=="S1"]["pct_compound_risk"].values[0]
                                      if len(m_df[m_df["level"]=="S1"])>0 else None,
            "compound_risk_s7_pct":   m_df[m_df["level"]=="S7"]["pct_compound_risk"].values[0]
                                      if len(m_df[m_df["level"]=="S7"])>0 else None,
            "cleveland_esi":          clev_esi,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / "master_summary_table.csv", index=False)
    print(f"  Saved master_summary_table.csv")

    print(f"\n  Master summary:")
    cols = ["model","auc_at_s1","esi_bootstrapped","overall_reliability",
            "local_lss_mean","compound_risk_s7_pct","cleveland_esi"]
    print(f"\n{summary_df[cols].to_string(index=False)}")
    print("\nMaster results table complete. ✓")


if __name__ == "__main__":
    main()