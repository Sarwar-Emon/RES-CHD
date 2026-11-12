import argparse, os, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]   # same order as phase1/phase3
CONT = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
LEVELS = [f"S{i}" for i in range(1, 8)]
ALL_MODELS = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]

def shap_2d(sv):
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv

def explain(model, name, Xtrain, Xp, n_bg, seed):
    if name in ("XGBoost", "RandomForest"):
        return shap_2d(shap.TreeExplainer(model).shap_values(Xp))
    if name == "LogisticRegression":
        bg = Xtrain.mean(axis=0).values.reshape(1, -1)               # same background as phase3
        return shap_2d(shap.LinearExplainer(model, bg).shap_values(Xp))
    rng = np.random.default_rng(seed)
    bg = Xtrain.iloc[rng.choice(len(Xtrain), size=min(n_bg, len(Xtrain)), replace=False)]
    f = lambda z: model.predict_proba(pd.DataFrame(z, columns=FEATURES))[:, 1]
    np.random.seed(seed)
    return shap_2d(shap.KernelExplainer(f, bg).shap_values(Xp, nsamples="auto", silent=True))

def cosine_abs(a, b):
    a, b = np.abs(a), np.abs(b)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    na[na == 0] = 1; nb[nb == 0] = 1
    return np.clip((a * b).sum(axis=1) / (na * nb), 0, 1)

def main(a):
    D, M = os.path.join(a.base, "data", "scarcity_levels"), os.path.join(a.base, "models")
    s1 = pd.read_csv(os.path.join(D, "S1_test.csv"))
    chd = s1["chd"].values
    idx = np.arange(len(s1))
    if 0 < a.n_patients < len(s1):
        idx, _ = train_test_split(idx, train_size=a.n_patients, stratify=chd, random_state=42)
    idx = np.sort(idx); fixed = s1.iloc[idx].reset_index(drop=True)
    print(f"Fixed patient set: {len(fixed)} patients from the S1 test set (CHD cases: {int(fixed['chd'].sum())})")

    sc1 = joblib.load(os.path.join(D, "S1_scaler.pkl"))
    raw = fixed[FEATURES].copy()
    raw[CONT] = sc1.inverse_transform(fixed[CONT])                    # back to real units
    out_rows, summary, prof_rows = [], [], []

    for name in a.models:
        print(f"\n=== {name} ===", flush=True)
        phi, proba = {}, {}
        for lv in LEVELS:
            sc = joblib.load(os.path.join(D, f"{lv}_scaler.pkl"))
            Xp = raw.copy(); Xp[CONT] = sc.transform(raw[CONT])        # same patients, this level's scaling
            model = joblib.load(os.path.join(M, f"{lv}_{name}.pkl"))
            Xtr = pd.read_csv(os.path.join(D, f"{lv}_train.csv"))[FEATURES]
            phi[lv] = explain(model, name, Xtr, Xp, a.background, 42)
            proba[lv] = model.predict_proba(Xp)[:, 1]
            print(f"  {lv} explained", flush=True)
        lss = {lv: cosine_abs(phi["S1"], phi[lv]) for lv in LEVELS[1:]}
        mean_lss = np.mean([lss[lv] for lv in LEVELS[1:]], axis=0)
        n = len(mean_lss)
        stable, volatile = mean_lss >= a.stable, mean_lss < a.volatile
        print(f"  mean LSS {mean_lss.mean():.3f} | stable (>= {a.stable}) {stable.sum()}/{n} ({stable.mean():.1%}) | "
              f"volatile (< {a.volatile}) {volatile.sum()}/{n} ({volatile.mean():.1%})")

        # compound risk, thesis definition: Conf = 2|p-0.5|, median splits
        def quad(conf, l):
            hc, hl = conf >= np.median(conf), l >= np.median(l)
            return {"Q1": (hc & hl).mean(), "Q2": (hc & ~hl).mean(), "Q3": (~hc & hl).mean(), "Q4": (~hc & ~hl).mean()}
        q_s1 = quad(2 * np.abs(proba["S1"] - 0.5), mean_lss)
        q_s7 = quad(2 * np.abs(proba["S7"] - 0.5), lss["S7"])
        print("  Q-shares  S1 conf x mean LSS :", {k: f"{v:.1%}" for k, v in q_s1.items()})
        print("  Q-shares  S7 conf x LSS(S7) :", {k: f"{v:.1%}" for k, v in q_s7.items()})
        summary.append(dict(model=name, n=n, mean_lss=round(mean_lss.mean(), 4), pct_stable=round(stable.mean() * 100, 1),
                            pct_volatile=round(volatile.mean() * 100, 1), n_volatile=int(volatile.sum()),
                            **{f"S1_{k}": round(v * 100, 1) for k, v in q_s1.items()}, **{f"S7_{k}": round(v * 100, 1) for k, v in q_s7.items()}))
        pat = raw.copy(); pat.insert(0, "model", name); pat["chd"] = fixed["chd"].values
        pat["mean_lss"] = mean_lss; pat["p_S1"] = proba["S1"]; pat["conf_S1"] = 2 * np.abs(proba["S1"] - 0.5)
        for lv in LEVELS[1:]: pat[f"lss_{lv}"] = lss[lv]
        out_rows.append(pat)

        # profiling volatile vs stable (real units)
        if volatile.sum() >= 10 and stable.sum() >= 10:
            print(f"  Volatile vs stable (Mann-Whitney / chi-square):")
            for f in FEATURES:
                v, s = raw.loc[volatile, f], raw.loc[stable, f]
                if f in CONT:
                    p = mannwhitneyu(v, s).pvalue; lab = f"{v.mean():.1f} vs {s.mean():.1f}"
                else:
                    tab = [[(v == 1).sum(), (v == 0).sum()], [(s == 1).sum(), (s == 0).sum()]]
                    p = chi2_contingency(tab)[1]; lab = f"{v.mean():.1%} vs {s.mean():.1%}"
                prof_rows.append(dict(model=name, feature=f, volatile_vs_stable=lab, p_value=round(float(p), 4)))
                print(f"    {f:<11} {lab:<18} p = {p:.4f}")
        else:
            print("  (fewer than 10 volatile or stable patients -> no profiling for this model)")

    os.makedirs(a.outdir, exist_ok=True)
    pd.concat(out_rows).to_csv(os.path.join(a.outdir, "patient_lss_aligned.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(a.outdir, "lss_summary_aligned.csv"), index=False)
    if prof_rows: pd.DataFrame(prof_rows).to_csv(os.path.join(a.outdir, "profiling_aligned.csv"), index=False)
    print("\n=== SUMMARY ===\n" + pd.DataFrame(summary).to_string(index=False))
    print(f"\nSaved to {a.outdir}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=".")
    ap.add_argument("--n-patients", type=int, default=500); ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--stable", type=float, default=0.85, help="thesis uses 0.85; phase5d code used 0.90")
    ap.add_argument("--volatile", type=float, default=0.70); ap.add_argument("--outdir", default="outputs/lss_aligned")
    main(ap.parse_args())
