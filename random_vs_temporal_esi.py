"""
REAL matched experiment for RQ2: does TEMPORAL scarcity hurt explanation stability more than RANDOM scarcity of the same size?

For every level S_k (k = 2..7) and model:
  temporal : your saved level-k model (trained on the cycles that remain)
  random   : a NEW model trained on a random, stratified subsample of S1_train with exactly the same number of patients
             (same preprocessing as phase2: scaler on train only, SMOTE capped at 30%, same hyper-parameters)
Both are explained on the SAME fixed patients (from the S1 test set), and compared with the SAME S1 baseline ranking.
rho = Spearman between the S1 importance ranking and the level-k importance ranking. ESI = mean rho over S2..S7.

Run from the Project folder (needs data/scarcity_levels and models/):
    python random_vs_temporal_esi.py
    python random_vs_temporal_esi.py --models XGBoost RandomForest LogisticRegression --reps 5     # faster, skip MLP
Runtime: roughly 20-60 minutes with all four models (MLP uses exact KernelSHAP).
"""
import argparse, os, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
CONT = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
ALL_MODELS = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
SEED = 42

def make_model(name):     # identical to MODEL_CONFIG in phase2_training.py
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             eval_metric="auc", random_state=SEED, n_jobs=-1)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10, random_state=SEED)

def apply_smote(X, y):    # identical logic to phase2_training.apply_smote
    n_min, n_maj = int(y.sum()), int((y == 0).sum()); k = min(5, n_min - 1)
    if k < 1: return X, y
    target = max(int(n_maj * 0.30), n_min)
    return SMOTE(sampling_strategy=target / n_maj, k_neighbors=k, random_state=SEED).fit_resample(X, y)

def shap_2d(sv):
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv); return sv[:, :, 1] if sv.ndim == 3 else sv

def importance(model, name, Xtrain, Xp, n_bg):
    if name in ("XGBoost", "RandomForest"):
        sv = shap_2d(shap.TreeExplainer(model).shap_values(Xp))
    elif name == "LogisticRegression":
        sv = shap_2d(shap.LinearExplainer(model, Xtrain.mean(axis=0).values.reshape(1, -1)).shap_values(Xp))
    else:
        rng = np.random.default_rng(SEED)
        bg = Xtrain.iloc[rng.choice(len(Xtrain), size=min(n_bg, len(Xtrain)), replace=False)]
        f = lambda z: model.predict_proba(pd.DataFrame(z, columns=FEATURES))[:, 1]
        np.random.seed(SEED)
        sv = shap_2d(shap.KernelExplainer(f, bg).shap_values(Xp, nsamples="auto", silent=True))
    return np.abs(sv).mean(axis=0)

def main(a):
    D, M = os.path.join(a.base, "data", "scarcity_levels"), os.path.join(a.base, "models")
    s1_test = pd.read_csv(os.path.join(D, "S1_test.csv"))
    ix = np.arange(len(s1_test))
    if 0 < a.n_eval < len(s1_test):
        ix, _ = train_test_split(ix, train_size=a.n_eval, stratify=s1_test["chd"], random_state=SEED)
    sc1 = joblib.load(os.path.join(D, "S1_scaler.pkl"))
    ev_raw = s1_test.iloc[np.sort(ix)][FEATURES].reset_index(drop=True); ev_raw[CONT] = sc1.inverse_transform(ev_raw[CONT])
    tr1 = pd.read_csv(os.path.join(D, "S1_train.csv")); raw_tr = tr1.copy(); raw_tr[CONT] = sc1.inverse_transform(tr1[CONT])
    print(f"Evaluation patients: {len(ev_raw)} (fixed, from the S1 test set)")
    rows = []
    for name in a.models:
        print(f"\n=== {name} ===", flush=True)
        imp1 = importance(joblib.load(os.path.join(M, f"S1_{name}.pkl")), name, tr1[FEATURES], ev_raw.assign(**{c: sc1.transform(ev_raw[CONT])[:, i] for i, c in enumerate(CONT)}), a.background)
        for k in range(2, 8):
            lv = f"S{k}"; sck = joblib.load(os.path.join(D, f"{lv}_scaler.pkl")); trk = pd.read_csv(os.path.join(D, f"{lv}_train.csv"))
            Xp = ev_raw.copy(); Xp[CONT] = sck.transform(ev_raw[CONT])
            rho_t = spearmanr(imp1, importance(joblib.load(os.path.join(M, f"{lv}_{name}.pkl")), name, trk[FEATURES], Xp, a.background))[0]
            rows.append(dict(model=name, level=lv, condition="temporal", rep=0, n_train=len(trk), rho=float(rho_t)))
            rr = []
            for r in range(a.reps):
                sub, _ = train_test_split(raw_tr, train_size=len(trk), stratify=raw_tr["chd"], random_state=1000 * r + k)
                scr = StandardScaler().fit(sub[CONT]); Xs = sub[FEATURES].copy(); Xs[CONT] = scr.transform(sub[CONT])
                Xb, yb = apply_smote(Xs, sub["chd"]); m = make_model(name).fit(Xb, yb)
                Xpr = ev_raw.copy(); Xpr[CONT] = scr.transform(ev_raw[CONT])
                rho_r = spearmanr(imp1, importance(m, name, Xs, Xpr, a.background))[0]; rr.append(float(rho_r))
                rows.append(dict(model=name, level=lv, condition="random", rep=r + 1, n_train=len(trk), rho=float(rho_r)))
            print(f"  {lv} (n_train={len(trk):,}): temporal rho = {rho_t:.3f} | random rho = {np.mean(rr):.3f} (+/- {np.std(rr):.3f})", flush=True)
    df = pd.DataFrame(rows); os.makedirs(a.outdir, exist_ok=True); df.to_csv(os.path.join(a.outdir, "random_vs_temporal_rho.csv"), index=False)
    out = []
    for name in a.models:
        d = df[df.model == name]; t = d[d.condition == "temporal"].rho.mean()
        rnd = d[d.condition == "random"].groupby("rep").rho.mean()
        out.append(dict(model=name, ESI_temporal=round(t, 4), ESI_random_mean=round(rnd.mean(), 4), ESI_random_SD=round(rnd.std(ddof=1) if len(rnd) > 1 else np.nan, 4),
                        difference=round(rnd.mean() - t, 4)))
    s = pd.DataFrame(out); s.to_csv(os.path.join(a.outdir, "random_vs_temporal_summary.csv"), index=False)
    print("\n=== SUMMARY (same patients, same baseline, same training size) ===\n" + s.to_string(index=False))
    print("\ndifference = random minus temporal. A clearly positive difference means temporal scarcity hurts stability MORE than random scarcity of the same size.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(a.models), figsize=(4.6 * len(a.models), 4.2), sharey=True)
        for ax, name in zip(np.atleast_1d(axes), a.models):
            d = df[df.model == name]; lv = sorted(d.level.unique())
            t = d[d.condition == "temporal"].set_index("level").rho.reindex(lv)
            r = d[d.condition == "random"].groupby("level").rho.agg(["mean", "std"]).reindex(lv)
            ax.plot(lv, t.values, "o-", color="#C0392B", lw=2, label="Temporal removal"); ax.errorbar(lv, r["mean"], yerr=r["std"].fillna(0), fmt="s--", color="#2E75B6", lw=2, capsize=3, label="Random subsample")
            ax.set_title(name, fontweight="bold"); ax.set_xlabel("Scarcity level"); ax.grid(alpha=.25, ls=":")
        np.atleast_1d(axes)[0].set_ylabel("Spearman rho vs S1"); np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.suptitle("Explanation stability: temporal vs random scarcity (matched training size)", fontweight="bold"); plt.tight_layout()
        fig.savefig(os.path.join(a.outdir, "random_vs_temporal.png"), dpi=200)
    except Exception as e:
        print("plot skipped:", e)
    print("Saved to", a.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=".")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--background", type=int, default=100); ap.add_argument("--outdir", default="outputs/random_vs_temporal")
    main(ap.parse_args())
