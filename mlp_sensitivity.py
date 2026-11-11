"""
MLP sensitivity study for RES-CHD: activation function, depth, seed noise, SHAP-approximation noise.

QUICK mode (minutes): AUC, ECE, confusion matrix at S1 for every MLP configuration, 3 seeds.
    python3 mlp_sensitivity.py nhanes_2005_2018_unified.csv --mode quick

FULL mode (roughly 15-40 min per configuration; use --configs to pick some): also ESI (S1 vs S2..S7) with 95% bootstrap CI for every configuration, plus two
"noise floor" checks:  (a) explainer noise  = same model explained twice with different random draws,
                       (b) training-seed noise = model retrained with another seed.
    python3 mlp_sensitivity.py nhanes_2005_2018_unified.csv --mode full
    (use --configs 0 1 6 to run only some configurations, --n-explain 200 to go faster;
     SHAP uses nsamples='auto' = all 254 coalitions = exact KernelSHAP for 8 features. Use --nsamples 100 to mimic the thesis.)

Design (leakage-free): the S1 test set is fixed. Level S_k trains on the S1-TRAIN patients whose cycle is
within the last (8-k) cycles. SHAP is always computed for the same patients from the fixed test set.
NOTE: the thesis built a separate 80/20 split per level, so the baseline here can differ slightly from
the thesis ESI (0.6576). Compare configurations to the baseline row of THIS script, not to the thesis.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["age", "sex", "sbp", "dbp", "hdl", "total_chol", "bmi", "smoking"]
TARGET, CONT = "chd", ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
CONFIGS = [  # (label, activation, hidden layers)
    ("relu (64,32)  [thesis]", "relu", (64, 32)),
    ("tanh (64,32)", "tanh", (64, 32)),
    ("logistic (64,32)", "logistic", (64, 32)),
    ("relu 1 layer (64)", "relu", (64,)),
    ("relu 3 layers (64,32,16)", "relu", (64, 32, 16)),
    ("relu 4 layers (64,64,32,32)", "relu", (64, 64, 32, 32)),
    ("relu 6 layers (64,64,32,32,16,16)", "relu", (64, 64, 32, 32, 16, 16)),
]

def ece(y, p, bins=10):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def fit_level(train, act, hidden, seed, winsorize):
    X, y = train[FEATURES].copy(), train[TARGET].astype(int)
    lo = hi = None
    if winsorize:
        lo, hi = X[CONT].quantile(0.01), X[CONT].quantile(0.99)
        X[CONT] = X[CONT].clip(lo, hi, axis=1)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    Xb, yb = SMOTE(sampling_strategy=0.30, k_neighbors=5, random_state=seed).fit_resample(Xs, y)
    m = MLPClassifier(hidden_layer_sizes=hidden, activation=act, early_stopping=True,
                      max_iter=200, random_state=seed).fit(Xb, yb)
    def tf(Xnew):
        Z = Xnew[FEATURES].copy()
        if winsorize:
            Z[CONT] = Z[CONT].clip(lo, hi, axis=1)
        return sc.transform(Z)
    return m, tf, Xs

def explain(model, Xbg, Xexp, seed, nsamples="auto"):
    import shap
    np.random.seed(seed)
    ex = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], Xbg)
    sv = ex.shap_values(Xexp, nsamples=nsamples, silent=True)
    if isinstance(sv, list): sv = sv[-1]
    sv = np.asarray(sv)
    return sv[:, :, -1] if sv.ndim == 3 else sv

def rho(a, b):
    return float(spearmanr(a, b)[0])

def run(a):
    df = pd.read_csv(a.csv)[FEATURES + [TARGET, "cycle"]].dropna()
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    cycles = sorted(df["cycle"].unique())
    os.makedirs(a.outdir, exist_ok=True)
    chosen = [CONFIGS[i] for i in (a.configs if a.configs else range(len(CONFIGS)))]
    print(f"train {len(train):,} | test {len(test):,} | cycles {cycles[0]}..{cycles[-1]}")

    # ---------------- QUICK: AUC / ECE / confusion matrix at S1 ----------------
    q = []
    for label, act, hid in chosen:
        rows = []
        for s in range(a.seeds):
            m, tf, _ = fit_level(train, act, hid, SEED + s, not a.no_winsorize)
            p = m.predict_proba(tf(test))[:, 1]; y = test[TARGET].values
            tn, fp, fn, tp = confusion_matrix(y, (p >= a.threshold).astype(int), labels=[0, 1]).ravel()
            rows.append(dict(auc=roc_auc_score(y, p), ece=ece(y, p), TN=tn, FP=fp, FN=fn, TP=tp,
                             precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
        r = pd.DataFrame(rows)
        q.append(dict(config=label, AUC=f"{r.auc.mean():.4f} +/- {r.auc.std(ddof=0):.4f}",
                      ECE=f"{r.ece.mean():.4f} +/- {r.ece.std(ddof=0):.4f}",
                      TN=r.TN.mean().round(1), FP=r.FP.mean().round(1), FN=r.FN.mean().round(1), TP=r.TP.mean().round(1),
                      precision=round(r.precision.mean(), 4), recall=round(r.recall.mean(), 4)))
        print(q[-1])
    qdf = pd.DataFrame(q); qdf.to_csv(os.path.join(a.outdir, "mlp_sensitivity_quick.csv"), index=False)
    print("\n=== QUICK RESULTS (S1, threshold %.2f, %d seeds) ===" % (a.threshold, a.seeds))
    print(qdf.to_string(index=False))
    if a.mode == "quick":
        print("\nSaved mlp_sensitivity_quick.csv"); return

    # ---------------- FULL: ESI + noise floors ----------------
    rng = np.random.default_rng(SEED)
    _, ex_idx = train_test_split(np.arange(len(test)), train_size=min(a.n_explain, len(test) - 1),
                                 stratify=test[TARGET].values, random_state=SEED)
    texp = test.iloc[ex_idx]
    out = []
    for label, act, hid in chosen:
        print(f"\n--- ESI for {label} ---", flush=True)
        imps, svs = [], []
        for k in range(1, 8):
            tr = train[train["cycle"].isin(cycles[k - 1:])]
            m, tf, Xs = fit_level(train=tr, act=act, hidden=hid, seed=SEED, winsorize=not a.no_winsorize)
            Xbg = Xs[rng.choice(len(Xs), min(a.background, len(Xs)), replace=False)]
            sv = explain(m, Xbg, tf(texp), SEED, a.ns)
            svs.append(sv); imps.append(np.abs(sv).mean(0))
            print(f"  S{k}: train n={len(tr):,}", flush=True)
            if k == 1:   # noise floors at S1
                sv_b = explain(m, Xbg, tf(texp), SEED + 100, a.ns)
                rho_expl = rho(imps[0], np.abs(sv_b).mean(0))
                m2, tf2, Xs2 = fit_level(tr, act, hid, SEED + 1, not a.no_winsorize)
                sv_c = explain(m2, Xs2[rng.choice(len(Xs2), min(a.background, len(Xs2)), replace=False)], tf2(texp), SEED, a.ns)
                rho_seed = rho(imps[0], np.abs(sv_c).mean(0))
        rhos = [rho(imps[0], imps[k]) for k in range(1, 7)]
        esi = float(np.mean(rhos))
        boots = []
        for _ in range(a.boot):
            ix = rng.integers(0, len(texp), len(texp))
            im = [np.abs(s[ix]).mean(0) for s in svs]
            boots.append(np.mean([rho(im[0], im[k]) for k in range(1, 7)]))
        lo, hi = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
        out.append(dict(config=label, ESI=round(esi, 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                        **{f"rho_S{k+2}": round(r, 3) for k, r in enumerate(rhos)},
                        explainer_noise_rho=round(rho_expl, 3), training_seed_rho=round(rho_seed, 3)))
        print(out[-1], flush=True)
        pd.DataFrame(out).to_csv(os.path.join(a.outdir, "mlp_sensitivity_esi.csv"), index=False)
    res = pd.DataFrame(out)
    print("\n=== FULL RESULTS (ESI) ===\n" + res.to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.8))
        y = np.arange(len(res))[::-1]
        ax.barh(y, res.ESI, xerr=[res.ESI - res.CI_low, res.CI_high - res.ESI], color="#2E5FA3", capsize=3)
        ax.axvline(0.85, ls="--", c="green"); ax.axvline(0.70, ls="--", c="orange")
        ax.set_yticks(y); ax.set_yticklabels(res.config); ax.set_xlabel("ESI (95% bootstrap CI)")
        ax.set_title("MLP architecture sensitivity: ESI by activation function and depth"); plt.tight_layout()
        fig.savefig(os.path.join(a.outdir, "mlp_sensitivity_esi.png"), dpi=200)
    except Exception as e:
        print("plot skipped:", e)
    print("\nSaved CSVs/PNG to", a.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--configs", type=int, nargs="*"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--no-winsorize", action="store_true")
    ap.add_argument("--n-explain", type=int, default=500); ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--nsamples", default="auto"); ap.add_argument("--outdir", default="outputs/mlp_sensitivity")
    args = ap.parse_args(); args.ns = int(args.nsamples) if str(args.nsamples).isdigit() else "auto"
    run(args)
