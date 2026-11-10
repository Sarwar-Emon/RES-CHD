"""
Version 2.  Answers the committee's questions about class imbalance.  Put this file in the Project folder (next to claims_experiments.py) and run from there.

  python imbalance_experiments.py --task threshold
  python imbalance_experiments.py --task methods
  python imbalance_experiments.py --task esi_methods --seeds 42 43 44

Final models are trained on the FULL S1 training data (like the thesis). Cutoffs come from 5-fold out-of-fold predictions on the training data; the test set is never used to pick a cutoff.
Outputs go to outputs/imbalance_v2/ .
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_experiments as ce
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FEATURES, CONT, LEVELS = ce.FEATURES, ce.CONT, ce.LEVELS
SEED = 42
warnings.filterwarnings("ignore")
METHODS = ["none", "smote30", "ros30", "classweight"]
LABEL = {"none": "no resampling", "smote30": "SMOTE 30% (thesis)", "ros30": "random oversampling 30%", "classweight": "class weights"}

def ros(X, y, cap=0.30):
    n_min, n_maj = int(y.sum()), int((y == 0).sum()); return RandomOverSampler(sampling_strategy=max(int(n_maj * cap), n_min) / n_maj, random_state=SEED).fit_resample(X, y)

def fit(name, method, X, y):
    m = ce.make_model(name)
    if method == "smote30": X, y = ce.apply_smote(X, y, 0.30)
    elif method == "ros30": X, y = ros(X, y)
    elif method == "classweight":
        if name == "LogisticRegression": m.set_params(class_weight="balanced")
        elif name == "RandomForest": m.set_params(class_weight="balanced_subsample")
        elif name == "XGBoost": m.set_params(scale_pos_weight=float((y == 0).sum() / max(int(y.sum()), 1)))
        else: return None
    return m.fit(X, y)

def at(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    rec, pre, spec = tp / max(tp + fn, 1), tp / max(tp + fp, 1), tn / max(tn + fp, 1)
    return dict(threshold=round(float(t), 4), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), recall=round(rec, 4), precision=round(pre, 4), specificity=round(spec, 4), F1=round(2 * pre * rec / max(pre + rec, 1e-9), 4))
def thr_f1(y, p):
    pr, rc, th = precision_recall_curve(y, p); f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-9); return th[int(np.argmax(f))]
def thr_recall(y, p, target):
    pr, rc, th = precision_recall_curve(y, p); ok = np.where(rc[:-1] >= target)[0]; return th[ok.max()] if len(ok) else th[0]

def load(a):
    D = os.path.join(a.base, "data", "scarcity_levels"); tr = pd.read_csv(os.path.join(D, "S1_train.csv")); te = pd.read_csv(os.path.join(D, "S1_test.csv"))
    return tr[FEATURES], tr["chd"].values, te[FEATURES], te["chd"].values

def oof(name, method, X, y, folds):
    p = np.zeros(len(y))
    for tr_i, va_i in StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, y):
        m = fit(name, method, X.iloc[tr_i], y[tr_i])
        if m is None: return None
        p[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    return p

def task_threshold(a):
    X, ytr, Xte, y = load(a); pt, po, rows = {}, {}, []
    for name in a.models:
        print(f"== {name}", flush=True); pt[name] = fit(name, "smote30", X, ytr).predict_proba(Xte)[:, 1]; po[name] = oof(name, "smote30", X, ytr, a.folds)
        rules = {"fixed 0.5": 0.5, "best F1 (cutoff from 5-fold out-of-fold training predictions)": thr_f1(ytr, po[name]), "recall about 0.5 (cutoff from out-of-fold)": thr_recall(ytr, po[name], 0.5), "recall about 0.7 (cutoff from out-of-fold)": thr_recall(ytr, po[name], 0.7)}
        for rule, t in rules.items(): rows.append(dict(model=name, rule=rule, **at(y, pt[name], t)))
        pr, rc, _ = precision_recall_curve(y, pt[name])
        for tgt in (0.5, 0.7): rows.append(dict(model=name, rule=f"best possible precision at recall >= {tgt} (test PR curve)", recall=tgt, precision=round(float(pr[rc >= tgt].max()), 4)))
    rng = np.random.default_rng(SEED); A = {n: [] for n in a.models}; P = {n: [] for n in a.models}
    for _ in range(a.boot):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].sum() == 0: continue
        for n in a.models: A[n].append(roc_auc_score(y[ix], pt[n][ix])); P[n].append(average_precision_score(y[ix], pt[n][ix]))
    ci = [dict(model=n, AUC=round(roc_auc_score(y, pt[n]), 4), AUC_95CI=f"{np.percentile(A[n], 2.5):.3f}-{np.percentile(A[n], 97.5):.3f}", PR_AUC=round(average_precision_score(y, pt[n]), 4), PR_AUC_95CI=f"{np.percentile(P[n], 2.5):.3f}-{np.percentile(P[n], 97.5):.3f}", chance_precision=round(float(y.mean()), 4)) for n in a.models]
    pairs = [("LogisticRegression", "MLP"), ("RandomForest", "MLP"), ("XGBoost", "MLP"), ("LogisticRegression", "RandomForest"), ("LogisticRegression", "XGBoost"), ("RandomForest", "XGBoost")]; dif = []
    for u, v in pairs:
        if u in A and v in A:
            for label, D_ in (("AUC", A), ("PR_AUC", P)):
                d = np.array(D_[u]) - np.array(D_[v]); dif.append(dict(comparison=f"{u} minus {v}", metric=label, mean_difference=round(float(d.mean()), 4), CI_95=f"{np.percentile(d, 2.5):.4f} to {np.percentile(d, 97.5):.4f}", share_of_resamples_first_is_higher=round(float((d > 0).mean()), 3)))
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "threshold_analysis.csv"), index=False); pd.DataFrame(ci).to_csv(os.path.join(a.outdir, "auc_prauc_intervals.csv"), index=False); pd.DataFrame(dif).to_csv(os.path.join(a.outdir, "paired_differences.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for n in a.models: pr, rc, _ = precision_recall_curve(y, pt[n]); ax.plot(rc, pr, lw=2, label=n)
    ax.axhline(float(y.mean()), color="gray", ls="--", label=f"chance ({y.mean():.3f})"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves (S1 test set)"); ax.legend(frameon=False); ax.set_ylim(0, 1)
    fig.savefig(os.path.join(a.outdir, "pr_curves.png"), dpi=200, bbox_inches="tight"); pd.set_option("display.width", 230)
    print("\n=== AUC and PR-AUC with 95% intervals ===\n" + pd.DataFrame(ci).to_string(index=False)); print("\n=== PAIRED DIFFERENCES (same resampled patients for both models) ===\n" + pd.DataFrame(dif).to_string(index=False)); print("\n=== CONFUSION MATRIX AT DIFFERENT CUTOFFS ===\n" + df.to_string(index=False))

def task_methods(a):
    X, ytr, Xte, y = load(a); rows = []
    for name in a.models:
        for meth in METHODS:
            m = fit(name, meth, X, ytr)
            if m is None: continue
            pt = m.predict_proba(Xte)[:, 1]; po = oof(name, meth, X, ytr, a.folds); t = thr_f1(ytr, po); b = at(y, pt, t); h = at(y, pt, 0.5)
            rows.append(dict(model=name, imbalance_method=LABEL[meth], AUC=round(roc_auc_score(y, pt), 4), PR_AUC=round(average_precision_score(y, pt), 4), recall_at_0_5=h["recall"], precision_at_0_5=h["precision"], F1_at_0_5=h["F1"], best_F1_cutoff=b["threshold"], recall_best_F1=b["recall"], precision_best_F1=b["precision"], F1_best=b["F1"]))
            print(f"   {name} | {LABEL[meth]} done", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "methods_comparison.csv"), index=False); pd.set_option("display.width", 250); print("\n=== IMBALANCE METHODS ===\n" + df.to_string(index=False))

def task_esi(a):
    global SEED
    D, s1_test, raw = ce.setup(a); rows = []
    for seed in a.seeds:
        SEED = seed; ce.SEED = seed
        for name in a.models:
            for meth in METHODS:
                sv = {}
                for lv in LEVELS:
                    tr = pd.read_csv(os.path.join(D, f"{lv}_train.csv")); sc = joblib.load(os.path.join(D, f"{lv}_scaler.pkl")); m = fit(name, meth, tr[FEATURES], tr["chd"].values)
                    if m is None: break
                    Xe = raw.copy(); Xe[CONT] = sc.transform(raw[CONT]); sv[lv] = ce.shap_matrix(m, name, tr[FEATURES], Xe, a.background)
                if len(sv) < len(LEVELS): continue
                imp = {lv: np.abs(sv[lv]).mean(0) for lv in LEVELS}; esi = float(np.mean([spearmanr(imp["S1"], imp[lv])[0] for lv in LEVELS[1:]]))
                rows.append(dict(seed=seed, model=name, imbalance_method=LABEL[meth], ESI=round(esi, 4))); print(f"   seed {seed} | {name} | {LABEL[meth]}: ESI {esi:.4f}", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "esi_by_imbalance_method.csv"), index=False)
    g = df.groupby(["model", "imbalance_method"])["ESI"]; s = pd.DataFrame({"mean": g.mean().round(4), "sd": g.std(ddof=1).round(4), "n_seeds": g.count()}); s.to_csv(os.path.join(a.outdir, "esi_by_imbalance_method_summary.csv"))
    print("\n=== ESI BY IMBALANCE METHOD (mean over seeds) ===\n" + df.groupby(["model", "imbalance_method"])["ESI"].mean().unstack().round(4).to_string()); print("\n=== SD over seeds ===\n" + df.groupby(["model", "imbalance_method"])["ESI"].std(ddof=1).unstack().round(4).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=["threshold", "methods", "esi_methods"], required=True); ap.add_argument("--base", default=".")
    ap.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForest", "XGBoost", "MLP"], choices=["XGBoost", "RandomForest", "LogisticRegression", "MLP"])
    ap.add_argument("--boot", type=int, default=500); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--background", type=int, default=100); ap.add_argument("--outdir", default="outputs/imbalance_v2")
    a = ap.parse_args(); {"threshold": task_threshold, "methods": task_methods, "esi_methods": task_esi}[a.task](a)
