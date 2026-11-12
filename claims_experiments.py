
import argparse, os, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")
FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]; CONT = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
LEVELS = [f"S{i}" for i in range(1, 8)]; SEED = 42

def make_model(name):
    if name == "XGBoost": return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=SEED, n_jobs=-1)
    if name == "RandomForest": return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1)
    if name == "LogisticRegression": return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=SEED)

def apply_smote(X, y, cap):
    if cap is None: return X, y
    n_min, n_maj = int(y.sum()), int((y == 0).sum()); k = min(5, n_min - 1)
    if k < 1: return X, y
    return SMOTE(sampling_strategy=max(int(n_maj * cap), n_min) / n_maj, k_neighbors=k, random_state=SEED).fit_resample(X, y)

def shap_matrix(model, name, Xtrain, Xe, n_bg):
    def s2(sv):
        if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv); return sv[:, :, 1] if sv.ndim == 3 else sv
    if name in ("XGBoost", "RandomForest"): return s2(shap.TreeExplainer(model).shap_values(Xe))
    if name == "LogisticRegression": return s2(shap.LinearExplainer(model, Xtrain.mean(axis=0).values.reshape(1, -1)).shap_values(Xe))
    rng = np.random.default_rng(SEED); bg = Xtrain.iloc[rng.choice(len(Xtrain), size=min(n_bg, len(Xtrain)), replace=False)]
    f = lambda z: model.predict_proba(pd.DataFrame(z, columns=FEATURES))[:, 1]; np.random.seed(SEED)
    return s2(shap.KernelExplainer(f, bg).shap_values(Xe, nsamples="auto", silent=True))

def jaccard3(a, b): A, B = set(np.argsort(-a)[:3]), set(np.argsort(-b)[:3]); return len(A & B) / len(A | B)
def rv(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); Sxx, Syy, Sxy = X.T @ X, Y.T @ Y, X.T @ Y
    return float(np.trace(Sxy @ Sxy.T) / np.sqrt(np.trace(Sxx @ Sxx) * np.trace(Syy @ Syy)))
def marc(a, b): return float(np.mean(np.abs(rankdata(-a) - rankdata(-b))))

def setup(a):
    D = os.path.join(a.base, "data", "scarcity_levels"); s1_test = pd.read_csv(os.path.join(D, "S1_test.csv")); sc1 = joblib.load(os.path.join(D, "S1_scaler.pkl"))
    ix, _ = train_test_split(np.arange(len(s1_test)), train_size=min(a.n_eval, len(s1_test) - 1), stratify=s1_test["chd"], random_state=SEED)
    raw = s1_test.iloc[np.sort(ix)][FEATURES].reset_index(drop=True); raw[CONT] = sc1.inverse_transform(raw[CONT]); return D, s1_test, raw

def run_levels(D, raw, name, cap, a, want_perf=False):
    sv, perf = {}, None
    for lv in LEVELS:
        tr = pd.read_csv(os.path.join(D, f"{lv}_train.csv")); sc = joblib.load(os.path.join(D, f"{lv}_scaler.pkl"))
        Xb, yb = apply_smote(tr[FEATURES], tr["chd"], cap); m = make_model(name).fit(Xb, yb)
        Xe = raw.copy(); Xe[CONT] = sc.transform(raw[CONT]); sv[lv] = shap_matrix(m, name, tr[FEATURES], Xe, a.background)
        if want_perf and lv == "S1":
            te = pd.read_csv(os.path.join(D, "S1_test.csv")); p = m.predict_proba(te[FEATURES])[:, 1]; y = te["chd"].values
            tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel(); pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            perf = dict(AUC_S1=round(roc_auc_score(y, p), 4), recall_S1=round(rc, 4), precision_S1=round(pr, 4), F1_S1=round(2 * pr * rc / max(pr + rc, 1e-9), 4))
        print(f"    {name} {lv} done", flush=True)
    return sv, perf

def task_core(a):
    D, s1_test, raw = setup(a); rows, boot = [], []; rng = np.random.default_rng(SEED)
    for name in a.models:
        print(f"== {name}", flush=True); sv, _ = run_levels(D, raw, name, 0.30, a); imp = {lv: np.abs(sv[lv]).mean(0) for lv in LEVELS}
        keep = np.argsort(-imp["S1"])[:5]
        for lv in LEVELS[1:]:
            rows.append(dict(model=name, level=lv, spearman=spearmanr(imp["S1"], imp[lv])[0], kendall=kendalltau(imp["S1"], imp[lv])[0], jaccard_top3=jaccard3(imp["S1"], imp[lv]),
                             RV=rv(sv["S1"], sv[lv]), MARC=marc(imp["S1"], imp[lv]), spearman_top5_features=spearmanr(imp["S1"][keep], imp[lv][keep])[0]))
        n = len(raw)
        for B in (50, 100, 200, 300, 500):
            esis, widths = [], []
            for _ in range(a.reps):
                e = []
                for _b in range(B):
                    ix = rng.integers(0, n, n); b1 = np.abs(sv["S1"][ix]).mean(0)
                    e.append(np.mean([spearmanr(b1, np.abs(sv[lv][ix]).mean(0))[0] for lv in LEVELS[1:]]))
                esis.append(np.mean(e)); widths.append(np.percentile(e, 97.5) - np.percentile(e, 2.5))
            boot.append(dict(model=name, B=B, mean_ESI=np.mean(esis), SD_of_ESI_across_replicates=np.std(esis, ddof=1), mean_CI_width=np.mean(widths), SD_of_CI_width=np.std(widths, ddof=1)))
            print(f"    bootstrap B={B}: ESI {np.mean(esis):.4f}, CI width {np.mean(widths):.4f}", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "claims_core_levels.csv"), index=False)
    summ = df.groupby("model")[["spearman", "kendall", "jaccard_top3", "RV", "MARC", "spearman_top5_features"]].mean().round(4); summ.to_csv(os.path.join(a.outdir, "claims_core_summary.csv"))
    pd.DataFrame(boot).round(4).to_csv(os.path.join(a.outdir, "bootstrap_convergence.csv"), index=False)
    print("\n=== MEAN OVER S2..S7 (ESI under each metric) ===\n" + summ.to_string()); print("\nPer-level values: claims_core_levels.csv ; convergence: bootstrap_convergence.csv")
    piv = df.pivot(index="level", columns="model", values="spearman").round(3); print("\nSpearman by level:\n" + piv.to_string()); print("\nJaccard top-3 by level:\n" + df.pivot(index="level", columns="model", values="jaccard_top3").round(3).to_string())

def task_smote(a):
    D, s1_test, raw = setup(a); rows = []
    for name in a.models:
        for label, cap in (("none", None), ("20%", 0.2), ("30%", 0.3), ("40%", 0.4)):
            print(f"== {name} | SMOTE cap {label}", flush=True); sv, perf = run_levels(D, raw, name, cap, a, want_perf=True); imp = {lv: np.abs(sv[lv]).mean(0) for lv in LEVELS}
            esi = float(np.mean([spearmanr(imp["S1"], imp[lv])[0] for lv in LEVELS[1:]])); rows.append(dict(model=name, smote_cap=label, ESI=round(esi, 4), **perf))
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "smote_sensitivity.csv"), index=False); print("\n=== SMOTE SENSITIVITY ===\n" + df.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=["core", "smote"], required=True); ap.add_argument("--base", default="."); ap.add_argument("--models", nargs="+", default=["XGBoost", "RandomForest", "LogisticRegression", "MLP"], choices=["XGBoost", "RandomForest", "LogisticRegression", "MLP"])
    ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--background", type=int, default=100); ap.add_argument("--reps", type=int, default=20, help="bootstrap replicates per B"); ap.add_argument("--outdir", default="outputs/claims")
    a = ap.parse_args(); (task_core if a.task == "core" else task_smote)(a)
