import argparse, os, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
CONT = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
ALL_MODELS = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
SEED = 42

def make_model(name):
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=SEED, n_jobs=-1)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=SEED)

def apply_smote(X, y):
    n_min, n_maj = int(y.sum()), int((y == 0).sum()); k = min(5, n_min - 1)
    if k < 1: return X, y
    return SMOTE(sampling_strategy=max(int(n_maj * 0.30), n_min) / n_maj, k_neighbors=k, random_state=SEED).fit_resample(X, y)

def ece(y, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def main(a):
    D = os.path.join(a.base, "data", "scarcity_levels")
    tr, te = pd.read_csv(os.path.join(D, f"{a.level}_train.csv")), pd.read_csv(os.path.join(D, f"{a.level}_test.csv"))
    sc = joblib.load(os.path.join(D, f"{a.level}_scaler.pkl")); pool = pd.concat([tr, te], ignore_index=True); pool[CONT] = sc.inverse_transform(pool[CONT])
    y = pool["chd"].values; print(f"{a.level}: {len(pool):,} patients, {int(y.sum())} CHD cases ({y.mean():.2%}), {a.folds}-fold stratified CV")
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=SEED); rows = []
    for name in a.models:
        for f, (i_tr, i_te) in enumerate(skf.split(pool, y), 1):
            trn, tst = pool.iloc[i_tr], pool.iloc[i_te]
            s = StandardScaler().fit(trn[CONT]); Xtr, Xte = trn[FEATURES].copy(), tst[FEATURES].copy(); Xtr[CONT] = s.transform(trn[CONT]); Xte[CONT] = s.transform(tst[CONT])
            Xb, yb = apply_smote(Xtr, trn["chd"]); m = make_model(name).fit(Xb, yb); p = m.predict_proba(Xte)[:, 1]; yt = tst["chd"].values
            tn, fp, fn, tp = confusion_matrix(yt, (p >= 0.5).astype(int), labels=[0, 1]).ravel(); prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append(dict(model=name, fold=f, auc=roc_auc_score(yt, p), pr_auc=average_precision_score(yt, p), f1=2 * prec * rec / max(prec + rec, 1e-9),
                             precision=prec, recall=rec, specificity=tn / (tn + fp), ece=ece(yt, p)))
            print(f"  {name:<19} fold {f}: AUC {rows[-1]['auc']:.4f} | PR-AUC {rows[-1]['pr_auc']:.4f}", flush=True)
    df = pd.DataFrame(rows); os.makedirs(a.outdir, exist_ok=True); df.to_csv(os.path.join(a.outdir, f"cv_{a.level}_folds.csv"), index=False)
    g = df.groupby("model"); mean, sd = g.mean(numeric_only=True).drop(columns="fold"), g.std(numeric_only=True).drop(columns="fold")
    summ = mean.round(4).astype(str) + " +/- " + sd.round(4).astype(str); summ = summ.loc[[m for m in a.models]]
    summ.to_csv(os.path.join(a.outdir, f"cv_{a.level}_summary.csv"))
    print(f"\n=== {a.folds}-FOLD CV SUMMARY ({a.level}, mean +/- SD across folds) ===\n" + summ.to_string()); print("Saved to", a.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--level", default="S1"); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS); ap.add_argument("--outdir", default="outputs/cv")
    main(ap.parse_args())
