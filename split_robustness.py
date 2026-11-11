"""
Split/training robustness (repeated stratified hold-out) for RES-CHD.
Your pipeline used ONE random 80/20 split per scarcity level. This repeats the WHOLE procedure with several random seeds
(new split at every level, new scaler, new SMOTE, retrained models, new SHAP) and reports mean +/- SD of AUC and ESI.
It answers: "are the results an accident of one particular split / one particular training run?"

It re-splits the same winsorized cohorts your pipeline already built (train+test of each S{k} are pooled and re-split).
Run from the Project folder:
    python split_robustness.py --seeds 3 --models XGBoost RandomForest LogisticRegression     # fast, skips the MLP
    python split_robustness.py --seeds 5                                                       # all four models (about an hour)
Tree/linear models are explained on the full test set (exact). The MLP is explained on --n-eval-mlp random test patients with exact KernelSHAP.
"""
import argparse, os, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
CONT = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi"]
ALL_MODELS = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
LEVELS = [f"S{i}" for i in range(1, 8)]
MLP_HIDDEN, MLP_ACT = (64, 32), "relu"

def make_model(name, seed):      # same settings as MODEL_CONFIG in phase2_training.py
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=seed, n_jobs=-1)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    return MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation=MLP_ACT, max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=seed)

def apply_smote(X, y, seed):     # same logic as phase2_training.apply_smote
    n_min, n_maj = int(y.sum()), int((y == 0).sum()); k = min(5, n_min - 1)
    if k < 1: return X, y
    return SMOTE(sampling_strategy=max(int(n_maj * 0.30), n_min) / n_maj, k_neighbors=k, random_state=seed).fit_resample(X, y)

def shap_2d(sv):
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv); return sv[:, :, 1] if sv.ndim == 3 else sv

def importance(model, name, Xtrain, Xe, n_bg, seed):
    if name in ("XGBoost", "RandomForest"):
        sv = shap_2d(shap.TreeExplainer(model).shap_values(Xe))
    elif name == "LogisticRegression":
        sv = shap_2d(shap.LinearExplainer(model, Xtrain.mean(axis=0).values.reshape(1, -1)).shap_values(Xe))
    else:
        rng = np.random.default_rng(seed); bg = Xtrain.iloc[rng.choice(len(Xtrain), size=min(n_bg, len(Xtrain)), replace=False)]
        f = lambda z: model.predict_proba(pd.DataFrame(z, columns=FEATURES))[:, 1]
        np.random.seed(seed); sv = shap_2d(shap.KernelExplainer(f, bg).shap_values(Xe, nsamples="auto", silent=True))
    return np.abs(sv).mean(axis=0)

def main(a):
    D = os.path.join(a.base, "data", "scarcity_levels"); pools = {}
    for lv in LEVELS:
        tr, te = pd.read_csv(os.path.join(D, f"{lv}_train.csv")), pd.read_csv(os.path.join(D, f"{lv}_test.csv")); sc = joblib.load(os.path.join(D, f"{lv}_scaler.pkl"))
        pool = pd.concat([tr, te], ignore_index=True); pool[CONT] = sc.inverse_transform(pool[CONT]); pools[lv] = pool
    rows = []
    for s in range(a.seeds):
        seed = 100 + s
        for name in a.models:
            imps = {}
            for lv in LEVELS:
                trn, tst = train_test_split(pools[lv], test_size=0.2, stratify=pools[lv]["chd"], random_state=seed)
                sc = StandardScaler().fit(trn[CONT]); Xtr, Xte = trn[FEATURES].copy(), tst[FEATURES].copy()
                Xtr[CONT] = sc.transform(trn[CONT]); Xte[CONT] = sc.transform(tst[CONT])
                Xb, yb = apply_smote(Xtr, trn["chd"], seed); m = make_model(name, seed).fit(Xb, yb)
                auc = roc_auc_score(tst["chd"], m.predict_proba(Xte)[:, 1])
                n_e = a.n_eval_mlp if name == "MLP" else a.n_eval
                Xe = Xte.sample(n=n_e, random_state=seed) if 0 < n_e < len(Xte) else Xte
                imps[lv] = importance(m, name, Xtr, Xe, a.background, seed)
                print(f"    seed {seed} | {name} | {lv} done (explained {len(Xe)} patients)", flush=True)
                rows.append(dict(seed=seed, model=name, level=lv, auc=auc, n_train=len(Xtr), n_test=len(Xte), rho=(1.0 if lv == "S1" else float(spearmanr(imps["S1"], imps[lv])[0]))))
            esi = np.mean([r["rho"] for r in rows if r["seed"] == seed and r["model"] == name and r["level"] != "S1"])
            print(f"seed {seed} | {name:<19} | AUC(S1) = {[r['auc'] for r in rows if r['seed']==seed and r['model']==name and r['level']=='S1'][0]:.4f} | ESI = {esi:.4f}", flush=True)
    df = pd.DataFrame(rows); os.makedirs(a.outdir, exist_ok=True); df.to_csv(os.path.join(a.outdir, "split_robustness_raw.csv"), index=False)
    out = []
    for name in a.models:
        d = df[df.model == name]; esis = d[d.level != "S1"].groupby("seed").rho.mean(); aucs = d[d.level == "S1"].set_index("seed").auc
        sd = lambda v: round(float(v.std(ddof=1)), 4) if len(v) > 1 else float("nan")
        out.append(dict(model=name, AUC_S1_mean=round(aucs.mean(), 4), AUC_S1_SD=sd(aucs), ESI_mean=round(esis.mean(), 4), ESI_SD=sd(esis), ESI_min=round(esis.min(), 4), ESI_max=round(esis.max(), 4)))
    summ = pd.DataFrame(out); summ.to_csv(os.path.join(a.outdir, "split_robustness_summary.csv"), index=False)
    print(f"\n=== SUMMARY over {a.seeds} independent splits/trainings ===\n" + summ.to_string(index=False))
    if len(a.models) > 1:
        piv = df[df.level != "S1"].groupby(["seed", "model"]).rho.mean().unstack()
        if "MLP" in piv: print(f"\nSeeds where the MLP has the lowest ESI: {int((piv.idxmin(axis=1) == 'MLP').sum())}/{len(piv)}")
        if {"RandomForest", "XGBoost"} <= set(piv.columns): print(f"Seeds where RandomForest ESI > XGBoost ESI: {int((piv['RandomForest'] > piv['XGBoost']).sum())}/{len(piv)}")
        if {"LogisticRegression", "XGBoost"} <= set(piv.columns): print(f"Seeds where LogisticRegression ESI > XGBoost ESI: {int((piv['LogisticRegression'] > piv['XGBoost']).sum())}/{len(piv)}")
    print("Saved to", a.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS); ap.add_argument("--n-eval-mlp", type=int, default=300); ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--background", type=int, default=100); ap.add_argument("--outdir", default="outputs/split_robustness")
    ap.add_argument("--mlp-activation", default="relu"); ap.add_argument("--mlp-hidden", default="64,32")
    args = ap.parse_args(); MLP_ACT = args.mlp_activation; MLP_HIDDEN = tuple(int(x) for x in args.mlp_hidden.split(","))
    main(args)
