"""
Noise-floor control for ESI: how much do SHAP importance rankings change when the SAME data is simply re-trained with a different random seed?
ESI mixes two things: (a) the effect of having less/older training data, and (b) plain retraining randomness. This script measures (b) alone.
For each model: train R times on the full S1 training set (same data, different seed for the model AND for SMOTE), explain the SAME 300 fixed test
patients, and compute Spearman rho between the seed-0 ranking and each other seed's ranking. Compare with ESI (S1 vs S2..S7):
  noise-floor rho close to ESI  ->  most of the 'instability' is retraining randomness, not data scarcity
  noise-floor rho near 1.0      ->  the instability is caused by the change in training data
Logistic Regression itself has no randomness, but SMOTE creates different synthetic patients for each seed, so its noise floor can still be below 1.0 (it measures the effect of the SMOTE draw).

Run from the Project folder:
    python noise_floor.py                                   # all four models, 6 retrainings each
    python noise_floor.py --models XGBoost MLP --reps 8
    python noise_floor.py --models MLP --mlp-activation logistic --mlp-hidden 64,32 --outdir outputs/noise_mlp_logistic
"""
import argparse, os, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]
ALL_MODELS = ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]
MLP_HIDDEN, MLP_ACT = (64, 32), "relu"

def make_model(name, seed):      # same settings as phase2_training.py (only the seed varies)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", random_state=seed, n_jobs=-1)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    return MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation=MLP_ACT, max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=seed)

def apply_smote(X, y, seed):
    n_min, n_maj = int(y.sum()), int((y == 0).sum()); k = min(5, n_min - 1)
    if k < 1: return X, y
    return SMOTE(sampling_strategy=max(int(n_maj * 0.30), n_min) / n_maj, k_neighbors=k, random_state=seed).fit_resample(X, y)

def shap_2d(sv):
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv); return sv[:, :, 1] if sv.ndim == 3 else sv

def importance(model, name, Xtrain, Xe, n_bg):
    if name in ("XGBoost", "RandomForest"):
        sv = shap_2d(shap.TreeExplainer(model).shap_values(Xe))
    elif name == "LogisticRegression":
        sv = shap_2d(shap.LinearExplainer(model, Xtrain.mean(axis=0).values.reshape(1, -1)).shap_values(Xe))
    else:
        rng = np.random.default_rng(42); bg = Xtrain.iloc[rng.choice(len(Xtrain), size=min(n_bg, len(Xtrain)), replace=False)]
        f = lambda z: model.predict_proba(pd.DataFrame(z, columns=FEATURES))[:, 1]
        np.random.seed(42); sv = shap_2d(shap.KernelExplainer(f, bg).shap_values(Xe, nsamples="auto", silent=True))
    return np.abs(sv).mean(axis=0)

def main(a):
    D = os.path.join(a.base, "data", "scarcity_levels")
    tr, te = pd.read_csv(os.path.join(D, "S1_train.csv")), pd.read_csv(os.path.join(D, "S1_test.csv"))
    ix, _ = train_test_split(np.arange(len(te)), train_size=min(a.n_eval, len(te) - 1), stratify=te["chd"], random_state=42)
    Xe = te.iloc[np.sort(ix)][FEATURES].reset_index(drop=True); Xtr = tr[FEATURES]; rows = []
    print(f"Same data every time: S1 train ({len(tr):,} patients); {len(Xe)} fixed explained patients; {a.reps} retrainings per model")
    for name in a.models:
        imps = []
        for r in range(a.reps):
            Xb, yb = apply_smote(Xtr, tr["chd"], seed=r); m = make_model(name, r).fit(Xb, yb)
            imps.append(importance(m, name, Xtr, Xe, a.background)); print(f"  {name:<19} retraining {r + 1}/{a.reps} done", flush=True)
        rhos = [float(spearmanr(imps[0], imps[r])[0]) for r in range(1, a.reps)]
        allp = [float(spearmanr(imps[i], imps[j])[0]) for i in range(a.reps) for j in range(i + 1, a.reps)]
        rows.append(dict(model=name, noise_floor_rho_vs_seed0=round(float(np.mean(rhos)), 4), SD=round(float(np.std(rhos, ddof=1)), 4), min=round(min(rhos), 4), all_pairs_mean=round(float(np.mean(allp)), 4)))
    out = pd.DataFrame(rows); os.makedirs(a.outdir, exist_ok=True); out.to_csv(os.path.join(a.outdir, "noise_floor.csv"), index=False)
    print("\n=== NOISE FLOOR: same S1 data, retrained with different seeds ===\n" + out.to_string(index=False))
    print("\nCompare with ESI over scarcity levels (S1 vs S2..S7): LR ~0.91, RF ~0.92, XGBoost ~0.86, MLP ~0.72 (relu 64,32).\nIf a model's noise floor is about as low as its ESI, its 'instability' is mostly retraining randomness rather than data scarcity.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--reps", type=int, default=6); ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--mlp-activation", default="relu"); ap.add_argument("--mlp-hidden", default="64,32"); ap.add_argument("--outdir", default="outputs/noise_floor")
    args = ap.parse_args(); MLP_ACT = args.mlp_activation; MLP_HIDDEN = tuple(int(x) for x in args.mlp_hidden.split(","))
    main(args)
