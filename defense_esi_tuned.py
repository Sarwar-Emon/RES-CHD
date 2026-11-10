"""
defense_esi_tuned.py  -  NEW FILE.  Put it in the Project folder next to claims_experiments.py, imbalance_experiments.py and defense_experiments.py.

Question it answers: is the MLP's low explanation stability (ESI) just because it was trained with default settings?
It computes ESI (300 fixed patients, mean over S2-S7, 3 seeds) for:
   tuned_smote30  - the best settings found by the tuning run (Logistic Regression, Random Forest, XGBoost, MLP), with SMOTE 30%
   default_rus    - the thesis' default settings, with random undersampling (1:3)   (this is the method that gave the MLP an AUC of 0.851)
   tuned_rus      - tuned settings with random undersampling
and prints the test-set confusion matrix at cutoff 0.5 for each tuned configuration.

  python defense_esi_tuned.py                                    (slow: run it overnight)
  python defense_esi_tuned.py --models XGBoost LogisticRegression MLP     (quicker, leaves out Random Forest)

Outputs go to outputs/defense/ .
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_experiments as ce
import imbalance_experiments as ie
from scipy.stats import spearmanr
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
TUNED = {"LogisticRegression": {"C": 0.01},
         "RandomForest": {"n_estimators": 200, "min_samples_leaf": 50, "max_features": "sqrt", "max_depth": 6},
         "XGBoost": {"subsample": 0.8, "n_estimators": 400, "min_child_weight": 5, "max_depth": 3, "learning_rate": 0.03, "colsample_bytree": 0.6},
         "MLP": {"learning_rate_init": 0.001, "hidden_layer_sizes": (32,), "alpha": 1e-5, "activation": "relu"}}
CONFIGS = {"default_smote30": (False, "smote30"), "tuned_smote30": (True, "smote30"), "default_rus": (False, "rus"), "tuned_rus": (True, "rus")}
FEATURES, CONT, LEVELS = ce.FEATURES, ce.CONT, ce.LEVELS

def fit_cfg(name, config, X, y):
    tuned, method = CONFIGS[config]; m = ce.make_model(name)
    if tuned: m.set_params(**TUNED[name])
    if method == "smote30": X, y = ce.apply_smote(X, y, 0.30)
    else: X, y = RandomUnderSampler(sampling_strategy=1 / 3, random_state=ce.SEED).fit_resample(X, y)
    return m.fit(X, y)

def main(a):
    D, s1_test, raw = ce.setup(a); rows, perf = [], []          # the 300 patients are chosen once (seed 42) and stay fixed
    tr1 = pd.read_csv(os.path.join(D, "S1_train.csv")); te = pd.read_csv(os.path.join(D, "S1_test.csv")); y = te["chd"].values
    for name in a.models:
        for cfg in a.configs:
            ce.SEED = 42; m = fit_cfg(name, cfg, tr1[FEATURES], tr1["chd"].values); pt = m.predict_proba(te[FEATURES])[:, 1]
            perf.append(dict(model=name, config=cfg, AUC=round(roc_auc_score(y, pt), 4), PR_AUC=round(average_precision_score(y, pt), 4), **ie.at(y, pt, 0.5)))
            for seed in a.seeds:
                ce.SEED = seed; sv = {}
                for lv in LEVELS:
                    tr = pd.read_csv(os.path.join(D, f"{lv}_train.csv")); sc = joblib.load(os.path.join(D, f"{lv}_scaler.pkl")); m = fit_cfg(name, cfg, tr[FEATURES], tr["chd"].values)
                    Xe = raw.copy(); Xe[CONT] = sc.transform(raw[CONT]); sv[lv] = ce.shap_matrix(m, name, tr[FEATURES], Xe, a.background)
                imp = {lv: np.abs(sv[lv]).mean(0) for lv in LEVELS}; esi = float(np.mean([spearmanr(imp["S1"], imp[lv])[0] for lv in LEVELS[1:]]))
                rows.append(dict(seed=seed, model=name, config=cfg, ESI=round(esi, 4))); print(f"   seed {seed} | {name} | {cfg}: ESI {esi:.4f}", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "esi_tuned.csv"), index=False); pf = pd.DataFrame(perf); pf.to_csv(os.path.join(a.outdir, "tuned_config_performance.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n=== ESI (mean over seeds) ===\n" + df.groupby(["model", "config"])["ESI"].mean().unstack().round(4).to_string()); print("\n=== SD over seeds ===\n" + df.groupby(["model", "config"])["ESI"].std(ddof=1).unstack().round(4).to_string())
    print("\n=== TEST SET AT CUTOFF 0.5 FOR EACH CONFIGURATION (259 CHD, 6,165 without) ===\n" + pf.drop(columns=["threshold", "specificity"]).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForest", "XGBoost", "MLP"], choices=["XGBoost", "RandomForest", "LogisticRegression", "MLP"])
    ap.add_argument("--configs", nargs="+", default=["tuned_smote30", "default_rus"], choices=list(CONFIGS)); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--background", type=int, default=100); ap.add_argument("--outdir", default="outputs/defense")
    main(ap.parse_args())
