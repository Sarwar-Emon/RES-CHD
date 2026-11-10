"""
defense_experiments.py  -  extra experiments for the meeting with Dr. Chen.  NEW FILE: put it in the Project folder next to
claims_experiments.py and imbalance_experiments.py (version 2), and run from there.

  python defense_experiments.py --task matrices      (about 30-90 min)
      -> FULL confusion matrices (TN, FP, FN, TP) for every model, BEFORE and AFTER SMOTE and for the other imbalance methods
         (no resampling, SMOTE 30%, random oversampling, class weights, random undersampling, Borderline-SMOTE, SMOTE-Tomek, balanced random forest),
         at the 0.5 cutoff and at the best-F1 cutoff (chosen from 5-fold out-of-fold training predictions), plus a before/after SMOTE picture.
  python defense_experiments.py --task tune          (about 1-3 hours)
      -> light hyperparameter search for every model (SMOTE inside the cross-validation, scored by PR-AUC), then test-set results.

Outputs go to outputs/defense/ .
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_experiments as ce
import imbalance_experiments as ie
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FEATURES = ce.FEATURES; SEED = 42
warnings.filterwarnings("ignore")
EXTRA = {"rus": "random undersampling (1:3)", "borderline": "Borderline-SMOTE 30%", "smotetomek": "SMOTE-Tomek 30%", "brf": "balanced random forest"}
ALL = list(ie.METHODS) + list(EXTRA)
LABEL = dict(ie.LABEL, **EXTRA)

def fit_any(name, method, X, y):
    if method in ie.METHODS: return ie.fit(name, method, X, y)
    m = ce.make_model(name)
    if method == "rus": X, y = RandomUnderSampler(sampling_strategy=1 / 3, random_state=SEED).fit_resample(X, y)
    elif method == "borderline": X, y = BorderlineSMOTE(sampling_strategy=0.30, random_state=SEED).fit_resample(X, y)
    elif method == "smotetomek": X, y = SMOTETomek(smote=SMOTE(sampling_strategy=0.30, random_state=SEED), random_state=SEED).fit_resample(X, y)
    elif method == "brf":
        if name != "RandomForest": return None
        m = BalancedRandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    return m.fit(X, y)

def oof_any(name, method, X, y, folds):
    p = np.zeros(len(y))
    for tr_i, va_i in StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, y):
        m = fit_any(name, method, X.iloc[tr_i], y[tr_i])
        if m is None: return None
        p[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
    return p

def task_matrices(a):
    X, ytr, Xte, y = ie.load(a); rows = []
    for name in a.models:
        for meth in ALL:
            m = fit_any(name, meth, X, ytr)
            if m is None: continue
            pt = m.predict_proba(Xte)[:, 1]; po = oof_any(name, meth, X, ytr, a.folds); tb = ie.thr_f1(ytr, po)
            for rule, t in (("cutoff 0.5", 0.5), ("best-F1 cutoff (out-of-fold)", tb)):
                rows.append(dict(model=name, imbalance_method=LABEL[meth], rule=rule, AUC=round(roc_auc_score(y, pt), 4), PR_AUC=round(average_precision_score(y, pt), 4), **ie.at(y, pt, t)))
            print(f"   {name} | {LABEL[meth]} done", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "confusion_matrices_by_method.csv"), index=False); pd.set_option("display.width", 260)
    h = df[df.rule == "cutoff 0.5"]; b = h[h.imbalance_method == LABEL["none"]].set_index("model"); s = h[h.imbalance_method == LABEL["smote30"]].set_index("model")
    cols = ["TN", "FP", "FN", "TP", "recall", "precision"]; ba = pd.concat({"BEFORE SMOTE (no resampling)": b[cols], "AFTER SMOTE 30% (thesis)": s[cols]}, axis=1)
    print("\n=== BEFORE vs AFTER SMOTE, cutoff 0.5 (test set: 259 CHD, 6,165 without) ===\n" + ba.to_string())
    print("\n=== ALL MATRICES ===\n" + df.drop(columns=["threshold"]).to_string(index=False))
    fig, ax = plt.subplots(2, len(a.models), figsize=(3.0 * len(a.models), 5.6), squeeze=False)
    for j, name in enumerate(a.models):
        for i, (lab, tab) in enumerate((("Before SMOTE", b), ("After SMOTE", s))):
            r = tab.loc[name]; M = np.array([[r.TN, r.FP], [r.FN, r.TP]]); ax[i, j].imshow(np.log10(M + 1), cmap="Blues", vmin=0, vmax=4)
            for u in range(2):
                for v in range(2): ax[i, j].text(v, u, f"{int(M[u, v]):,}", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if np.log10(M[u, v] + 1) > 2.3 else "black")
            ax[i, j].set_xticks([0, 1]); ax[i, j].set_yticks([0, 1]); ax[i, j].set_xticklabels(["Pred. no", "Pred. CHD"], fontsize=8); ax[i, j].set_yticklabels(["No CHD", "CHD"], fontsize=8)
            ax[i, j].set_title(f"{name}\n{lab} (recall {r.recall:.2f}, precision {r.precision:.2f})", fontsize=8.5)
    fig.suptitle("Confusion matrices at cutoff 0.5, before and after SMOTE (S1 test set)", fontweight="bold"); plt.tight_layout(); fig.savefig(os.path.join(a.outdir, "before_after_smote_confusion.png"), dpi=200, bbox_inches="tight")

GRIDS = {"LogisticRegression": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
         "RandomForest": {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [None, 6, 10, 16], "clf__min_samples_leaf": [1, 5, 20, 50], "clf__max_features": ["sqrt", 0.5]},
         "XGBoost": {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [3, 4, 6, 8], "clf__learning_rate": [0.01, 0.03, 0.05, 0.1], "clf__subsample": [0.6, 0.8, 1.0], "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 5, 10]},
         "MLP": {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2], "clf__learning_rate_init": [1e-3, 3e-3, 1e-2], "clf__activation": ["relu", "tanh"]}}

def task_tune(a):
    X, ytr, Xte, y = ie.load(a); rows = []
    for name in a.models:
        print(f"== tuning {name}", flush=True); pipe = Pipeline([("smote", SMOTE(sampling_strategy=0.30, random_state=SEED)), ("clf", ce.make_model(name))])
        n_iter = min(a.iters, int(np.prod([len(v) for v in GRIDS[name].values()])))
        rs = RandomizedSearchCV(pipe, GRIDS[name], n_iter=n_iter, scoring="average_precision", cv=StratifiedKFold(a.cv, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=1, refit=True).fit(X, ytr)
        best = rs.best_estimator_; pt = best.predict_proba(Xte)[:, 1]; po = cross_val_predict(best, X, ytr, cv=StratifiedKFold(a.cv, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        h, b = ie.at(y, pt, 0.5), ie.at(y, pt, ie.thr_f1(ytr, po))
        rows.append(dict(model=name, best_params=str({k.replace("clf__", ""): v for k, v in rs.best_params_.items()}), CV_PR_AUC=round(rs.best_score_, 4), test_AUC=round(roc_auc_score(y, pt), 4), test_PR_AUC=round(average_precision_score(y, pt), 4),
                         recall_at_0_5=h["recall"], precision_at_0_5=h["precision"], F1_at_0_5=h["F1"], best_F1_cutoff=b["threshold"], recall_best=b["recall"], precision_best=b["precision"], F1_best=b["F1"]))
        print(f"   {name}: test AUC {rows[-1]['test_AUC']}, PR-AUC {rows[-1]['test_PR_AUC']}, best F1 {rows[-1]['F1_best']}", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "tuning_results.csv"), index=False); pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 90)
    print("\n=== AFTER HYPERPARAMETER TUNING (compare with the untuned thesis models) ===\n" + df.drop(columns=["best_params"]).to_string(index=False)); print("\nBest settings:\n" + df[["model", "best_params"]].to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=["matrices", "tune"], required=True); ap.add_argument("--base", default=".")
    ap.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForest", "XGBoost", "MLP"], choices=["XGBoost", "RandomForest", "LogisticRegression", "MLP"])
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--iters", type=int, default=10); ap.add_argument("--cv", type=int, default=3); ap.add_argument("--outdir", default="outputs/defense")
    a = ap.parse_args(); {"matrices": task_matrices, "tune": task_tune}[a.task](a)
