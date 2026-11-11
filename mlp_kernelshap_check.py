"""
Is MLP's low ESI real, or caused by how SHAP was approximated?
Re-uses your SAVED MLP models (models/S{1..7}_MLP.pkl) and your per-level data
(data/scarcity_levels/S{k}_train.csv / _test.csv). No re-training.

phase3_shap.py explained the MLP with KernelExplainer(nsamples=100) on 200 patients. For 8 features
the library default is ~2,064 coalition samples, so the original SHAP values are noisy. This script recomputes
MLP ESI under four settings:
  A  thesis setting (nsamples=100, 200 patients)           -> should reproduce ESI ~0.6576  (sanity check)
  B  same patients, ACCURATE SHAP (nsamples='auto')         -> effect of approximation accuracy alone
  C  accurate SHAP, more patients (--n-eval, default 500)   -> effect of sample size too
  D  setting A with a different random draw (seed 43)       -> how much ESI moves from sampling noise alone
Run from the Project folder (the one with data/ and models/):
    python3 mlp_kernelshap_check.py
    python3 mlp_kernelshap_check.py --skip C      # faster
    python3 mlp_kernelshap_check.py --skip A B C D --draws 10   # only the multi-draw check (mean +/- SD)
"""
import argparse, os, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

FEATURES = ["age", "sbp", "dbp", "hdl", "total_chol", "bmi", "sex", "smoking"]   # same order as phase3_shap.py
LEVELS = [f"S{i}" for i in range(1, 8)]

def mlp_importance(base, level, nsamples, n_eval, seed, n_bg=100):
    tr = pd.read_csv(os.path.join(base, "data", "scarcity_levels", f"{level}_train.csv"))[FEATURES]
    te = pd.read_csv(os.path.join(base, "data", "scarcity_levels", f"{level}_test.csv"))[FEATURES]
    model = joblib.load(os.path.join(base, "models", f"{level}_MLP.pkl"))
    np.random.seed(seed)                                                       # identical logic to phase3_shap.py
    bg = tr.iloc[np.random.choice(len(tr), size=min(n_bg, len(tr)), replace=False)]
    ev = te.iloc[np.random.choice(len(te), size=min(n_eval, len(te)), replace=False)].reset_index(drop=True)
    ex = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], bg)
    sv = np.asarray(ex.shap_values(ev, nsamples=nsamples, silent=True))
    if sv.ndim == 3: sv = sv[:, :, -1]
    return np.abs(sv).mean(axis=0)

def esi_for(base, nsamples, n_eval, seed, verbose=True):
    ranks = {}
    for lv in LEVELS:
        imp = pd.Series(mlp_importance(base, lv, nsamples, n_eval, seed), index=FEATURES)
        ranks[lv] = imp.rank(ascending=False, method="min")            # same ranking rule as phase3
        if verbose: print(f"    {lv} done", flush=True)
    rhos = [spearmanr(ranks["S1"], ranks[lv])[0] for lv in LEVELS[1:]]
    return float(np.mean(rhos)), [round(float(r), 3) for r in rhos]

def main(a):
    settings = {"A": ("A  thesis setting      (nsamples=100,  200 patients, seed 42)", 100, 200, 42),
                "B": ("B  accurate SHAP       (nsamples=auto, 200 patients, seed 42)", "auto", 200, 42),
                "C": (f"C  accurate + larger   (nsamples=auto, {a.n_eval} patients, seed 42)", "auto", a.n_eval, 42),
                "D": ("D  setting A, other draw (nsamples=100,  200 patients, seed 43)", 100, 200, 43)}
    rows = []
    for key, (label, ns, ne, seed) in settings.items():
        if key in a.skip: continue
        print(f"\n[{label}]", flush=True)
        esi, rhos = esi_for(a.base, ns, ne, seed)
        rows.append(dict(setting=label, ESI=round(esi, 4), rho_S2_to_S7=rhos))
        print(f"  -> MLP ESI = {esi:.4f}   per-level rho {rhos}", flush=True)
    if a.draws:
        print(f"\n[E  {a.draws} random draws, exact SHAP (nsamples=auto), {a.n_eval} patients each]", flush=True)
        vals = []
        for d in range(a.draws):
            e, r = esi_for(a.base, "auto", a.n_eval, 42 + d, verbose=False)
            vals.append(e); print(f"  draw {d + 1:>2} (seed {42 + d}): ESI = {e:.4f}", flush=True)
        v = np.array(vals)
        print(f"\n  MLP ESI over {a.draws} draws: mean {v.mean():.4f} | SD {v.std(ddof=1):.4f} | min {v.min():.4f} | max {v.max():.4f}"
              f" | draws below 0.70 (LOW): {(v < 0.70).sum()}/{a.draws}")
        rows.append(dict(setting=f"E  {a.draws} draws (exact SHAP, {a.n_eval} patients)", ESI=round(float(v.mean()), 4),
                         rho_S2_to_S7=f"SD {v.std(ddof=1):.3f}, range {v.min():.3f}-{v.max():.3f}"))
    out = pd.DataFrame(rows); os.makedirs(a.outdir, exist_ok=True)
    out.to_csv(os.path.join(a.outdir, "mlp_kernelshap_check.csv"), index=False)
    print("\n=== SUMMARY ===\n" + out.to_string(index=False))
    print("\nThesis MLP ESI = 0.6576. Setting A should be close to that.\n"
          "If B/C stay near 0.66 -> the low MLP ESI is NOT an artifact of the SHAP approximation.\n"
          "If B/C are much higher -> part of the instability was SHAP noise, and the thesis wording must be revised.\n"
          "If D differs a lot from A -> the original estimate is sensitive to sampling noise.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=".")
    ap.add_argument("--n-eval", type=int, default=500); ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--draws", type=int, default=0, help="extra: repeat with N different random draws and report mean/SD")
    ap.add_argument("--outdir", default="outputs/mlp_check"); main(ap.parse_args())
