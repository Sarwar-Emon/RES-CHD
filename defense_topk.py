import argparse, os, sys, warnings
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_experiments as ce
import defense_esi_tuned as dt
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
FEATURES, CONT, LEVELS = ce.FEATURES, ce.CONT, ce.LEVELS

def main(a):
    D, s1_test, raw = ce.setup(a); rows, orders = [], []
    for name in a.models:
        for cfg in a.configs:
            for seed in a.seeds:
                ce.SEED = seed; sv = {}
                for lv in LEVELS:
                    tr = pd.read_csv(os.path.join(D, f"{lv}_train.csv")); sc = joblib.load(os.path.join(D, f"{lv}_scaler.pkl")); m = dt.fit_cfg(name, cfg, tr[FEATURES], tr["chd"].values)
                    Xe = raw.copy(); Xe[CONT] = sc.transform(raw[CONT]); sv[lv] = ce.shap_matrix(m, name, tr[FEATURES], Xe, a.background)
                imp = {lv: np.abs(sv[lv]).mean(0) for lv in LEVELS}; order = np.argsort(-imp["S1"]); top5 = order[:5]; top3 = set(order[:3]); lv2 = LEVELS[1:]
                rows.append(dict(seed=seed, model=name, config=cfg, ESI_all8=np.mean([spearmanr(imp["S1"], imp[l])[0] for l in lv2]), ESI_top5=np.mean([spearmanr(imp["S1"][top5], imp[l][top5])[0] for l in lv2]),
                                 top3_same_set=np.mean([set(np.argsort(-imp[l])[:3]) == top3 for l in lv2]), top1_same=np.mean([int(np.argmax(imp[l])) == int(order[0]) for l in lv2])))
                if seed == a.seeds[0]: orders.append(dict(model=name, config=cfg, S1_order_most_to_least_important=" > ".join(FEATURES[i] for i in order)))
                print(f"   seed {seed} | {name} | {cfg}: ESI(all 8) {rows[-1]['ESI_all8']:.3f} | ESI(top 5) {rows[-1]['ESI_top5']:.3f} | top-3 set same {rows[-1]['top3_same_set']:.2f} | top-1 same {rows[-1]['top1_same']:.2f}", flush=True)
    os.makedirs(a.outdir, exist_ok=True); df = pd.DataFrame(rows).round(4); df.to_csv(os.path.join(a.outdir, "esi_topk.csv"), index=False); pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 120)
    print("\n=== MEAN OVER SEEDS ===\n" + df.groupby(["model", "config"])[["ESI_all8", "ESI_top5", "top3_same_set", "top1_same"]].mean().round(3).to_string())
    print("\n=== SD OVER SEEDS ===\n" + df.groupby(["model", "config"])[["ESI_all8", "ESI_top5"]].std(ddof=1).round(3).to_string())
    print("\n=== FEATURE ORDER AT S1 (first seed) ===\n" + pd.DataFrame(orders).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="."); ap.add_argument("--models", nargs="+", default=["LogisticRegression", "RandomForest", "XGBoost", "MLP"], choices=["XGBoost", "RandomForest", "LogisticRegression", "MLP"])
    ap.add_argument("--configs", nargs="+", default=["default_smote30", "tuned_smote30"], choices=list(dt.CONFIGS)); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--background", type=int, default=100); ap.add_argument("--outdir", default="outputs/defense")
    main(ap.parse_args())
