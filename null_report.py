"""
Missing-value report: which columns were missing, how many rows were dropped, and whether
the dropped rows differ from the kept rows (age, CHD rate).
Uses the same merge logic as Data_Set_Preparing.ipynb, stopping BEFORE dropna().

    python3 null_report.py raw_nhanes
"""
import sys
import pandas as pd
from dataset_funnel import CYCLES, find

COLS = {"RIDAGEYR": "age", "RIAGENDR": "sex", "BPXSY1": "sbp", "BPXDI1": "dbp",
        "LBDHDD": "hdl", "LBXTC": "total_chol", "BMXBMI": "bmi", "SMQ020": "smoking"}

def merged_before_dropna(demo, bpx, hdl, tchol, bmx, smq, mcq):
    hdl_col = "LBDHDD" if "LBDHDD" in hdl.columns else "LBXHDD"
    mcq = mcq[mcq["MCQ160C"].isin([1, 2])].copy()
    mcq["CHD"] = mcq["MCQ160C"].replace({1: 1, 2: 0})
    df = (demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
          .merge(bpx[["SEQN", "BPXSY1", "BPXDI1"]], on="SEQN")
          .merge(hdl[["SEQN", hdl_col]].rename(columns={hdl_col: "LBDHDD"}), on="SEQN")
          .merge(tchol[["SEQN", "LBXTC"]], on="SEQN")
          .merge(bmx[["SEQN", "BMXBMI"]], on="SEQN")
          .merge(smq[["SEQN", "SMQ020"]], on="SEQN")
          .merge(mcq[["SEQN", "CHD"]], on="SEQN"))
    return df

def report(df):
    n = len(df)
    nulls = df[list(COLS) ].isna().sum().rename(index=COLS)
    any_null = df[list(COLS)].isna().any(axis=1)
    print(f"Linked records before dropna: {n:,}")
    print("\nMissing values per column (count and % of linked records):")
    for c, v in nulls.items():
        print(f"  {c:<11}{int(v):>7,}  {v / n:6.1%}")
    k = int(any_null.sum())
    print(f"\nRecords with >= 1 missing value (deleted): {k:,} = {k / n:.1%} of linked records")
    print(f"Records kept (complete): {n - k:,}")
    print("\nAre the deleted records different from the kept ones?")
    for name, part in (("deleted", df[any_null]), ("kept", df[~any_null])):
        print(f"  {name:<8} n={len(part):>6,} | mean age {part['RIDAGEYR'].mean():5.1f} | CHD rate {part['CHD'].mean():.2%}")
    return nulls, k, n

def main(folder):
    parts = []
    for cyc, L in CYCLES.items():
        f = {s: pd.read_sas(find(folder, s, L), format="xport")
             for s in ["DEMO", "BPX", "HDL", "TCHOL", "BMX", "SMQ", "MCQ"]}
        parts.append(merged_before_dropna(f["DEMO"], f["BPX"], f["HDL"], f["TCHOL"], f["BMX"], f["SMQ"], f["MCQ"]))
    report(pd.concat(parts, ignore_index=True))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_nhanes")
