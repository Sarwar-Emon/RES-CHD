"""
Dataset funnel: raw NHANES participants -> final analytic sample, per cycle.
Replicates the logic of Data_Set_Preparing.ipynb (clean_nhanes_cycle) step by step
and prints how many rows remain after each step.

Run (folder that contains the raw *.XPT files, sub-folders are searched):
    python dataset_funnel.py /path/to/raw_xpt_folder
"""
import os, sys, glob
import pandas as pd

CYCLES = {"2005_2006": "D", "2007_2008": "E", "2009_2010": "F", "2011_2012": "G",
          "2013_2014": "H", "2015_2016": "I", "2017_2018": "J"}
# Final counts per cycle printed by your notebook (value_counts) -> used as a check
EXPECTED_FINAL = {"2005_2006": 3829, "2007_2008": 4855, "2009_2010": 5158, "2011_2012": 4430,
                  "2013_2014": 4799, "2015_2016": 4766, "2017_2018": 4281}

def funnel(demo, bpx, hdl, tchol, bmx, smq, mcq):
    n = {"1_all_participants (DEMO)": len(demo)}
    hdl_col = "LBDHDD" if "LBDHDD" in hdl.columns else "LBXHDD"
    mcq = mcq[mcq["MCQ160C"].isin([1, 2])].copy()
    n["2_answered_CHD_question (MCQ160C = 1 or 2)"] = len(mcq)
    mcq["CHD"] = mcq["MCQ160C"].replace({1: 1, 2: 0})
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    bpx = bpx[["SEQN", "BPXSY1", "BPXDI1"]]
    hdl = hdl[["SEQN", hdl_col]].rename(columns={hdl_col: "LBDHDD"})
    tchol = tchol[["SEQN", "LBXTC"]]
    bmx = bmx[["SEQN", "BMXBMI"]]
    smq = smq[["SEQN", "SMQ020"]]
    mcq = mcq[["SEQN", "CHD"]]
    df = demo.merge(bpx, on="SEQN");   n["3_with_blood_pressure"] = len(df)
    df = df.merge(hdl, on="SEQN");     n["4_with_HDL"] = len(df)
    df = df.merge(tchol, on="SEQN");   n["5_with_total_cholesterol"] = len(df)
    df = df.merge(bmx, on="SEQN");     n["6_with_BMI"] = len(df)
    df = df.merge(smq, on="SEQN");     n["7_with_smoking_question"] = len(df)
    df = df.merge(mcq, on="SEQN");     n["8_with_CHD_label"] = len(df)
    df = df.dropna();                  n["9_after_dropna (no missing)"] = len(df)
    df = df[(df["BPXSY1"] > 50) & (df["BPXSY1"] < 250)]; n["10_SBP in (50,250)"] = len(df)
    df = df[(df["BPXDI1"] > 30) & (df["BPXDI1"] < 150)]; n["11_DBP in (30,150)"] = len(df)
    df = df[df["BMXBMI"] < 70];                           n["12_BMI < 70"] = len(df)
    df = df[df["SMQ020"].isin([1, 2])];                   n["13_FINAL (smoking = 1 or 2)"] = len(df)
    n["_min_age_in_final"] = float(df["RIDAGEYR"].min()) if len(df) else None
    return n

def find(folder, stem, letter):
    want = f"{stem}_{letter}.xpt".lower()
    for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if os.path.basename(p).lower() == want:
            return p
    raise FileNotFoundError(f"{stem}_{letter}.XPT not found under {folder}")

def main(folder):
    rows = {}
    for cyc, L in CYCLES.items():
        try:
            f = {s: pd.read_sas(find(folder, s, L), format="xport")
                 for s in ["DEMO", "BPX", "HDL", "TCHOL", "BMX", "SMQ", "MCQ"]}
        except FileNotFoundError as e:
            print("SKIPPED", cyc, "->", e); continue
        rows[cyc] = funnel(f["DEMO"], f["BPX"], f["HDL"], f["TCHOL"], f["BMX"], f["SMQ"], f["MCQ"])
    if not rows:
        print("\nNo cycles could be loaded. Nothing to count.\n"
              "Check the folder path (it must contain DEMO_D.XPT ... DEMO_J.XPT and the other files),\n"
              "or run download_nhanes.py first to fetch them.")
        return
    t = pd.DataFrame(rows)
    num = t.drop(index="_min_age_in_final")
    num["TOTAL"] = num.sum(axis=1)
    pd.set_option("display.width", 220)
    print(num.astype(int).to_string())
    print("\nMin age in final data per cycle:", t.loc["_min_age_in_final"].to_dict())
    tot_raw, tot_final = int(num.loc["1_all_participants (DEMO)", "TOTAL"]), int(num.loc["13_FINAL (smoking = 1 or 2)", "TOTAL"])
    print(f"\nRAW participants (all ages): {tot_raw:,}  ->  FINAL: {tot_final:,}  ({tot_final / tot_raw:.1%} retained)")
    print("\nCHECK final vs notebook value_counts:")
    for c, exp in EXPECTED_FINAL.items():
        if c in rows:
            got = rows[c]["13_FINAL (smoking = 1 or 2)"]
            print(f"  {c}: {got:,} vs {exp:,}  {'OK' if got == exp else '<-- MISMATCH'}")
    num.to_csv("dataset_funnel.csv")
    print("\nSaved dataset_funnel.csv")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
