"""
Download the raw NHANES files needed for dataset_funnel.py (7 cycles x 7 files = 49 files).
Saves them to ./raw_nhanes/ and skips files that already exist.

    python3 download_nhanes.py
    python3 dataset_funnel.py raw_nhanes
"""
import os, urllib.request

CYCLES = {"2005": "D", "2007": "E", "2009": "F", "2011": "G", "2013": "H", "2015": "I", "2017": "J"}
FILES = ["DEMO", "BPX", "HDL", "TCHOL", "BMX", "SMQ", "MCQ"]
OUT = "raw_nhanes"
os.makedirs(OUT, exist_ok=True)

def urls(year, stem, letter):
    name = f"{stem}_{letter}"
    return [f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt",
            f"https://wwwn.cdc.gov/Nchs/Nhanes/{year}-{int(year)+1}/{name}.XPT"]

ok = bad = 0
for year, letter in CYCLES.items():
    for stem in FILES:
        dest = os.path.join(OUT, f"{stem}_{letter}.XPT")
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            ok += 1; continue
        done = False
        for u in urls(year, stem, letter):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                data = urllib.request.urlopen(req, timeout=60).read()
                if len(data) > 1000 and not data.lstrip().startswith(b"<"):   # not an HTML error page
                    open(dest, "wb").write(data); done = True; break
            except Exception as e:
                last = e
        print(("OK      " if done else "FAILED  ") + f"{stem}_{letter}")
        ok += done; bad += (not done)
print(f"\n{ok} files ready, {bad} failed. Files are in ./{OUT}/")
if bad:
    print("For failed files, download them manually from the CDC NHANES website (Data > Questionnaires/Examination/Laboratory/Demographics) and put them in the folder.")
