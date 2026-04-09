"""
RES-CHD — Master Reproducibility Script
========================================
Thesis: RES-CHD: A Reliability-Aware Explainability Stability Framework
        for Coronary Heart Disease Risk Prediction under Progressive Data Scarcity

Author: [Your Name]
Institution: Troy University, Department of Computer Science
Year: 2026

Purpose:
  This script runs the complete RES-CHD pipeline from scratch.
  All results in the thesis are fully reproducible by running this script.
  Random seeds are fixed at 42 throughout all phases.

Dataset required:
  NHANES 2005-2018 unified CSV at: data/preprocessed/nhanes_base.csv
  Cleveland Heart Disease data at: data/cleveland/processed.cleveland.data
    (auto-downloaded from UCI if not present)

Environment:
  Python 3.12 (Anaconda)
  numpy==1.26.4 (pinned — numpy 2.0 breaks dependencies)
  See requirements below for full package list.

Install requirements:
  pip install pandas numpy==1.26.4 scipy scikit-learn xgboost
              imbalanced-learn shap joblib matplotlib

Estimated total runtime:
  Phase 1:  ~2 minutes
  Phase 2:  ~15 minutes (training 28 models)
  Phase 3:  ~30 minutes (SHAP extraction including MLP KernelExplainer)
  Phase 4:  ~2 minutes  (SHAP-array bootstrap)
  Phase 4B: ~1 minute   (MLP bootstrap)
  Phase 5:  ~1 minute   (ablation study)
  Phase 5B: ~5 minutes  (distributional shift analysis)
  Phase 5C: ~1 minute   (ESI baseline comparison)
  Phase 5D: ~1 minute   (local SHAP stability)
  Phase 5E: ~3 minutes  (confidence vs stability)
  Phase 5F: ~3 minutes  (calibration analysis)
  Phase 6:  ~1 minute   (original figures)
  Phase 6B: ~1 minute   (new figures)
  Cleveland: ~8 minutes (external validation)
  Total:    ~75 minutes

Usage:
  python run_all.py                    # run everything
  python run_all.py --phase 3          # run specific phase
  python run_all.py --skip-cleveland   # skip external validation
  python run_all.py --check-only       # verify all outputs exist
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Pipeline definition ────────────────────────────────────────────────────────
# Each entry: (phase_id, script_name, description, estimated_minutes)
PIPELINE = [
    ("1",        "phase1_preprocessing.py",       "Data preprocessing & scarcity splits",        2),
    ("2",        "phase2_training.py",             "Model training (28 models × 7 levels)",       15),
    ("3",        "phase3_shap.py",                 "SHAP extraction & global ESI",                30),
    ("4",        "phase4_framework.py",            "RES-CHD reliability framework + bootstrap",   2),
    ("4b",       "phase4B_mlp_bootstrap.py",       "MLP bootstrapped ESI (SHAP-array method)",    1),
    ("5",        "phase5_ablation.py",             "Ablation study (5 experiments)",              1),
    ("5b",       "phase5b_distribution_shift.py",  "Distributional shift analysis",               5),
    ("5c",       "phase5c_esi_baselines.py",       "ESI baseline comparison",                     1),
    ("5d",       "phase5d_local_stability.py",     "Local SHAP stability per patient",            1),
    ("5e",       "phase5e_confidence_stability.py","Prediction confidence vs explanation stability",3),
    ("5f",       "phase5f_calibration.py",         "Calibration analysis (Brier, ECE, CSI)",      3),
    ("6",        "phase6_visualization.py",        "Publication figures (original 10)",           1),
    ("6b",       "phase6b_master_figures.py",      "Updated master figures (all 18)",             1),
    ("cleveland","cleveland_validation.py",        "External validation on Cleveland dataset",    8),
]

TOTAL_MINUTES = sum(p[3] for p in PIPELINE)


def check_script_exists(script: str) -> bool:
    return (BASE_DIR / script).exists()


def run_phase(phase_id: str, script: str,
              description: str, est_minutes: int) -> bool:
    """Run a single phase script and report success/failure."""
    script_path = BASE_DIR / script
    if not script_path.exists():
        print(f"  [MISSING] {script} — script not found, skipping")
        return False

    print(f"\n  {'='*60}")
    print(f"  Phase {phase_id}: {description}")
    print(f"  Script: {script}")
    print(f"  Estimated: ~{est_minutes} min")
    print(f"  {'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  [OK] Completed in {elapsed/60:.1f} min")
        return True
    else:
        print(f"  [FAILED] Exit code {result.returncode} after {elapsed/60:.1f} min")
        return False


def check_outputs() -> None:
    """Verify all key output files exist."""
    print("\nChecking output files ...")

    expected_outputs = [
        # Phase 1
        "data/scarcity_levels/S1_train.csv",
        "data/scarcity_levels/S7_test.csv",
        "data/preprocessed/scarcity_summary.csv",
        # Phase 2
        "models/S1_XGBoost.pkl",
        "models/S7_MLP.pkl",
        "results/metrics_per_level.csv",
        # Phase 3
        "shap/stability/esi_scores.csv",
        "shap/stability/rank_stability.csv",
        "shap/global/all_global_shap.csv",
        "shap/local/S1_RandomForest_local_shap.csv",
        "shap/ranks/S1_XGBoost_ranks.csv",
        # Phase 4
        "results/phase4/bootstrapped_esi.csv",
        "results/phase4/reliability_flags.csv",
        "results/phase4/res_chd_report.csv",
        "results/phase4/scarcity_thresholds.csv",
        "results/phase4/feature_reliability.csv",
        # Phase 5
        "results/phase5/ablation_esi_thresholds.csv",
        "results/phase5/ablation_kendall_tau.csv",
        "results/phase5/ablation_random_scarcity.csv",
        "results/phase5/final_consolidated_results.csv",
        "results/phase5/publication_summary.csv",
        # Phase 5B
        "results/phase5b/distributional_shift.csv",
        "results/phase5b/random_vs_temporal_drift.csv",
        "results/phase5b/cycle_drift.csv",
        # Phase 5C
        "results/phase5c/metric_comparison.csv",
        "results/phase5c/metric_correlations.csv",
        # Phase 5D
        "results/phase5d/patient_stability_scores.csv",
        "results/phase5d/model_local_esi.csv",
        # Phase 5E
        "results/phase5e/compound_risk_by_level.csv",
        "results/phase5e/quadrant_analysis.csv",
        # Phase 5F
        "results/phase5f/calibration_metrics.csv",
        "results/phase5f/calibration_stability.csv",
        # Phase 6 figures
        "results/phase6/figures/fig1_esi_stability_curves.png",
        "results/phase6/figures/fig9_auc_vs_esi.png",
        # Phase 6B figures
        "results/phase6b/figures/fig11_distributional_shift.png",
        "results/phase6b/figures/fig18_updated_esi_all_models.png",
        "results/phase6b/figure_index.csv",
        # Cleveland
        "results/cleveland/esi_scores.csv",
        "results/cleveland/comparison_table.csv",
        "results/cleveland/figures/fig17_cleveland_vs_nhanes.png",
        # Logs
        "logs/phase4_report.txt",
        "logs/phase5b_report.txt",
        "logs/cleveland_validation.txt",
    ]

    missing = []
    present = []
    for rel_path in expected_outputs:
        full = BASE_DIR / rel_path
        if full.exists():
            present.append(rel_path)
        else:
            missing.append(rel_path)

    print(f"\n  Present : {len(present)}/{len(expected_outputs)}")
    if missing:
        print(f"  Missing : {len(missing)} files")
        for m in missing:
            print(f"    - {m}")
    else:
        print("  All expected outputs present. ✓")


def print_summary(results: dict) -> None:
    """Print final run summary."""
    print(f"\n{'='*65}")
    print("  RES-CHD Pipeline Run Summary")
    print(f"{'='*65}")

    success = sum(1 for v in results.values() if v)
    total   = len(results)

    for phase_id, ok in results.items():
        status = "✓ OK    " if ok else "✗ FAILED"
        script = next((p[1] for p in PIPELINE if p[0] == phase_id), "?")
        print(f"  {status}  Phase {phase_id:<12}  {script}")

    print(f"\n  {success}/{total} phases completed successfully")

    if success == total:
        print("\n  All phases complete. Results ready for thesis writing. ✓")
    else:
        failed = [p for p, ok in results.items() if not ok]
        print(f"\n  Failed phases: {failed}")
        print("  Check individual log files in logs/ for details.")


def main():
    parser = argparse.ArgumentParser(
        description="RES-CHD Master Pipeline Runner"
    )
    parser.add_argument("--phase", type=str, default=None,
                       help="Run only this phase (e.g. --phase 3)")
    parser.add_argument("--skip-cleveland", action="store_true",
                       help="Skip external validation")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check if outputs exist, do not run")
    parser.add_argument("--from-phase", type=str, default=None,
                       help="Start from this phase (e.g. --from-phase 5b)")
    args = parser.parse_args()

    print("=" * 65)
    print("  RES-CHD — Master Reproducibility Script")
    print("=" * 65)
    print(f"  Working directory: {BASE_DIR}")
    print(f"  Python: {sys.executable}")
    print(f"  Estimated total runtime: ~{TOTAL_MINUTES} minutes")

    if args.check_only:
        check_outputs()
        return

    # Filter pipeline
    pipeline = PIPELINE.copy()

    if args.skip_cleveland:
        pipeline = [p for p in pipeline if p[0] != "cleveland"]
        print("\n  [INFO] Skipping Cleveland validation")

    if args.phase:
        pipeline = [p for p in pipeline if p[0] == args.phase]
        if not pipeline:
            print(f"  [ERROR] Phase '{args.phase}' not found")
            sys.exit(1)

    if args.from_phase:
        phase_ids = [p[0] for p in PIPELINE]
        if args.from_phase not in phase_ids:
            print(f"  [ERROR] Phase '{args.from_phase}' not found")
            sys.exit(1)
        start_idx = phase_ids.index(args.from_phase)
        pipeline  = PIPELINE[start_idx:]
        print(f"\n  [INFO] Starting from phase {args.from_phase}")

    print(f"\n  Phases to run: {[p[0] for p in pipeline]}")
    print(f"  Total phases : {len(pipeline)}")

    # Run pipeline
    results    = {}
    total_start = time.time()

    for phase_id, script, description, est_minutes in pipeline:
        ok = run_phase(phase_id, script, description, est_minutes)
        results[phase_id] = ok

        if not ok:
            print(f"\n  [WARNING] Phase {phase_id} failed.")
            answer = input("  Continue with next phase? [y/N]: ").strip().lower()
            if answer != "y":
                print("  Stopping pipeline.")
                break

    total_elapsed = time.time() - total_start
    print(f"\n  Total elapsed: {total_elapsed/60:.1f} minutes")

    print_summary(results)
    check_outputs()


if __name__ == "__main__":
    main()