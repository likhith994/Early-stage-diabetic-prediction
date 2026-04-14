"""
=========================================================
FULL PIPELINE — Early-Stage Diabetes Risk Prediction
=========================================================
Runs all 6 steps sequentially
=========================================================
"""

import time
import subprocess
import sys
import os

# ── Setup ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEPS = [
    ("Step 1 — Exploratory Data Analysis",          "step1_eda.py"),
    ("Step 2 — Preprocessing + SMOTE-ENN + GAN",   "step2_preprocessing_augmentation.py"),
    ("Step 3 — SHAP Feature Selection",             "step3_shap_feature_selection.py"),
    ("Step 4 — DNN Model",                          "step4_dnn_model.py"),
    ("Step 5 — RBFNN Model",                        "step5_rbfnn_model.py"),
    ("Step 6 — Final Evaluation & Comparison",      "step6_evaluation.py"),
]

print("=" * 65)
print("  EARLY-STAGE DIABETES RISK PREDICTION — FULL PIPELINE")
print("=" * 65)

total_start = time.time()

# ── Run Steps ───────────────────────────────────────────
for i, (label, script) in enumerate(STEPS, 1):
    print(f"\n{'─'*65}")
    print(f"  [{i}/{len(STEPS)}]  {label}")
    print(f"{'─'*65}")

    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ {label} FAILED  (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n  ✓ Done in {elapsed:.1f}s")

# ── Completion ──────────────────────────────────────────
total = time.time() - total_start

print(f"\n{'='*65}")
print(f"  PIPELINE COMPLETE  ({total:.1f}s total)")
print(f"{'='*65}")

# ── Output Files ────────────────────────────────────────
print("\nOutput files (saved in /outputs folder):")

outputs = [
    "eda_distributions.png",
    "eda_correlation.png",
    "eda_class_balance.png",
    "class_balance_comparison.png",
    "gan_training_loss.png",
    "shap_importance.png",
    "shap_summary.png",
    "dnn_training_curves.png",
    "dnn_confusion_matrix.png",
    "dnn_roc_curve.png",
    "rbfnn_training_curves.png",
    "rbfnn_confusion_matrix.png",
    "rbfnn_roc_curve.png",
    "comparison_metrics.png",
    "comparison_roc.png",
    "comparison_confusion_matrices.png",
    "final_comparison.csv",
    "dnn_confidence_sample.csv",
    "shap_importance.csv",
]

for f in outputs:
    print(f"  {os.path.join(OUTPUT_DIR, f)}")