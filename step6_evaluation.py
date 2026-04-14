"""
=========================================================
STEP 6 — Final Evaluation & Model Comparison
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
)

# ── Setup ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data (✅ FIXED PATHS) ──────────────────────────
y_test       = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

dnn_y_prob   = np.load(os.path.join(OUTPUT_DIR, "dnn_y_prob.npy"))
dnn_y_pred   = np.load(os.path.join(OUTPUT_DIR, "dnn_y_pred.npy"))

rbfnn_y_prob = np.load(os.path.join(OUTPUT_DIR, "rbfnn_y_prob.npy"))
rbfnn_y_pred = np.load(os.path.join(OUTPUT_DIR, "rbfnn_y_pred.npy"))

# ── Metrics Function ────────────────────────────────────
def metrics(y_true, y_pred, y_prob, name):
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision_score(y_true, y_pred, average="macro") * 100, 2),
        "Recall":    round(recall_score(y_true, y_pred, average="macro") * 100, 2),
        "F1-Score":  round(f1_score(y_true, y_pred, average="macro") * 100, 2),
        "AUC-ROC":   round(roc_auc_score(y_true, y_prob), 4),
    }

# ── Compute Results ─────────────────────────────────────
results = pd.DataFrame([
    metrics(y_test, dnn_y_pred,   dnn_y_prob,   "DNN"),
    metrics(y_test, rbfnn_y_pred, rbfnn_y_prob, "RBFNN"),
])

print("=" * 65)
print("  STEP 6 — FINAL MODEL COMPARISON")
print("=" * 65)
print(results)

results.to_csv(os.path.join(OUTPUT_DIR, "final_comparison.csv"), index=False)

# ── Plot 1: Metrics Comparison ──────────────────────────
plt.figure(figsize=(10, 5))

metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics_list))
width = 0.35

for i, (_, row) in enumerate(results.iterrows()):
    plt.bar(x + i*width,
            [row[m] for m in metrics_list],
            width,
            label=row["Model"])

plt.xticks(x + width/2, metrics_list)
plt.ylabel("Score (%)")
plt.title("DNN vs RBFNN Comparison")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_metrics.png"))
plt.close()

# ── Plot 2: ROC Curves ──────────────────────────────────
plt.figure()

for name, y_prob in [
    ("DNN", dnn_y_prob),
    ("RBFNN", rbfnn_y_prob),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.legend()
plt.title("ROC Curve Comparison")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_roc.png"))
plt.close()

# ── Plot 3: Confusion Matrices ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, name, y_pred in [
    (axes[0], "DNN", dnn_y_pred),
    (axes[1], "RBFNN", rbfnn_y_pred),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    acc = accuracy_score(y_test, y_pred) * 100
    ax.set_title(f"{name} (Acc={acc:.2f}%)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_confusion_matrices.png"))
plt.close()

# ── Summary ─────────────────────────────────────────────
best = results.loc[results["Accuracy"].idxmax(), "Model"]

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"\nBest Model: {best}")

for _, row in results.iterrows():
    print(f"\n{row['Model']}")
    for k in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
        print(f"{k}: {row[k]}")

print("\n✓ STEP 6 COMPLETE\n")