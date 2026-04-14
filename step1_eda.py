"""
=========================================================
STEP 1 — Exploratory Data Analysis (EDA)
=========================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Setup ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ───────────────────────────────────────────
DATA_PATH = "pima-indians-diabetes.csv"   # keep file in same folder
# OR use this if needed:
# DATA_PATH = r"C:\dl_project\pima-indians-diabetes.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("  PIMA INDIAN DIABETES — EDA REPORT")
print("=" * 60)

print(f"\nShape          : {df.shape}")
print(f"Missing values : {df.isnull().sum().sum()}")

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe().round(2))

# ── Class Balance ───────────────────────────────────────
print("\nClass distribution:")
print(df["Outcome"].value_counts())

ratio = df["Outcome"].value_counts()[0] / df["Outcome"].value_counts()[1]
print(f"  → Imbalance ratio: {ratio:.2f}:1")

# ── Zero-value check ────────────────────────────────────
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero-value counts (clinically invalid):")
for col in zero_cols:
    n = (df[col] == 0).sum()
    pct = n / len(df) * 100
    print(f"  {col:25s}: {n:3d}  ({pct:.1f}%)")

# ── Plot 1: Distributions ───────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Feature Distributions by Outcome", fontsize=16)

cols = df.columns[:-1]
palette = {0: "#4C72B0", 1: "#DD8452"}

for ax, col in zip(axes.flatten(), cols):
    for outcome in [0, 1]:
        subset = df[df["Outcome"] == outcome][col]
        ax.hist(subset, bins=25, alpha=0.6,
                label="Diabetic" if outcome else "Non-Diabetic",
                color=palette[outcome])
    ax.set_title(col)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_distributions.png"))
plt.close()
print("\n[Saved] eda_distributions.png")

# ── Plot 2: Correlation ─────────────────────────────────
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.heatmap(df.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation.png"))
plt.close()
print("[Saved] eda_correlation.png")

# ── Plot 3: Class Balance ───────────────────────────────
plt.figure(figsize=(5, 5))
counts = df["Outcome"].value_counts()

plt.pie(counts,
        labels=["Non-Diabetic", "Diabetic"],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"])

plt.title("Class Distribution")
plt.savefig(os.path.join(OUTPUT_DIR, "eda_class_balance.png"))
plt.close()
print("[Saved] eda_class_balance.png")

print("\n✓ STEP 1 COMPLETE\n")