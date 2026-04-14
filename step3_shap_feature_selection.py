"""
=========================================================
STEP 3 — SHAP-Based Feature Selection
=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier

# ── Setup ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 7

print("=" * 60)
print("  STEP 3 — SHAP FEATURE SELECTION")
print("=" * 60)

# ── Load Data ───────────────────────────────────────────
X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))

X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))

X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

with open(os.path.join(OUTPUT_DIR, "features.pkl"), "rb") as f:
    FEATURES = pickle.load(f)

# ── Train Random Forest ─────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_acc = rf.score(X_test, y_test)
print(f"\nRF Accuracy: {rf_acc:.4f}")

# ── SHAP Computation ────────────────────────────────────
print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)

# Handle SHAP formats
sv = np.array(shap_values)

if sv.ndim == 3:
    if sv.shape[-1] == 2:
        sv = sv[:, :, 1]
    elif sv.shape[0] == 2:
        sv = sv[1]
elif isinstance(shap_values, list):
    sv = np.array(shap_values[1])

mean_shap = np.abs(sv).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": FEATURES,
    "Mean_SHAP": mean_shap
}).sort_values("Mean_SHAP", ascending=False)

print("\nFeature Ranking:")
print(shap_df)

# ── Select Top Features ─────────────────────────────────
top_features = shap_df["Feature"].iloc[:TOP_K].tolist()
top_indices = [FEATURES.index(f) for f in top_features]

X_train_sel = X_train[:, top_indices]
X_val_sel   = X_val[:, top_indices]
X_test_sel  = X_test[:, top_indices]

print(f"\nTop {TOP_K} features: {top_features}")

# ── Save Files ──────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "X_train_sel.npy"), X_train_sel)
np.save(os.path.join(OUTPUT_DIR, "X_val_sel.npy"), X_val_sel)
np.save(os.path.join(OUTPUT_DIR, "X_test_sel.npy"), X_test_sel)

with open(os.path.join(OUTPUT_DIR, "top_features.pkl"), "wb") as f:
    pickle.dump(top_features, f)

with open(os.path.join(OUTPUT_DIR, "top_indices.pkl"), "wb") as f:
    pickle.dump(top_indices, f)

shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)

# ── Plot 1: SHAP Importance ─────────────────────────────
plt.figure(figsize=(8, 6))

colors = ["blue" if f in top_features else "gray" for f in shap_df["Feature"]]

plt.barh(shap_df["Feature"][::-1],
         shap_df["Mean_SHAP"][::-1],
         color=colors[::-1])

plt.title("SHAP Feature Importance")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"))
plt.close()

# ── Plot 2: SHAP Summary ────────────────────────────────
plt.figure()

shap.summary_plot(sv, X_train, feature_names=FEATURES, show=False)

plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"))
plt.close()

print("\n✓ STEP 3 COMPLETE\n")