"""
=========================================================
STEP 4 — Deep Neural Network (DNN) Classifier
=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score
)
import seaborn as sns

# ── Setup ───────────────────────────────────────────────
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data (✅ FIXED PATHS) ──────────────────────────
X_train = np.load(os.path.join(OUTPUT_DIR, "X_train_sel.npy"))
X_val   = np.load(os.path.join(OUTPUT_DIR, "X_val_sel.npy"))
X_test  = np.load(os.path.join(OUTPUT_DIR, "X_test_sel.npy"))

y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

INPUT_DIM = X_train.shape[1]
EPOCHS    = 60
BATCH     = 32

print("=" * 60)
print("  STEP 4 — DEEP NEURAL NETWORK")
print("=" * 60)

# ── Build Model ─────────────────────────────────────────
def build_dnn(input_dim):
    inp = keras.Input(shape=(input_dim,))

    x = keras.layers.Dense(256)(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    out = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_dnn(INPUT_DIM)
model.summary()

# ── Callbacks ───────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7
    ),
    keras.callbacks.ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "dnn_best.h5"),
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ── Train ───────────────────────────────────────────────
print("\nTraining DNN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ────────────────────────────────────────────
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc*100:.2f}%")
print(f"AUC: {roc_auc:.4f}")
print("\n", classification_report(y_test, y_pred))

# ── Save Predictions ────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "dnn_y_prob.npy"), y_prob)
np.save(os.path.join(OUTPUT_DIR, "dnn_y_pred.npy"), y_pred)

# ── Plot Training Curves ────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dnn_training_curves.png"))
plt.close()

# ── Confusion Matrix ────────────────────────────────────
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "dnn_confusion_matrix.png"))
plt.close()

# ── ROC Curve ───────────────────────────────────────────
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.legend()
plt.title("ROC Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "dnn_roc_curve.png"))
plt.close()

# ── Confidence Table ────────────────────────────────────
conf_df = pd.DataFrame({
    "Actual": y_test[:10],
    "Predicted": y_pred[:10],
    "Confidence": np.round(y_prob[:10], 4)
})

print("\nSample Predictions:")
print(conf_df)

conf_df.to_csv(os.path.join(OUTPUT_DIR, "dnn_confidence_sample.csv"), index=False)

print("\n✓ STEP 4 COMPLETE\n")