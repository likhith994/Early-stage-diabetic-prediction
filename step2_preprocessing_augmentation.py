"""
=========================================================
STEP 2 — Preprocessing + Hybrid SMOTE-ENN + GAN
=========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import pickle
import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Setup ───────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ──────────────────────────────────────────────
DATA_PATH = "pima-indians-diabetes.csv"
RANDOM_SEED = 42
GAN_EPOCHS = 500
GAN_BATCH = 64
LATENT_DIM = 32
EXTRA_SAMPLES = 200

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ── Load Data ───────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"].copy()

FEATURES = list(X.columns)
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("=" * 60)
print("  STEP 2 — PREPROCESSING + SMOTE-ENN + GAN")
print("=" * 60)

# ── Fix zero values ─────────────────────────────────────
for col in ZERO_COLS:
    median_val = X.loc[X[col] != 0, col].median()
    X[col] = X[col].replace(0, median_val)

# ── Standardize ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── SMOTE-ENN ───────────────────────────────────────────
smote_enn = SMOTEENN(random_state=RANDOM_SEED)
X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)

# ── GAN Setup ───────────────────────────────────────────
minority_X = X_resampled[y_resampled == 1].astype(np.float32)
INPUT_DIM = minority_X.shape[1]

def build_generator():
    return keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(LATENT_DIM,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(INPUT_DIM, activation="tanh"),
    ])

def build_discriminator():
    return keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(INPUT_DIM,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

generator = build_generator()
discriminator = build_discriminator()

loss_fn = keras.losses.BinaryCrossentropy()
gen_opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
disc_opt = keras.optimizers.Adam(0.0002, beta_1=0.5)

@tf.function
def train_step(real_batch):
    noise = tf.random.normal([tf.shape(real_batch)[0], LATENT_DIM])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(noise, training=True)
        real_out = discriminator(real_batch, training=True)
        fake_out = discriminator(fake, training=True)

        d_loss = loss_fn(tf.ones_like(real_out), real_out) + \
                 loss_fn(tf.zeros_like(fake_out), fake_out)
        g_loss = loss_fn(tf.ones_like(fake_out), fake_out)

    gen_opt.apply_gradients(zip(gt.gradient(g_loss, generator.trainable_variables),
                                generator.trainable_variables))
    disc_opt.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables),
                                 discriminator.trainable_variables))
    return g_loss, d_loss

# ── Train GAN ───────────────────────────────────────────
dataset = tf.data.Dataset.from_tensor_slices(minority_X).batch(GAN_BATCH)

g_losses, d_losses = [], []

for epoch in range(GAN_EPOCHS):
    for batch in dataset:
        gl, dl = train_step(batch)

    g_losses.append(float(gl))
    d_losses.append(float(dl))

# ── Generate synthetic samples ──────────────────────────
noise = tf.random.normal([EXTRA_SAMPLES, LATENT_DIM])
synthetic_X = generator(noise).numpy()
synthetic_y = np.ones(EXTRA_SAMPLES)

X_final = np.vstack([X_resampled, synthetic_X])
y_final = np.concatenate([y_resampled, synthetic_y])

# ── Split ───────────────────────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_final, y_final, test_size=0.2, stratify=y_final)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.125, stratify=y_trainval)

# ── Save Files ──────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)

np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

# ✅ CRITICAL FIX (this was missing)
with open(os.path.join(OUTPUT_DIR, "features.pkl"), "wb") as f:
    pickle.dump(FEATURES, f)

with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ── Plot GAN Loss ───────────────────────────────────────
plt.figure()
plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.legend()
plt.title("GAN Training Loss")
plt.savefig(os.path.join(OUTPUT_DIR, "gan_training_loss.png"))
plt.close()

# ── Class Balance Plot ──────────────────────────────────
plt.figure()
plt.bar(["0", "1"], [sum(y_final==0), sum(y_final==1)])
plt.title("Final Class Balance")
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance_comparison.png"))
plt.close()

print("\n✓ STEP 2 COMPLETE\n")