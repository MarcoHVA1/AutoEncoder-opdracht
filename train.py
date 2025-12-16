# src/train.py
"""
Complete training script voor Surface Crack Anomaly Detection
Train autoencoder op NORMALE beelden, test op normale + anomalies.
"""

import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Zorgen dat we vanaf project-root modules kunnen importeren
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import create_train_val_test_split
from models.autoencoder import build_autoencoder


# ============================================================
# Helper functies
# ============================================================

def calculate_reconstruction_errors(model, images):
    """
    Bereken reconstruction error (MSE per afbeelding).
    """
    recon = model.predict(images, batch_size=32, verbose=0)
    mse = np.mean((images - recon) ** 2, axis=(1, 2, 3))
    return mse, recon


def compute_reconstruction_metrics(original, reconstruction):
    """
    Bereken MSE, RMSE en MAPE.
    """
    diff = original - reconstruction
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(diff) / (np.abs(original) + 1e-8)) * 100
    return mse, rmse, mape


def find_threshold(model, X_val, percentile=95):
    """
    Bepaal threshold op basis van validation errors (percentiel).
    """
    errors, _ = calculate_reconstruction_errors(model, X_val)
    thr = np.percentile(errors, percentile)
    print(f"\n🎚 Threshold (P{percentile}): {thr:.6f}")
    return thr


def evaluate_autoencoder(model, X_test, y_test, threshold):
    """
    Evalueer autoencoder als anomaly detector.
    """
    errors, recon = calculate_reconstruction_errors(model, X_test)
    y_pred = (errors > threshold).astype(int)

    print("\n📊 AUTOENCODER – Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal", "Anomaly"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (autoencoder):")
    print(cm)

    # Reconstruction metrics per klasse
    normal_mask = y_test == 0
    anomaly_mask = y_test == 1

    mse_n, rmse_n, mape_n = compute_reconstruction_metrics(
        X_test[normal_mask], recon[normal_mask]
    )
    mse_a, rmse_a, mape_a = compute_reconstruction_metrics(
        X_test[anomaly_mask], recon[anomaly_mask]
    )

    print("\n📈 Reconstruction Metrics:")
    print(f"Normal : MSE={mse_n:.6f}, RMSE={rmse_n:.6f}, MAPE={mape_n:.2f}%")
    print(f"Anomaly: MSE={mse_a:.6f}, RMSE={rmse_a:.6f}, MAPE={mape_a:.2f}%")

    try:
        auc = roc_auc_score(y_test, errors)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        print("ROC AUC: N/A")

    return {
        "errors": errors,
        "recon": recon,
        "cm": cm,
        "mse_normal": mse_n,
        "rmse_normal": rmse_n,
        "mape_normal": mape_n,
        "mse_anomaly": mse_a,
        "rmse_anomaly": rmse_a,
        "mape_anomaly": mape_a,
    }


# ============================================================
# K-means baseline
# ============================================================

def evaluate_kmeans_baseline(encoder, X_train, X_test, y_test):
    """
    Vergelijkingsmodel: K-means clustering in latent space.
    """
    print("\n🔁 K-MEANS BASELINE (latent space)")

    Z_train = encoder.predict(X_train, batch_size=64, verbose=0).reshape(len(X_train), -1)
    Z_test = encoder.predict(X_test, batch_size=64, verbose=0).reshape(len(X_test), -1)

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_test_s = scaler.transform(Z_test)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(Z_train_s)

    train_clusters = kmeans.predict(Z_train_s)
    counts = np.bincount(train_clusters)
    normal_cluster = np.argmax(counts)

    test_clusters = kmeans.predict(Z_test_s)
    y_pred = (test_clusters != normal_cluster).astype(int)

    print("\n📊 K-means – Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Normal", "Anomaly"], digits=4
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (K-means):")
    print(cm)

    return cm


# ============================================================
# Visualisaties
# ============================================================

def plot_training_history(history, save_path="models/saved_models/training_history.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("Training history")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Training history plot saved to {save_path}")


def plot_recon_examples(X_test, recon, save_path="models/saved_models/recon_examples.png"):
    n = min(5, len(X_test))
    idx = np.random.choice(len(X_test), n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i, ix in enumerate(idx):
        axes[0, i].imshow(X_test[ix])
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(recon[ix])
        axes[1, i].axis("off")
        axes[1, i].set_title("Recon")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Reconstruction examples saved to {save_path}")


# ============================================================
# Main pipeline
# ============================================================

def main():
    print("\n=== SURFACE CRACK ANOMALY DETECTION TRAINING ===\n")

    # Data inladen met beperking (5000 per klasse)
    X_train, X_val, X_test, y_test = create_train_val_test_split(
        train_ratio=0.7,
        val_ratio=0.15,
        max_normal=5000,
        max_anomaly=5000,
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")

    # Model bouwen
    input_shape = (64, 64, 3)
    autoencoder, encoder = build_autoencoder(input_shape=input_shape)
    autoencoder.summary()

    os.makedirs("models/saved_models", exist_ok=True)

    # Trainen (minder epochs voor snelheid)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            "models/saved_models/best_autoencoder.h5",
            monitor="val_loss",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=15,          # <-- hier minder epochs
        batch_size=32,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
    )

    autoencoder.save("models/saved_models/final_autoencoder.h5")
    print("💾 Final autoencoder saved.")

    # Training plot
    plot_training_history(history)

    # Threshold bepalen
    threshold = find_threshold(autoencoder, X_val, percentile=95)

    # Evalueren autoencoder
    results = evaluate_autoencoder(autoencoder, X_test, y_test, threshold)
    plot_recon_examples(X_test, results["recon"])

    # K-means baseline
    evaluate_kmeans_baseline(encoder, X_train, X_test, y_test)

    # Metrics opslaan voor dashboard
    cm = results["cm"]
    accuracy = float((cm[0, 0] + cm[1, 1]) / np.sum(cm))
    precision = float(cm[1, 1] / max(cm[0, 1] + cm[1, 1], 1))
    recall = float(cm[1, 1] / max(cm[1, 0] + cm[1, 1], 1))
    f1 = float(
        2
        * (precision * recall)
        / max(precision + recall, 1e-8)
    )

    metrics_to_save = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse_normal": float(results["mse_normal"]),
        "mse_anomaly": float(results["mse_anomaly"]),
        "mape_normal": float(results["mape_normal"]),
        "mape_anomaly": float(results["mape_anomaly"]),
    }

    metrics_path = os.path.join("models", "saved_models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"💾 Metrics opgeslagen in {metrics_path}")

    print("\n✅ Training pipeline voltooid.\n")


if __name__ == "__main__":
    main()