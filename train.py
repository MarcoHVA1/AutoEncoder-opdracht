"""
Autoencoder training voor anomaly detection.
Trainen op NORMAL beelden.
Evalueren op NORMAL + ANOMALY beelden.
"""

import os
import json
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from models.autoencoder import build_autoencoder


# ==============================
# Config
# ==============================

IMG_SIZE = (64, 64)
NORMAL_DIR = "Normal"
ANOMALY_DIR = "Anomaly"
SAVE_DIR = "models/saved_models"
EPOCHS = 20
BATCH_SIZE = 16
THRESHOLD_PERCENTILE = 95


# ==============================
# Data loading
# ==============================

def load_images_from_folder(folder):
    images = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        images.append(img)
    return np.array(images)


# ==============================
# Main
# ==============================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    X_normal = load_images_from_folder(NORMAL_DIR)
    X_anomaly = load_images_from_folder(ANOMALY_DIR)

    print(f"Normal images  : {len(X_normal)}")
    print(f"Anomaly images : {len(X_anomaly)}")

    # Train / val split (ALLEEN normal)
    X_train, X_val = train_test_split(
        X_normal, test_size=0.2, random_state=42
    )

    # Testset (normal + anomaly)
    X_test = np.concatenate([X_normal, X_anomaly], axis=0)
    y_test = np.array(
        [0] * len(X_normal) + [1] * len(X_anomaly)
    )

    # Model
    autoencoder, _ = build_autoencoder(input_shape=(64, 64, 3))
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.summary()

    # Training
    autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # Save model
    model_path = os.path.join(SAVE_DIR, "final_autoencoder.h5")
    autoencoder.save(model_path)

    # ==============================
    # Threshold bepalen
    # ==============================

    recon_val = autoencoder.predict(X_val)
    val_errors = np.mean((X_val - recon_val) ** 2, axis=(1, 2, 3))
    threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)

    np.save(os.path.join(SAVE_DIR, "threshold.npy"), threshold)

    print(f"Threshold: {threshold:.6f}")

    # ==============================
    # Evaluatie
    # ==============================

    recon_test = autoencoder.predict(X_test)
    test_errors = np.mean((X_test - recon_test) ** 2, axis=(1, 2, 3))
    y_pred = (test_errors > threshold).astype(int)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Metrics voor dashboard
    accuracy = float((cm[0, 0] + cm[1, 1]) / np.sum(cm))
    precision = float(cm[1, 1] / max(cm[0, 1] + cm[1, 1], 1))
    recall = float(cm[1, 1] / max(cm[1, 0] + cm[1, 1], 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-8))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse_normal": float(test_errors[y_test == 0].mean()),
        "mse_anomaly": float(test_errors[y_test == 1].mean()),
    }

    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training afgerond.")


if __name__ == "__main__":
    main()
