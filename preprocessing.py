# src/preprocessing.py
import os
import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Project-root en datamap bepalen
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")  # -> data/raw
IMG_SIZE = (64, 64)


def load_and_preprocess(img_path, target_size=IMG_SIZE):
    """
    Laad een afbeelding, converteer naar RGB, resize naar target_size en
    normaliseer naar [0, 1].
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Kon afbeelding niet lezen: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img


def create_train_val_test_split(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_normal: int = 5000,
    max_anomaly: int = 5000,
):
    """
    Laad Negative = normale beelden, Positive = anomalieën uit data/raw.

    Structuur verwacht:
        data/raw/Negative/*.jpg|*.png|*.jpeg
        data/raw/Positive/*.jpg|*.png|*.jpeg

    Door max_normal en max_anomaly te gebruiken, versnellen we de training.

    Returns
    -------
    X_train : np.ndarray (N_train, 64, 64, 3)
    X_val   : np.ndarray (N_val, 64, 64, 3)
    X_test  : np.ndarray (N_test, 64, 64, 3)
    y_test  : np.ndarray (N_test,)  -- 0 = normaal, 1 = anomaly
    """
    normal_dir = os.path.join(DATA_DIR, "Negative")
    anomaly_dir = os.path.join(DATA_DIR, "Positive")

    exts = ("*.jpg", "*.jpeg", "*.png")

    # Alle normale beelden verzamelen
    normal_images = []
    for ext in exts:
        normal_images.extend(glob.glob(os.path.join(normal_dir, ext)))

    # Alle anomaliebeelden verzamelen
    anomaly_images = []
    for ext in exts:
        anomaly_images.extend(glob.glob(os.path.join(anomaly_dir, ext)))

    # Beperk aantal samples per klasse voor snelheid
    if max_normal is not None:
        normal_images = normal_images[:max_normal]
    if max_anomaly is not None:
        anomaly_images = anomaly_images[:max_anomaly]

    print(f"Gevonden {len(normal_images)} normale en {len(anomaly_images)} anomalieën.")

    if len(normal_images) == 0:
        raise RuntimeError(
            f"Geen normale beelden gevonden in {normal_dir}. "
            f"Controleer je mapstructuur en extensies."
        )

    # Normal train/val/test split
    train_paths, temp_paths = train_test_split(
        normal_images,
        test_size=1 - train_ratio,
        random_state=42,
        shuffle=True,
    )

    val_rel = val_ratio / (1 - train_ratio)
    val_paths, test_normal_paths = train_test_split(
        temp_paths,
        test_size=1 - val_rel,
        random_state=42,
        shuffle=True,
    )

    # Alle anomalies in testset
    test_anomaly_paths = anomaly_images

    def paths_to_array(paths):
        imgs = [load_and_preprocess(p) for p in paths]
        return np.stack(imgs, axis=0)

    X_train = paths_to_array(train_paths)
    X_val = paths_to_array(val_paths)
    X_test_normal = paths_to_array(test_normal_paths)
    X_test_anomaly = paths_to_array(test_anomaly_paths)

    # Combineer test normal + anomaly
    X_test = np.concatenate([X_test_normal, X_test_anomaly], axis=0)
    y_test = np.concatenate(
        [
            np.zeros(len(X_test_normal), dtype=int),
            np.ones(len(X_test_anomaly), dtype=int),
        ]
    )

    return X_train, X_val, X_test, y_test