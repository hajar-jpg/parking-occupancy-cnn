"""Train a CNN for image-level parking occupancy classification using class weights.

The script loads preprocessed TFRecord datasets, applies class weights to
address class imbalance during training, trains a convolutional neural
network, and evaluates the best checkpoint on the held-out test set.

The original training examples are retained without data augmentation
or oversampling. Class weights give greater importance to the minority
class during training.

Labels:
    0: Less than 80% of annotated parking spaces are occupied.
    1: At least 80% of annotated parking spaces are occupied.

Expected TFRecord files:
    train.tfrecord
    valid.tfrecord
    test.tfrecord

Outputs:
    class_weights_best_model.keras
    class_weights_confusion_matrix.png
    class_weights_training_curves.png
    class_weights_test_results.json
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryFocalCrossentropy

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------
# Configuration and reproducibility
# ---------------------------------------------------------------------

SEED = 42
keras.utils.set_random_seed(SEED)

# Request deterministic operations where supported.
try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError) as exc:
    print(f"Could not enable deterministic operations: {exc}")

# Dataset and output directories.
# Environment variables allow the paths to be changed outside Kaggle.


DATASET_DIR = Path(
    os.environ.get(
        "PARKING_DATASET_DIR",
        "/kaggle/input/datasets/hajar20/my-data",
    )
)
OUTPUT_DIR = Path(
    os.environ.get("PARKING_OUTPUT_DIR", "/kaggle/working")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model and training configuration.
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE


val_pr_auc_history = history.history.get("val_pr_auc", [])

if not val_pr_auc_history:
    raise ValueError(
        "Validation PR-AUC is missing from the training history."
    )

# Identify the epoch with the highest validation PR-AUC.
best_epoch = int(np.argmax(val_pr_auc_history) + 1)
best_validation_pr_auc = float(max(val_pr_auc_history))

report = {
    "experiment": "CNN with class weighting",
    "selection_metric": "validation PR-AUC",
    "best_epoch": best_epoch,
    "epochs_completed": len(history.history["loss"]),

    "dataset": {
        "train_examples": train_stats["total"],
        "validation_examples": valid_stats["total"],
        "test_examples": test_stats["total"],
        "train_class_counts": {
            str(k): int(v)
            for k, v in train_stats["class_counts"].items()
        },
        "validation_class_counts": {
            str(k): int(v)
            for k, v in valid_stats["class_counts"].items()
        },
        "test_class_counts": {
            str(k): int(v)
            for k, v in test_stats["class_counts"].items()
        },
    },

    "class_weights": {
        str(k): float(v)
        for k, v in class_weights.items()
    },

    "configuration": {
        "image_shape": list(IMAGE_SHAPE),
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
        "seed": SEED,
        "optimizer": "AdamW",
        "initial_learning_rate": 0.0002,
        "weight_decay": 1e-5,
        "global_clipnorm": 5.0,
        "loss": "BinaryFocalCrossentropy",
        "focal_gamma": 2.0,
        "prediction_threshold": 0.5,
        "augmentation": False,
        "oversampling": False,
        "checkpoint_selection_metric": "val_pr_auc",
        "early_stopping_patience": 5,
        "reduce_lr_patience": 3,
        "minimum_learning_rate": 1e-6,
    },

    "best_validation_pr_auc": best_validation_pr_auc,

    "test_metrics_keras": {
        k: float(v)
        for k, v in keras_test_results.items()
    },

    "test_metrics_sklearn": sklearn_test_results,
    "confusion_matrix": cm.tolist(),

    "class_labels": {
        "0": "Less than 80% occupancy",
        "1": "At least 80% occupancy",
    },

    "artifacts": {
        "model": BEST_MODEL_PATH.name,
        "confusion_matrix": CONFUSION_MATRIX_PATH.name,
        "training_curves": TRAINING_CURVES_PATH.name,
    },
}

# Save the report as formatted JSON.
REPORT_PATH = OUTPUT_DIR / "class_weights_test_results.json"

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Report saved: {REPORT_PATH}")
print(f"Best epoch: {best_epoch}")
print(f"Best validation PR-AUC: {best_validation_pr_auc:.6f}")

# Verify that all expected output artifacts exist.
expected_artifacts = [
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    TRAINING_CURVES_PATH,
    REPORT_PATH,
]

print("\nArtifact verification:")
for path in expected_artifacts:
    status = "OK" if path.is_file() else "MISSING"
    print(f"{status:7} {path.name}")

missing_artifacts = [
    path.name for path in expected_artifacts if not path.is_file()
]

if missing_artifacts:
    raise FileNotFoundError(
        f"Expected artifacts are missing: {missing_artifacts}"
    )

