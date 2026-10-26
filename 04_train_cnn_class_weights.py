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


# ---------------------------------------------------------------------
# TFRecord loading and parsing
# ---------------------------------------------------------------------

def load_tfrecord_dataset(filename):
    """Load and parse a TFRecord file containing images and labels."""

    filepath = DATASET_DIR / filename
