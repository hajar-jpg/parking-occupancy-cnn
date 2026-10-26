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

    if not filepath.is_file():
        raise FileNotFoundError(
            f"TFRecord file not found: {filepath}"
        )

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_example(example_proto):
        """Decode one serialized image and its binary label."""

        parsed = tf.io.parse_single_example(
            example_proto,
            feature_description,
        )

        image = tf.io.parse_tensor(
            parsed["image"],
            out_type=tf.float32,
        )
        image = tf.ensure_shape(image, IMAGE_SHAPE)

        label = tf.cast(parsed["label"], tf.int32)

        # Ensure labels follow the expected binary classification scheme.
        tf.debugging.assert_greater_equal(label, 0)
        tf.debugging.assert_less_equal(label, 1)

        return image, label

    dataset = tf.data.TFRecordDataset(str(filepath))

    dataset = dataset.map(
        parse_example,
        num_parallel_calls=AUTOTUNE,
    )

    return dataset

# ---------------------------------------------------------------------
# Load dataset splits
# ---------------------------------------------------------------------

# Load the preprocessed training, validation, and test datasets.
train_dataset = load_tfrecord_dataset("train.tfrecord")
valid_dataset = load_tfrecord_dataset("valid.tfrecord")
test_dataset = load_tfrecord_dataset("test.tfrecord")

print("All three TFRecord datasets loaded successfully.")

# ---------------------------------------------------------------------
# Dataset inspection and validation
# ---------------------------------------------------------------------

def inspect_dataset(dataset, name):
    """Count examples and validate image shapes and binary labels."""

    total = 0
    class_counts = {0: 0, 1: 0}
    first_batch_shape = None

    for images, labels in dataset.batch(BATCH_SIZE):

        if first_batch_shape is None:
            first_batch_shape = tuple(images.shape)

        # Validate the image dimensions.
        if tuple(images.shape[1:]) != IMAGE_SHAPE:
            raise ValueError(
                f"{name} contains images with unexpected shape: "
                f"{images.shape[1:]}; expected {IMAGE_SHAPE}."
            )

        labels_np = labels.numpy()

        # Ensure labels belong to the expected binary classes.
        invalid_labels = set(np.unique(labels_np)) - {0, 1}

        if invalid_labels:
            raise ValueError(
                f"{name} contains invalid labels: {invalid_labels}"
            )

        total += len(labels_np)
        class_counts[0] += int(np.sum(labels_np == 0))
        class_counts[1] += int(np.sum(labels_np == 1))

    if total == 0:
        raise ValueError(f"{name} dataset is empty.")

    # Both classes should be represented in each dataset split.
    if any(count == 0 for count in class_counts.values()):
        raise ValueError(
            f"{name} must contain examples from both classes. "
