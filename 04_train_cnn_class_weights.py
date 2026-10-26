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
            f"Found counts: {class_counts}"
        )

    print(f"\n{name} dataset")
    print(f"  First batch shape: {first_batch_shape}")
    print(f"  Total examples: {total}")
    print(f"  Class 0 (<80% occupied): {class_counts[0]}")
    print(f"  Class 1 (>=80% occupied): {class_counts[1]}")

    return {
        "total": total,
        "class_counts": class_counts,
    }


# Inspect each original dataset split.
train_stats = inspect_dataset(train_dataset, "Training")
valid_stats = inspect_dataset(valid_dataset, "Validation")
test_stats = inspect_dataset(test_dataset, "Test")

# ---------------------------------------------------------------------
# Calculate class weights from the original training distribution
# ---------------------------------------------------------------------

class_counts = train_stats["class_counts"]
total_train_examples = train_stats["total"]
num_classes = 2

# Each class receives a weight inversely proportional to its frequency.
# Formula: weight = total_examples / (num_classes * class_count)
if any(class_counts[label] == 0 for label in (0, 1)):
    raise ValueError(
        f"Both classes must be present. Counts: {class_counts}"
    )

class_weights = {
    label: total_train_examples / (num_classes * count)
    for label, count in class_counts.items()
}

print("Class weights:")
for label, weight in class_weights.items():
    print(f"  Class {label}: {weight:.4f}")

print(f"\nOriginal training examples: {total_train_examples}")

# ---------------------------------------------------------------------
# Prepare datasets for training and evaluation
# ---------------------------------------------------------------------

def count_batches(total_examples):
    """Calculate batches per epoch, including a partial final batch."""
    return (total_examples + BATCH_SIZE - 1) // BATCH_SIZE


# Calculate expected batch counts from the verified dataset sizes.
train_batches = count_batches(train_stats["total"])
valid_batches = count_batches(valid_stats["total"])
test_batches = count_batches(test_stats["total"])

# Shuffle only the training data.
# Validation and test data retain their original order.
train_data = (
    train_dataset
    .shuffle(
        buffer_size=train_stats["total"],
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .apply(
        tf.data.experimental.assert_cardinality(train_batches)
    )
    .prefetch(AUTOTUNE)
)

valid_data = (
    valid_dataset
    .batch(BATCH_SIZE, drop_remainder=False)
    .apply(
        tf.data.experimental.assert_cardinality(valid_batches)
    )
    .prefetch(AUTOTUNE)
)

test_data = (
    test_dataset
    .batch(BATCH_SIZE, drop_remainder=False)
    .apply(
        tf.data.experimental.assert_cardinality(test_batches)
    )
    .prefetch(AUTOTUNE)
)

# Verify that TensorFlow reports the expected number of batches.
actual_cardinalities = {
    "Training": int(
        tf.data.experimental.cardinality(train_data).numpy()
    ),
    "Validation": int(
        tf.data.experimental.cardinality(valid_data).numpy()
    ),
    "Test": int(
        tf.data.experimental.cardinality(test_data).numpy()
    ),
}

expected_cardinalities = {
    "Training": train_batches,
    "Validation": valid_batches,
    "Test": test_batches,
}

print("Dataset batch counts:")

for name, actual in actual_cardinalities.items():
    expected = expected_cardinalities[name]
    print(f"  {name}: {actual} batches")

    if actual != expected:
        raise ValueError(
            f"{name} cardinality mismatch: "
            f"expected {expected}, got {actual}"
        )

# ---------------------------------------------------------------------
# Define the convolutional neural network (CNN)
# ---------------------------------------------------------------------

model = keras.Sequential(
    [
        # Input: RGB image with dimensions 128 x 128.
        keras.Input(shape=IMAGE_SHAPE),

        # First convolutional block: extract low-level image features.
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # Second convolutional block: learn more complex visual features.
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # Classification head: map extracted features to a binary prediction.
        layers.Flatten(),
        layers.Dense(128, activation="relu"),

        # Output probability for class 1 (at least 80% occupancy).
        layers.Dense(1, activation="sigmoid"),
    ],
    name="parking_occupancy_cnn_class_weights",
)

# Display the architecture, output shapes, and parameter counts.
model.summary()

# ---------------------------------------------------------------------
# Compile the CNN
# ---------------------------------------------------------------------

model.compile(
    # AdamW combines adaptive gradient updates with weight decay.
    optimizer=keras.optimizers.AdamW(
        learning_rate=0.0002,
        weight_decay=1e-5,
        global_clipnorm=5.0,
    ),

    # Focal loss gives more emphasis to difficult-to-classify examples.
    loss=BinaryFocalCrossentropy(gamma=2.0),

    # Track classification performance and ranking quality.
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="roc_auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

print("Model compiled successfully.")
print(f"Optimizer: {type(model.optimizer).__name__}")
print(f"Loss: {type(model.loss).__name__}")

# ---------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------

# Save the model checkpoint with the highest validation PR-AUC.
BEST_MODEL_PATH = OUTPUT_DIR / "class_weights_best_model.keras"

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    # Reduce the learning rate when validation PR-AUC stops improving.
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),

    # Stop training if validation PR-AUC does not improve,
    # and restore the weights from the best epoch.
    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
]

print("Training callbacks configured.")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

# ---------------------------------------------------------------------
# Train the CNN using class weights
# ---------------------------------------------------------------------

# Train on the original training examples.
# Class weights increase the contribution of the minority class
# to the training loss without duplicating or removing examples.
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

print("\nTraining finished.")
print(f"Best model saved: {BEST_MODEL_PATH.is_file()}")
print(f"Epochs completed: {len(history.history['loss'])}")

# ---------------------------------------------------------------------
# Load the best checkpoint and evaluate on the untouched test set
# ---------------------------------------------------------------------

if not BEST_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Best model checkpoint not found: {BEST_MODEL_PATH}"
    )

# Load the checkpoint selected using validation PR-AUC.
best_model = keras.models.load_model(BEST_MODEL_PATH)

# Evaluate the saved model using Keras metrics.
keras_test_results = best_model.evaluate(
    test_data,
    return_dict=True,
    verbose=1,
)

# Collect the ground-truth labels from the test set.
y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_data
]).astype(np.int32)

# Generate predicted probabilities for class 1.
y_prob = best_model.predict(
    test_data,
    verbose=1,
).reshape(-1)

# Validate the predictions before calculating metrics.
if len(y_true) != len(y_prob):
    raise ValueError(
        f"Mismatch: {len(y_true)} labels but "
        f"{len(y_prob)} predictions."
    )

if len(y_true) != test_stats["total"]:
    raise ValueError(
        f"Expected {test_stats['total']} test examples, "
        f"but collected {len(y_true)} labels."
    )

if not np.isfinite(y_prob).all():
    raise ValueError("Predictions contain NaN or infinite values.")

if not np.isin(y_true, [0, 1]).all():
    raise ValueError("Test labels contain values other than 0 and 1.")

if len(np.unique(y_true)) != 2:
    raise ValueError("ROC-AUC requires both classes in the test set.")

# Convert probabilities to binary predictions using threshold 0.5.
y_pred = (y_prob >= 0.5).astype(np.int32)

# Calculate independent scikit-learn evaluation metrics.
sklearn_test_results = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_true, y_prob)),
    "average_precision": float(
        average_precision_score(y_true, y_prob)
    ),
}

print("\nKeras test metrics:")
for name, value in keras_test_results.items():
    print(f"{name}: {value:.6f}")

print("\nIndependent scikit-learn test metrics:")
for name, value in sklearn_test_results.items():
    print(f"{name}: {value:.6f}")

print(f"\nTest examples: {len(y_true)}")
print(f"Predictions generated: {len(y_prob)}")
print(f"Best checkpoint: {BEST_MODEL_PATH.is_file()}")

# ---------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------

# Rows represent true labels; columns represent predicted labels.
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

if int(cm.sum()) != len(y_true):
    raise ValueError(
        "Confusion matrix total does not match the number of test examples."
    )

print("Confusion matrix (rows = true labels, columns = predictions):")
print(cm)

print("\nClass definitions:")
print("Class 0: Less than 80% occupancy (<80%)")
print("Class 1: At least 80% occupancy (>=80%)")

# Report the four types of classification outcomes.
true_class_0, false_class_1, false_class_0, true_class_1 = cm.ravel()

print("\nClassification counts:")
print(f"  Correct class 0 predictions (TN): {true_class_0}")
print(f"  Class 0 predicted as class 1 (FP): {false_class_1}")
print(f"  Class 1 predicted as class 0 (FN): {false_class_0}")
print(f"  Correct class 1 predictions (TP): {true_class_1}")
print(f"  Total test examples: {cm.sum()}")

display_labels = [
    "Below 80% occupancy",
    "At least 80% occupancy",
]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels,
)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)
ax.set_title("Class-Weighted CNN — Test Confusion Matrix")

plt.tight_layout()

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "class_weights_confusion_matrix.png"
)

fig.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")

# ---------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------

history_metrics = [
    ("loss", "Loss"),
    ("accuracy", "Accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

# Find the epoch with the highest validation PR-AUC.
val_pr_auc = history.history.get("val_pr_auc", [])
best_epoch = (
    int(np.argmax(val_pr_auc)) + 1
    if val_pr_auc
    else None
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (metric, title_text) in zip(axes, history_metrics):
    train_values = history.history.get(metric, [])
    val_values = history.history.get(f"val_{metric}", [])

    if train_values:
        ax.plot(
            range(1, len(train_values) + 1),
            train_values,
            label="Training",
        )

    if val_values:
        ax.plot(
            range(1, len(val_values) + 1),
            val_values,
            label="Validation",
        )

    # Mark the checkpoint-selection epoch on the PR-AUC plot.
    if metric == "pr_auc" and best_epoch is not None:
        ax.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best epoch ({best_epoch})",
        )

    ax.set_title(title_text)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title_text)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Class-Weighted CNN — Training History",
    fontsize=14,
)
fig.tight_layout()

TRAINING_CURVES_PATH = (
    OUTPUT_DIR / "class_weights_training_curves.png"
)

fig.savefig(
    TRAINING_CURVES_PATH,
    dpi=200,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

print(f"Saved training curves to: {TRAINING_CURVES_PATH}")

# ---------------------------------------------------------------------
# Save experiment results and configuration
# ---------------------------------------------------------------------

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
