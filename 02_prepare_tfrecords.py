
"""
Prepare image datasets as TensorFlow TFRecord files.

For each train, validation, and test split, this script:
1. Loads image-level labels generated from COCO annotations.
2. Resizes RGB images to 128 x 128 pixels.
3. Converts pixel values to float32 and normalizes them to [0, 1].
4. Writes each image and its binary label to a TFRecord file.
5. Reads the TFRecord back and verifies counts, labels, shape,
   and pixel-value range.

Label definition:
    0: Less than 80% of annotated parking spaces are occupied.
    1: At least 80% of annotated parking spaces are occupied.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_DATASET_DIR = (
    "/kaggle/input/datasets/ammarnassanalhajali/pklot-dataset"
)
DEFAULT_LABEL_DIR = "/kaggle/working"
DEFAULT_OUTPUT_DIR = "/kaggle/working"

TARGET_SIZE = (128, 128)
SPLITS = ("train", "valid", "test")

FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


# ============================================================
# 2. SERIALIZE ONE IMAGE AND LABEL
# ============================================================

def serialize_example(image_array: np.ndarray, label: int) -> bytes:
    """Serialize a float32 image and its integer label."""

    image_tensor = tf.convert_to_tensor(
        image_array, dtype=tf.float32
    )
    serialized_image = tf.io.serialize_tensor(image_tensor)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[serialized_image.numpy()]
                    )
                ),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[int(label)]
                    )
                ),
            }
        )
    )

    return example.SerializeToString()


# ============================================================
# 3. CREATE A TFRECORD FILE FOR ONE SPLIT
# ============================================================

def create_tfrecord(
    dataset_dir: Path,
    label_dir: Path,
    output_dir: Path,
    split: str,
) -> tuple[int, Counter]:
    """Preprocess images and write them to a split-specific TFRecord."""

    split_dir = dataset_dir / split
    label_path = label_dir / f"{split}_labels_threshold.json"
    output_path = output_dir / f"{split}.tfrecord"

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {split_dir}"
        )

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Label JSON not found: {label_path}"
        )

    with label_path.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, dict):
        raise ValueError(
            f"Expected a filename-to-label dictionary in {label_path}"
        )

    invalid_labels = {
        label
        for label in labels.values()
        if type(label) is not int or label not in (0, 1)
    }

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in {split}: {invalid_labels}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    expected_label_counts = Counter(labels.values())

    # Process one image at a time to avoid loading the whole split
    # into memory.
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for filename, label in labels.items():
            image_path = split_dir / filename

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image referenced by labels is missing: {image_path}"
                )

            image = load_img(
                str(image_path),
                target_size=TARGET_SIZE,
                color_mode="rgb",
            )

            image_array = (
                img_to_array(image).astype(np.float32) / 255.0
            )

            if image_array.shape != (
                TARGET_SIZE[0], TARGET_SIZE[1], 3
            ):
                raise ValueError(
                    f"Unexpected image shape for {filename}: "
                    f"{image_array.shape}"
                )

            writer.write(serialize_example(image_array, label))
            written += 1

    # Every label entry must correspond to one written example.
    if written != len(labels):
        raise AssertionError(
            f"{split}: expected {len(labels)} examples, wrote {written}"
        )

    print(f"\n{split}: wrote {written} examples")
    print(f"Expected label distribution: {dict(expected_label_counts)}")
    print(f"Saved to: {output_path}")

    return written, expected_label_counts


# ============================================================
# 4. PARSE TFRECORD EXAMPLES
# ============================================================

def parse_tfrecord_fn(example):
    """Decode a serialized TFRecord example."""

    parsed = tf.io.parse_single_example(
        example, FEATURE_DESCRIPTION
    )

    image = tf.io.parse_tensor(
        parsed["image"], out_type=tf.float32
    )

    image = tf.ensure_shape(
        image, [TARGET_SIZE[0], TARGET_SIZE[1], 3]
    )

    label = tf.cast(parsed["label"], tf.int32)

    return image, label


# ============================================================
# 5. VERIFY THE SAVED TFRECORD FILES
# ============================================================

def verify_tfrecord(
    output_path: Path,
    split: str,
    expected_count: int,
    expected_label_counts: Counter,
) -> None:
    """Verify example count, label distribution, shape, and pixel range."""

    dataset = tf.data.TFRecordDataset(str(output_path))
    dataset = dataset.map(
        parse_tfrecord_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    example_count = 0
    actual_label_counts = Counter()

    for image, label in dataset:
        label_value = int(label.numpy())
        actual_label_counts[label_value] += 1
        example_count += 1

        if label_value not in (0, 1):
            raise AssertionError(
                f"Invalid label {label_value} found in {split}"
            )

        # Inspect the first decoded image in each split.
        if example_count == 1:
            image_array = image.numpy()

            assert image_array.shape == (
                TARGET_SIZE[0], TARGET_SIZE[1], 3
            ), f"Unexpected shape in {split}: {image_array.shape}"

            assert np.isfinite(image_array).all(), (
                f"Non-finite pixel values found in {split}"
            )

            assert image_array.min() >= 0.0, (
                f"Pixel value below 0 found in {split}"
            )

            assert image_array.max() <= 1.0, (
                f"Pixel value above 1 found in {split}"
            )

    if example_count != expected_count:
        raise AssertionError(
            f"{split}: expected {expected_count} examples, "
            f"read {example_count}"
        )

    if actual_label_counts != expected_label_counts:
        raise AssertionError(
            f"{split}: label distribution mismatch. "
            f"Expected {dict(expected_label_counts)}, "
            f"got {dict(actual_label_counts)}"
        )

    print(f"\n{split}: verification passed")
    print(f"Examples: {example_count}")
    print(f"Label distribution: {dict(actual_label_counts)}")


# ============================================================
# 6. DISPLAY TRAINING EXAMPLES
# ============================================================

def show_training_examples(output_dir: Path, number_of_samples=5):
    """Display decoded examples from the training TFRecord."""

    train_path = output_dir / "train.tfrecord"

    dataset = tf.data.TFRecordDataset(str(train_path))
    dataset = dataset.map(parse_tfrecord_fn)

    examples = list(dataset.take(number_of_samples))

    if not examples:
        print("No training examples available to display.")
        return

    fig, axes = plt.subplots(
        1, len(examples),
        figsize=(4 * len(examples), 4),
        squeeze=False,
    )

    for ax, (image, label) in zip(axes[0], examples):
        ax.imshow(image.numpy())
        ax.axis("off")

        label_text = (
            "At least 80% occupied"
            if int(label.numpy()) == 1
            else "Below 80% occupied"
        )
        ax.set_title(label_text, fontsize=9)

    fig.suptitle("Training examples decoded from TFRecord")
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. RUN THE PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess parking images and create TFRecords."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help="Root directory containing train, valid, and test images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(DEFAULT_LABEL_DIR),
        help="Directory containing the generated label JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for the generated TFRecord files.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip displaying training examples.",
    )

    args = parser.parse_args()

    expected_results = {}

    for split in SPLITS:
        written, label_counts = create_tfrecord(
            dataset_dir=args.dataset_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            split=split,
        )

        expected_results[split] = (written, label_counts)

    print("\nTFRecord verification")
    print("-" * 45)

    for split in SPLITS:
        expected_count, expected_label_counts = expected_results[split]

        verify_tfrecord(
            output_path=args.output_dir / f"{split}.tfrecord",
            split=split,
            expected_count=expected_count,
            expected_label_counts=expected_label_counts,
        )

    if not args.skip_visualization:
        show_training_examples(args.output_dir)


if __name__ == "__main__":
    main()
