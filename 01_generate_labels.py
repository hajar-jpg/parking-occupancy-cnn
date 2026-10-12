
"""
Generate image-level parking occupancy labels from COCO annotations.

Label definition:
    0: Less than 80% of annotated parking spaces are occupied.
    1: At least 80% of annotated parking spaces are occupied.

Images without annotations or with missing image files are skipped.

The script processes the existing train, validation, and test splits.
It does not create a new split or classify individual parking spaces.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_DATASET_DIR = (
    "/kaggle/input/datasets/ammarnassanalhajali/pklot-dataset"
)
DEFAULT_OUTPUT_DIR = "/kaggle/working"

SPLITS = ("train", "valid", "test")
OCCUPANCY_THRESHOLD = 0.80
OCCUPIED_CATEGORY_ID = 2
RANDOM_SEED = 42


# ============================================================
# 2. GENERATE IMAGE-LEVEL LABELS
# ============================================================

def generate_labels(dataset_dir: Path, output_dir: Path, split: str) -> dict:
    """Generate threshold-based occupancy labels for one dataset split."""

    split_dir = dataset_dir / split
    annotation_path = split_dir / "_annotations.coco.json"

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {split_dir}")

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"COCO annotation file not found: {annotation_path}"
        )

    # Load the COCO annotation file.
    with annotation_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    # Verify the category mapping before interpreting the labels.
    categories = {
        category["id"]: category["name"]
        for category in coco_data.get("categories", [])
    }

    print(f"\n{'=' * 50}")
    print(f"Processing split: {split}")
    print(f"Categories: {categories}")

    if OCCUPIED_CATEGORY_ID not in categories:
        raise ValueError(
            f"Occupied category ID {OCCUPIED_CATEGORY_ID} "
            f"was not found in {annotation_path}."
        )

    occupied_category_name = str(
        categories[OCCUPIED_CATEGORY_ID]
    ).strip().lower()

    if "occupied" not in occupied_category_name:
        raise ValueError(
            f"Category ID {OCCUPIED_CATEGORY_ID} maps to "
            f"{categories[OCCUPIED_CATEGORY_ID]!r}, not an "
            "occupied-space category. Check the dataset mapping."
        )

    # Group annotations by image ID once for efficient lookup.
    annotations_by_image = defaultdict(list)

    for annotation in coco_data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(
            annotation
        )

    image_labels = {}
    skipped_no_annotations = 0
    skipped_missing_images = 0

    for image_info in coco_data["images"]:
        image_id = image_info["id"]
        image_filename = image_info["file_name"]
        image_path = split_dir / image_filename

        # Skip images whose files are unavailable.
        if not image_path.is_file():
            skipped_missing_images += 1
            continue

        image_annotations = annotations_by_image.get(image_id, [])

        # Images without annotated parking spaces cannot be labeled.
        if not image_annotations:
            skipped_no_annotations += 1
            continue

        total_spaces = len(image_annotations)

        occupied_spaces = sum(
            annotation["category_id"] == OCCUPIED_CATEGORY_ID
            for annotation in image_annotations
        )

        occupancy_rate = occupied_spaces / total_spaces

        # Convert the image-level occupancy rate into a binary label.
        image_labels[image_filename] = int(
            occupancy_rate >= OCCUPANCY_THRESHOLD
        )

    # Save the generated labels as a filename-to-label JSON mapping.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}_labels_threshold.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(image_labels, file, indent=4)

    # Report the number of examples in each class.
    label_counts = {
        label: sum(value == label for value in image_labels.values())
        for label in (0, 1)
    }

    print(f"Images labeled: {len(image_labels)}")
    print(f"Label 0 (<80% occupied): {label_counts[0]}")
    print(f"Label 1 (>=80% occupied): {label_counts[1]}")
    print(f"Skipped: no annotations = {skipped_no_annotations}")
    print(f"Skipped: missing images = {skipped_missing_images}")
    print(f"Saved labels to: {output_path}")

    return image_labels


# ============================================================
# 3. VISUALIZE RANDOM LABELED EXAMPLES
# ============================================================

def show_samples(
    dataset_dir: Path,
    split: str,
    labels: dict,
    number_of_samples: int = 5,
) -> None:
    """Display random images with their generated occupancy labels."""

    if not labels:
        print(f"No labeled images available for {split}.")
        return

    split_dir = dataset_dir / split

    sample_items = random.sample(
        list(labels.items()),
        min(number_of_samples, len(labels)),
    )

    fig, axes = plt.subplots(
        1,
        len(sample_items),
        figsize=(4 * len(sample_items), 4),
        squeeze=False,
    )

    for ax, (filename, label) in zip(axes[0], sample_items):
        image_path = split_dir / filename
        image = mpimg.imread(image_path)

        ax.imshow(image)
        ax.axis("off")

        label_text = (
            "At least 80% occupied"
            if label == 1
            else "Below 80% occupied"
        )

        ax.set_title(label_text, fontsize=9)

    fig.suptitle(f"Random labeled examples — {split}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. RUN THE COMPLETE LABEL-GENERATION PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate image-level parking occupancy labels "
            "from COCO annotations."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help="Root directory containing train, valid, and test.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for the generated label JSON files.",
    )

    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    all_labels = {}

    for split in SPLITS:
        all_labels[split] = generate_labels(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            split=split,
        )

    # Print the combined class distribution.
    print("\nLABEL GENERATION SUMMARY")
    print("-" * 60)

    total_images = 0

    for split in SPLITS:
        labels = all_labels[split]

        count_0 = sum(label == 0 for label in labels.values())
        count_1 = sum(label == 1 for label in labels.values())

        total_images += len(labels)

        print(
            f"{split:>5}: total={len(labels):>6}, "
            f"label 0={count_0:>6}, label 1={count_1:>6}"
        )

    print("-" * 60)
    print(f"Total labeled images: {total_images}")

    # Display examples from each split for a visual sanity check.
    for split in SPLITS:
        show_samples(
            dataset_dir=args.dataset_dir,
            split=split,
            labels=all_labels[split],
        )


if __name__ == "__main__":
    main()
