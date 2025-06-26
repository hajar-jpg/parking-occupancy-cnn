import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# -----------------------------
# Define dataset base directory
# -----------------------------
base_dir = "/kaggle/input/pklot-dataset"

# Paths to JSON label files for train, valid, and test splits
json_paths = {
    "train": "/kaggle/input/cnn-labels/train_labels_threshold.json",
    "valid": "/kaggle/input/cnn-valid-and-test-labels/valid_labels_threshold.json",
    "test": "/kaggle/input/cnn-valid-and-test-labels/test_labels_threshold.json",
}

# Image target size for resizing
target_size = (128, 128)

# -------------------------------------------------
# Function to load image paths and labels and preprocess
# -------------------------------------------------
def load_and_preprocess_data(split_name, target_size=(128, 128), subset_size=None):
    split_dir = os.path.join(base_dir, split_name)  # Dataset image directory
    json_path = json_paths.get(split_name, None)  # Corresponding label JSON file

    if not json_path or not os.path.exists(json_path):
        raise FileNotFoundError(f"Label file not found for {split_name}: {json_path}")

    # Load JSON labels: dict {image_filename: label}
    with open(json_path, "r") as f:
        data = json.load(f)

    images, labels = [], []

    # Collect image file paths and labels
    for img_filename, label in data.items():
        image_path = os.path.join(split_dir, img_filename)
        images.append(image_path)
        labels.append(label)  # 0 = empty, 1 = occupied

    # Optionally select only a subset (for quicker tests)
    if subset_size is not None:
        images, labels = images[:subset_size], labels[:subset_size]

    # Helper function to load and normalize a single image
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=target_size)  # Load and resize
        return img_to_array(img) / 255.0  # Normalize pixel values to [0,1]

    # Load and preprocess all images into a numpy array
    X_data = np.array([preprocess_image(img) for img in images])
    y_data = np.array(labels)

    return X_data, y_data

# -----------------------------------
# Function to write dataset into TFRecord format
# -----------------------------------
def write_tfrecord(X_data, y_data, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in zip(X_data, y_data):
            # Convert image to Tensor and serialize it
            image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
            image_serialized = tf.io.serialize_tensor(image_tensor)

            # Create feature dictionary
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_serialized.numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)]))
            }

            # Create Example message and serialize to string
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f"TFRecord written successfully to {filename}")

# ------------------------------
# Load and preprocess all splits
# ------------------------------
X_train, y_train = load_and_preprocess_data('train', target_size=target_size)
X_valid, y_valid = load_and_preprocess_data('valid', target_size=target_size)
X_test, y_test = load_and_preprocess_data('test', target_size=target_size)

# Write datasets to TFRecord files
write_tfrecord(X_train, y_train, 'train.tfrecord')
write_tfrecord(X_valid, y_valid, 'valid.tfrecord')
write_tfrecord(X_test, y_test, 'test.tfrecord')

# --------------------------------
# Function to parse TFRecord examples
# --------------------------------
def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.parse_tensor(parsed_example['image'], out_type=tf.float32)
    label = parsed_example['label']
    return image, label

# ----------------------------
# Function to count examples in a TFRecord dataset
# ----------------------------
def count_examples(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

# ----------------------------------
# Load TFRecord datasets and count examples
# ----------------------------------

for split in ['train', 'valid', 'test']:
    tfrecord_path = f"/kaggle/working/{split}.tfrecord"
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(parse_tfrecord_fn)
    count = count_examples(dataset)
    print(f"Examples in {split} data: {count}")

# -------------------------
# Display label distribution
# -------------------------
for y_data, split in zip([y_train, y_valid, y_test], ['Train', 'Valid', 'Test']):
    unique, counts = np.unique(y_data, return_counts=True)
    print(f"{split} Labels Distribution:", dict(zip(unique, counts)))

# -------------------------
# Display first 5 images from train TFRecord
# -------------------------
dataset = tf.data.TFRecordDataset(['/kaggle/working/train.tfrecord'])
dataset = dataset.map(parse_tfrecord_fn)

plt.figure(figsize=(15, 5))
for i, (image, label) in enumerate(dataset.take(5)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(image.numpy())  # Convert tensor to numpy array for display
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")

plt.show()
