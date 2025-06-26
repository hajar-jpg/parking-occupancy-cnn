import json
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# === Phase 1: Copy and Prepare the JSON Labels File ===

# Paths for the base dataset and annotations
base_dir = "/kaggle/input/pklot-dataset/train"  # Source folder (read-only)
json_filename = "_annotations.coco.json"        # Annotation file name
json_read_path = os.path.join(base_dir, json_filename)         # Original path
json_write_path = os.path.join("/kaggle/working/", json_filename)  # Writable path

# Copy the annotation file to a writable location
if not os.path.exists(json_write_path):
    shutil.copy(json_read_path, json_write_path)
    print("✅ JSON file copied to /kaggle/working/")

# === Phase 2: Generate Image-Level Labels Based on Threshold ===

# General setup for threshold-based labeling
base_dir = "/kaggle/input/pklot-dataset"
split_name = "train"  # Can be changed to "valid" or "test"
split_dir = os.path.join(base_dir, split_name)
json_path = "/kaggle/working/_annotations.coco.json"

THRESHOLD = 0.8  # Label as 'occupied' if 80% or more spaces are occupied

# Load COCO-formatted JSON annotations
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Create a dictionary to store the image-level binary labels
image_labels = {}

# Iterate over each image in the dataset
for image_info in coco_data['images']:
    image_id = image_info['id']
    image_filename = image_info['file_name']

    # Collect all annotations belonging to this image
    image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

    total_spaces = len(image_annotations)
    if total_spaces == 0:
        continue  # Skip if the image has no labeled parking spaces

    # Count the number of occupied parking spaces (category_id = 2)
    occupied_count = sum(1 for anno in image_annotations if anno['category_id'] == 2)

    # Compute occupancy rate
    occupancy_rate = occupied_count / total_spaces

    # Assign label based on threshold: 1 = Occupied, 0 = Empty
    image_label = 1 if occupancy_rate >= THRESHOLD else 0
    image_labels[image_filename] = image_label

# Save new labels as JSON
updated_json_path = os.path.join('/kaggle/working/', 'train_labels_threshold.json')
with open(updated_json_path, 'w') as f:
    json.dump(image_labels, f, indent=4)

print(f"✅ Updated image labels with threshold saved to '{updated_json_path}'.")

# === Phase 3: Print Class Distribution ===

# Count how many images are labeled 0 (empty) and 1 (occupied)
label_counts = {0: 0, 1: 0}
for label in image_labels.values():
    label_counts[label] += 1

print(f"Empty parking lots (0): {label_counts[0]}")
print(f"Occupied parking lots (1): {label_counts[1]}")

# === Phase 4: Visual Inspection of Labeled Images ===

# Load labeled data
with open("/kaggle/working/train_labels_threshold.json", 'r') as f:
    image_labels = json.load(f)

# Function to display a single image and its label
def show_image_and_label(image_filename, label):
    image_path = os.path.join(split_dir, image_filename)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(f"Label: {'Empty' if label == 0 else 'Occupied'}", fontsize=10)

# Show 5 random examples
random_images = random.sample(list(image_labels.items()), 5)
plt.figure(figsize=(12, 10))
for i, (image_filename, label) in enumerate(random_images):
    plt.subplot(1, 5, i + 1)
    show_image_and_label(image_filename, label)
plt.show()

# === Phase 5: Show Sample Images from Each Class ===

# Separate filenames by label
empty_images = [filename for filename, label in image_labels.items() if label == 0]
occupied_images = [filename for filename, label in image_labels.items() if label == 1]

# Display 5 "Empty" images
plt.figure(figsize=(15, 12))
for i, image_filename in enumerate(empty_images[:5]):
    plt.subplot(1, 5, i + 1)
    show_image_and_label(image_filename, 0)
plt.show()

# Display 5 "Occupied" images
plt.figure(figsize=(15, 12))
for i, image_filename in enumerate(occupied_images[:5]):
    plt.subplot(1, 5, i + 1)
    show_image_and_label(image_filename, 1)
plt.show()
