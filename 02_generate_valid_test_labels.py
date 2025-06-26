# Import necessary libraries
import json
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# ----------------------------
# VALIDATION LABELS GENERATION
# ----------------------------

# Define the dataset folder and annotation file name
base_dir = "/kaggle/input/pklot-dataset/valid"
json_filename = "_annotations.coco.json"
json_read_path = os.path.join(base_dir, json_filename)

# Set new filename and write path for the copied JSON
new_json_filename = "valid_annotations.coco.json"
json_write_path = os.path.join("/kaggle/working/", new_json_filename)

# Copy JSON file to writable directory if not already copied
if not os.path.exists(json_write_path):
    shutil.copy(json_read_path, json_write_path)
    print(f"✅ JSON file copied to /kaggle/working/ as: {new_json_filename}")

# Set labeling threshold
THRESHOLD = 0.8  # 80% or more occupied → label = 1 (Occupied)

# Load the COCO-formatted annotation file
json_path = json_write_path
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Dictionary to store image-level binary labels
image_labels = {}

# Loop through each image and assign label based on threshold
for image_info in coco_data['images']:
    image_id = image_info['id']
    image_filename = image_info['file_name']
    
    # Get annotations related to this image
    image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]
    
    total_spaces = len(image_annotations)
    if total_spaces == 0:
        continue  # Skip images without labeled parking spaces

    # Count the number of occupied spaces
    occupied_count = sum(1 for anno in image_annotations if anno['category_id'] == 2)

    # Compute occupancy rate
    occupancy_rate = occupied_count / total_spaces

    # Assign label: 1 = occupied, 0 = empty
    image_labels[image_filename] = 1 if occupancy_rate >= THRESHOLD else 0

# Save labels to a JSON file
updated_json_path = os.path.join('/kaggle/working/', 'valid_labels_threshold.json')
with open(updated_json_path, 'w') as f:
    json.dump(image_labels, f, indent=4)

print(f"✅ Validation labels saved to: '{updated_json_path}'")

# Count and print how many images are labeled empty vs occupied
label_counts = {0: 0, 1: 0}
for label in image_labels.values():
    label_counts[label] += 1
print(f"Empty: {label_counts[0]} | Occupied: {label_counts[1]}")

# ------------------------
# TEST LABELS GENERATION
# ------------------------

# Repeat same process for the test dataset
base_dir = "/kaggle/input/pklot-dataset/test"
json_filename = "_annotations.coco.json"
json_read_path = os.path.join(base_dir, json_filename)
new_json_filename = "test_annotations.coco.json"
json_write_path = os.path.join("/kaggle/working/", new_json_filename)

if not os.path.exists(json_write_path):
    shutil.copy(json_read_path, json_write_path)
    print(f"✅ JSON file copied to /kaggle/working/ as: {new_json_filename}")

json_path = json_write_path
with open(json_path, 'r') as f:
    coco_data = json.load(f)

image_labels = {}

for image_info in coco_data['images']:
    image_id = image_info['id']
    image_filename = image_info['file_name']
    
    image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]
    
    total_spaces = len(image_annotations)
    if total_spaces == 0:
        continue

    occupied_count = sum(1 for anno in image_annotations if anno['category_id'] == 2)
    occupancy_rate = occupied_count / total_spaces
    image_labels[image_filename] = 1 if occupancy_rate >= THRESHOLD else 0

updated_json_path = os.path.join('/kaggle/working/', 'test_labels_threshold.json')
with open(updated_json_path, 'w') as f:
    json.dump(image_labels, f, indent=4)

print(f"✅ Test labels saved to: '{updated_json_path}'")

label_counts = {0: 0, 1: 0}
for label in image_labels.values():
    label_counts[label] += 1
print(f"Empty: {label_counts[0]} | Occupied: {label_counts[1]}")

# -------------------------------------
# DISPLAY SAMPLE IMAGES: VALIDATION SET
# -------------------------------------

# Load labels and show 5 random labeled images from the validation set
base_dir = "/kaggle/input/pklot-dataset"
split_name = "valid"
split_dir = os.path.join(base_dir, split_name)
image_labels_path = "/kaggle/working/valid_labels_threshold.json"

with open(image_labels_path, 'r') as f:
    image_labels = json.load(f)

def show_image_and_label(image_filename, label):
    image_path = os.path.join(split_dir, image_filename)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Label: {'Empty' if label == 0 else 'Occupied'}", fontsize=10)

random_images = random.sample(list(image_labels.items()), 5)
plt.figure(figsize=(12, 10))
for i, (image_filename, label) in enumerate(random_images):
    plt.subplot(1, 5, i + 1)
    show_image_and_label(image_filename, label)
plt.show()

# ---------------------------------
# DISPLAY SAMPLE IMAGES: TEST SET
# ---------------------------------

# Same process for test dataset
split_name = "test"
split_dir = os.path.join(base_dir, split_name)
image_labels_path = "/kaggle/working/test_labels_threshold.json"

with open(image_labels_path, 'r') as f:
    image_labels = json.load(f)

random_images = random.sample(list(image_labels.items()), 5)
plt.figure(figsize=(12, 10))
for i, (image_filename, label) in enumerate(random_images):
    plt.subplot(1, 5, i + 1)
    show_image_and_label(image_filename, label)
plt.show()
