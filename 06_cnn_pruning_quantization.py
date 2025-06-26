# Install TensorFlow Model Optimization Toolkit for pruning
 !pip install -q tensorflow-model-optimization

import tensorflow_model_optimization as tfmot
import numpy as np  
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("TF MOT version:", tfmot.__version__)

# Define a simple CNN model architecture for binary classification (empty/occupied parking space)
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),  # Input image size 128x128 RGB
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # Conv layer 32 filters
        tf.keras.layers.MaxPooling2D((2, 2)),                   # Downsampling
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Conv layer 64 filters
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),                              # Flatten to 1D for Dense layers
        tf.keras.layers.Dense(128, activation='relu'),         # Fully connected hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')          # Binary output with sigmoid
    ])
    return model

# Instantiate model and load pretrained weights
model = build_model()
model.load_weights('/kaggle/input/model-with-calss-weights/best_model.weights.h5')

# Function to load TFRecord datasets and parse each example
def load_tfdataset(filename):
    def parse_example(example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.parse_tensor(parsed['image'], tf.float32)
        image = tf.reshape(image, (128, 128, 3))  # Reshape image tensor
        label = tf.cast(parsed['label'], tf.int32)  # Cast label to int32
        return image, label
    
    dataset = tf.data.TFRecordDataset(f'/kaggle/input/updated-prepro-data/{filename}')
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Load train, validation, and test datasets from TFRecords
raw_train = load_tfdataset('train.tfrecord')
raw_valid = load_tfdataset('valid.tfrecord')
raw_test = load_tfdataset('test.tfrecord')

# Prepare datasets with batching, shuffling, caching, and prefetching for performance
train_dataset = (
    raw_train
    .cache()
    .shuffle(1000)
    .batch(32)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

valid_dataset = (
    raw_valid
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    raw_test
    .cache()
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

# Count the number of examples in a dataset (used for steps calculation)
def count_examples(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

train_example_count = count_examples(raw_train)
valid_example_count = count_examples(raw_valid)

steps_per_epoch = train_example_count // 32
validation_steps = valid_example_count // 32

print(f"\nSteps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Count samples per class (binary: 0 and 1)
def count_classes(dataset, num_classes=2):
    class_counts = [0] * num_classes
    for _, label in dataset:
        class_counts[label.numpy()] += 1
    return class_counts

train_class_counts = count_classes(raw_train)

# Compute class weights to handle class imbalance during training
def compute_sklearn_class_weights(class_counts):
    labels = []
    for class_idx, count in enumerate(class_counts):
        labels.extend([class_idx] * count)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {i: w for i, w in zip(classes, weights)}

class_weight_dict = compute_sklearn_class_weights(train_class_counts)
print("\nClass weights:", class_weight_dict)

# Define pruning parameters - gradually prune from 0% to 30% sparsity over 1000 steps
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.3,
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning wrapper to the model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model with optimizer, loss, and evaluation metrics
pruned_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks to update pruning step and log summaries
pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='/tmp/pruning_logs')
]

print(pruned_model.summary())

# Train the pruned model
history = pruned_model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    callbacks=pruning_callbacks,
)

# Remove pruning wrappers to export the final pruned model
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Compile final model for evaluation
final_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall = final_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

# Plot training vs validation accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy: Training vs Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss: Training vs Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Confusion matrix on test data
y_pred = final_model.predict(test_dataset)
y_pred = np.round(y_pred).astype(int).flatten()
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

cm = confusion_matrix(y_true, y_pred)
disp = tf.keras.utils.plot_model(final_model, show_shapes=True)  # Optionally visualize model architecture
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty', 'Occupied'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Final Pruned Model)")
plt.show()

# Save the final pruned model
final_model.save('pruned_model.h5')

# Check file sizes to measure compression effects
original_model_path = '/kaggle/input/model-with-calss-weights/best_model.h5'
original_size = os.path.getsize(original_model_path) / (1024 * 1024)
pruned_size = os.path.getsize('pruned_model.h5') / (1024 * 1024)
print(f"Original model size: {original_size:.2f} MB")
print(f"Pruned model size: {pruned_size:.2f} MB")
print(f"Size reduction: {(1 - pruned_size / original_size) * 100:.2f}%")

# Load original model for TFLite conversion
original_model = tf.keras.models.load_model(original_model_path, compile=False)

# Convert original model to quantized TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

# Evaluate quantized model using TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare test images and labels for TFLite evaluation
test_images, test_labels = zip(*list(test_dataset))
test_images = np.vstack(test_images)
test_labels = np.hstack(test_labels)

predictions = []
for img in test_images:
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    predictions.append(output)

pred_labels = [1 if p > 0.5 else 0 for p in predictions]

# Calculate and print quantized model metrics
accuracy = accuracy_score(test_labels, pred_labels)
precision = precision_score(test_labels, pred_labels)
recall = recall_score(test_labels, pred_labels)
cm = confusion_matrix(test_labels, pred_labels)

print("\nQuantized Model Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:", cm)

# Convert pruned model to quantized TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_pruned_tflite_model = converter.convert()
with open('quantized_pruned_model.tflite', 'wb') as f:
    f.write(quantized_pruned_tflite_model)

# Evaluate quantized + pruned model similarly
interpreter = tf.lite.Interpreter(model_path="quantized_pruned_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

predictions = []
for img in test_images:
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    predictions.append(output)

pred_labels = [1 if p > 0.5 else 0 for p in predictions]

accuracy = accuracy_score(test_labels, pred_labels)
precision = precision_score(test_labels, pred_labels)
recall = recall_score(test_labels, pred_labels)
cm = confusion_matrix(test_labels, pred_labels)

print("\nQuantized and pruned Model Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:", cm)

# Compare model file sizes for all variants
original_size = os.path.getsize(original_model_path)
pruned_size = os.path.getsize('pruned_model.h5')
quantized_size = os.path.getsize('quantized_model.tflite')
pruned_quantized_size = os.path.getsize('quantized_pruned_model.tflite')

print("\n Model Size Comparison:\n")
print(f"Original Model: {original_size / (1024*1024):.2f} MB ({original_size / 1024:.2f} KB)")
print(f"Pruned Model: {pruned_size / (1024*1024):.2f} MB ({pruned_size / 1024:.2f} KB)")
print(f"Quantized Model: {quantized_size / (1024*1024):.2f} MB ({quantized_size / 1024:.2f} KB)")
print(f"Pruned + Quantized Model: {pruned_quantized_size / (1024*1024):.2f} MB ({pruned_quantized_size / 1024:.2f} KB)\n")

print(f"Original -> Pruned: {(1 - pruned_size / original_size) * 100:.2f}%")
print(f"Original -> Quantized: {(1 - quantized_size / original_size) * 100:.2f}%")
print(f"Original -> Pruned + Quantized: {(1 - pruned_quantized_size / original_size) * 100:.2f}%")
print(f"Pruned -> Pruned + Quantized: {(1 - pruned_quantized_size / pruned_size) * 100:.2f}%")
