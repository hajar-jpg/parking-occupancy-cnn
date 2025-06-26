import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to load and parse TFRecord dataset
def load_tfdataset(filename):
    def parse_example(example_proto):
        # Define the expected features in the TFRecord
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        # Parse a single example
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        # Deserialize the image tensor
        image = tf.io.parse_tensor(parsed['image'], tf.float32)
        image = tf.reshape(image, (128, 128, 3))  # Reshape to expected image size
        label = tf.cast(parsed['label'], tf.int32)  # Cast label to int32 for training
        return image, label
    
    # Load the TFRecord file and apply parsing function in parallel
    dataset = tf.data.TFRecordDataset(f'/kaggle/input/updated-prepro-data/{filename}')
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Constants for dataset pipeline
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Load datasets for training, validation, and testing
raw_train = load_tfdataset('train.tfrecord')
raw_valid = load_tfdataset('valid.tfrecord')
raw_test = load_tfdataset('test.tfrecord')

# Helper function to count examples in a dataset
def count_examples(dataset):
    count = 0
    for _ in dataset:  # Iterate over dataset to count elements
        count += 1
    return count

# Count and print the number of examples in each dataset
train_example_count = count_examples(raw_train)
print(f"Examples in train data: {train_example_count}")

valid_example_count = count_examples(raw_valid)
print(f"Examples in valid data: {valid_example_count}")

test_example_count = count_examples(raw_test)
print(f"Examples in test data: {test_example_count}")

# Function to count number of samples per class in the dataset
def count_classes(dataset, num_classes=2):
    """Count the number of samples per class in a dataset"""
    class_counts = [0] * num_classes
    for _, label in dataset:
        class_counts[label.numpy()] += 1
    return class_counts

# Compute class distribution for train, validation, and test datasets
train_class_counts = count_classes(raw_train)
valid_class_counts = count_classes(raw_valid)
test_class_counts = count_classes(raw_test)

print("\nTrain data class distribution:")
for i, count in enumerate(train_class_counts):
    print(f"Class {i} samples: {count}")

print("\nValidation data class distribution:")
for i, count in enumerate(valid_class_counts):
    print(f"Class {i} samples: {count}")

print("\nTest data class distribution:")
for i, count in enumerate(test_class_counts):
    print(f"Class {i} samples: {count}")

# Compute class weights to address class imbalance using sklearn's balanced strategy
def compute_sklearn_class_weights(class_counts):
    # Create label array matching distribution of classes
    labels = []
    for class_idx, count in enumerate(class_counts):
        labels.extend([class_idx] * count)
    
    # Compute weights automatically balancing the classes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    # Return as a dictionary mapping class index to weight (for Keras training)
    return {i: w for i, w in zip(classes, weights)}

# Calculate class weights from training data
class_weight_dict = compute_sklearn_class_weights(train_class_counts)
print("\nClass weights:", class_weight_dict)

# Prepare datasets for training with batching, shuffling, caching and prefetching
train_dataset = (
    raw_train
    .cache()
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

valid_dataset = (
    raw_valid
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    raw_test
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
])

model.summary()

# Compile the model with AdamW optimizer and Binary Focal Crossentropy loss, including key metrics
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0002, weight_decay=1e-5, global_clipnorm=5.0),
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc_roc'),
        tf.keras.metrics.AUC(name='auc_pr')
    ]
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Callbacks to improve training and save best models
early_stopping = EarlyStopping(
    monitor='val_auc_pr',  # Monitor validation AUC-PR metric
    mode='max',
    patience=5,            # Stop training after 5 epochs with no improvement
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc_pr',
    mode='max',
    factor=0.5,           # Reduce learning rate by half
    patience=3,           # If no improvement for 3 epochs
    min_lr=1e-6,
    verbose=1
)

# Paths for saving model and weights
full_model_path = "best_model.h5"
weights_path = "best_model.weights.h5"

# Save the full model (architecture + weights)
model_checkpoint_full = ModelCheckpoint(
    filepath=full_model_path,
    monitor='val_auc_pr',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Save weights only
model_checkpoint_weights = ModelCheckpoint(
    filepath=weights_path,
    monitor='val_auc_pr',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=20,
    class_weight=class_weight_dict,  # Use class weights to balance training
    callbacks=[early_stopping, reduce_lr, model_checkpoint_weights, model_checkpoint_full],
    verbose=2
)

# Load the best saved full model for evaluation
best_model = tf.keras.models.load_model('best_model.h5', compile=True)

# Evaluate the best model on test dataset
test_loss, test_accuracy, test_precision, test_recall, test_auc, test_auc_pr = best_model.evaluate(test_dataset)

print(f"\nBest Model Evaluation on Test Set:")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test AUC ROC: {test_auc}")
print(f"Test AUC PR: {test_auc_pr}")

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy: Training vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss: Training vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predict on test dataset
y_pred_probs = best_model.predict(test_dataset)
# Convert probabilities to binary labels (0 or 1)
y_pred = np.round(y_pred_probs).astype(int).flatten()

# Extract true labels from test dataset
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix using sklearn's utility
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty', 'Occupied'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
