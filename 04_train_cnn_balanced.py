# Import necessary libraries
from keras.losses import BinaryFocalCrossentropy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Function to load and parse TFRecord datasets
def load_tfdataset(filename):
    def parse_example(example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.parse_tensor(parsed['image'], tf.float32)
        image = tf.reshape(image, (128, 128, 3))  # Ensure image shape is correct
        label = tf.cast(parsed['label'], tf.int32)
        return image, label

    # Load dataset and apply parsing logic
    dataset = tf.data.TFRecordDataset(f'/kaggle/input/updated-prepro-data/{filename}')
    return dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

# Function to augment only class 1 images to balance the dataset
def augment_minority_class(dataset, target_majority_count=5416):
    def augment_image(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label

    class_0_ds = dataset.filter(lambda img, lbl: lbl == 0)
    class_1_ds = dataset.filter(lambda img, lbl: lbl == 1)

    class_1_count = 3086  # You can also count this dynamically
    extra_needed = target_majority_count - class_1_count

    augmented_class_1 = class_1_ds.repeat().map(augment_image).take(extra_needed)
    balanced_class_1 = class_1_ds.concatenate(augmented_class_1)

    # Merge both classes and shuffle the result
    balanced_dataset = class_0_ds.concatenate(balanced_class_1)
    return balanced_dataset.shuffle(10000, reshuffle_each_iteration=True)

# Basic training configuration
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Load the raw datasets
raw_train = load_tfdataset('train.tfrecord')
raw_valid = load_tfdataset('valid.tfrecord')
raw_test = load_tfdataset('test.tfrecord')

# Balance the training dataset using augmentation
balanced_train_dataset = augment_minority_class(raw_train)

# Function to count number of examples
def count_examples(dataset):
    return sum(1 for _ in dataset)

# Print dataset sizes
print("Examples in train data:", count_examples(raw_train))
print("Examples in balanced train data:", count_examples(balanced_train_dataset))
print("Examples in valid data:", count_examples(raw_valid))
print("Examples in test data:", count_examples(raw_test))

# Function to count samples per class
def count_classes(dataset, num_classes=2):
    class_counts = [0] * num_classes
    for _, label in dataset:
        class_counts[label.numpy()] += 1
    return class_counts

# Display class distributions
print("\nTrain class distribution:", count_classes(raw_train))
print("Balanced train class distribution:", count_classes(balanced_train_dataset))
print("Validation class distribution:", count_classes(raw_valid))
print("Test class distribution:", count_classes(raw_test))

# Prepare datasets with batching and prefetching
balanced_traindata = balanced_train_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE).shuffle(1000)
valid_dataset = raw_valid.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_dataset = raw_test.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Define a CNN model
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
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.summary()

# Compile the model with focal loss to handle class imbalance
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0002, weight_decay=1e-5, global_clipnorm=5.0),
    loss=BinaryFocalCrossentropy(gamma=2.0),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc_roc'),
        tf.keras.metrics.AUC(name='auc_pr')  # Precision-Recall AUC
    ]
)

# Define file paths for saving the model
full_model_path = "best_balanced_model.h5"
weights_path = "best_balanced_model.weights.h5"

# Set up callbacks
early_stopping = EarlyStopping(
    monitor='val_auc_pr', mode='max', patience=5,
    restore_best_weights=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc_pr', mode='max', factor=0.5,
    patience=3, min_lr=1e-6, verbose=1
)

model_checkpoint_full = ModelCheckpoint(
    filepath=full_model_path, monitor='val_auc_pr',
    mode='max', save_best_only=True, save_weights_only=False, verbose=1
)

model_checkpoint_weights = ModelCheckpoint(
    filepath=weights_path, monitor='val_auc_pr',
    mode='max', save_best_only=True, save_weights_only=True, verbose=1
)

# Train the model
history = model.fit(
    balanced_traindata,
    validation_data=valid_dataset,
    epochs=20,
    callbacks=[early_stopping, reduce_lr, model_checkpoint_weights, model_checkpoint_full],
    verbose=2
)

# Load the best model
best_balanced_model = load_model("best_balanced_model.h5", compile=True)

# Evaluate on test set
test_loss, test_accuracy, test_auc, test_auc_1, test_precision, test_recall = best_balanced_model.evaluate(test_dataset)

print(f"\nTest Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test AUC ROC: {test_auc}")
print(f"Test AUC PR: {test_auc_1}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss: Training vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Make predictions on test dataset
y_pred_probs = best_balanced_model.predict(test_dataset)
y_pred = np.round(y_pred_probs).astype(int).flatten()

# Extract true labels from test dataset
y_true = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0).astype(int)

# Compute and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty', 'Occupied'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
