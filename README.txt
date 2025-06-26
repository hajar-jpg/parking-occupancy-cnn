# Parking Occupancy Detection using CNNs and TFRecord Pipelines

This repository contains the full pipeline for detecting parking occupancy from images using 
Convolutional Neural Networks (CNNs), implemented in TensorFlow and trained on the PKLot dataset.
 The project includes data preprocessing, label generation, model training (with and without class balancing), 
 and model optimization via pruning and quantization.

---

## Repository Structure

| File                               | Description                                                                                    |
|------------------------------------|------------------------------------------------------------------------------------------------|
| `01_generate_train_labels.py`      | Generates binary labels for training data based on a threshold for occupied spaces.            |
| `02_generate_valid_test_labels.py` | Applies the same labeling logic to the validation and test datasets.                           |
| `03_prepare_tfrecords.py`          | Prepares and writes the image and label data into TFRecord format for efficient training.      |
| `04_train_cnn_balanced.py`         | Trains a CNN on a class-balanced version of the dataset using data augmentation.               |
| `05_train_cnn_class_weights.py`    | Trains a CNN using class weights to address imbalance (without augmentation).                  |
| `06_cnn_pruning_quantization.py`   | Applies model pruning and quantization to reduce size and optimize performance for deployment. |

---

## Dataset

We used the **PKLot** dataset for this project. The original dataset contains bounding box annotations for each parking space.
 For this binary classification task, we reformulated the problem at the image level:

- A label `1` is assigned if **≥ 80%** of the spaces are occupied.
- A label `0` is assigned otherwise.

###  TFRecord Files

The `.tfrecord` files (train, validation, and test) are **generated and stored on [Kaggle](https://www.kaggle.com/)** during data preprocessing. 
They are not included in this repository due to size constraints.

You can run the TFRecord generation scripts in a Kaggle Notebook to reproduce them:
- `train.tfrecord`
- `valid.tfrecord`
- `test.tfrecord`

---

##  Model Architecture

The CNN model consists of:
- Convolutional layers with ReLU and BatchNormalization
- MaxPooling and Dropout layers for regularization
- Dense layers with sigmoid activation for binary classification

---

##  Metrics Used

The following evaluation metrics are tracked:
- Accuracy
- Precision
- Recall
- AUC-ROC
- AUC-PR
- Confusion Matrix

---

##  Optimization Techniques

- **Class weights**: to handle data imbalance without changing sample counts
- **Data augmentation**: applied only to the minority class (class 1) in the balanced training version
- **Model pruning**: reduces model size by introducing sparsity
- **Post-training quantization**: converts model to TFLite for edge deployment

---

## Results Summary

Each version of the model is evaluated on the same test set using consistent metrics. Results are visualized with:
- Accuracy/Loss plots
- Confusion matrices
- Size comparisons for original, pruned, and quantized models

---

##  Requirements

tensorflow==2.15.0
tensorflow-model-optimization==0.7.5
scikit-learn==1.3.2
matplotlib==3.7.5
seaborn==0.12.2
numpy==1.24.3

