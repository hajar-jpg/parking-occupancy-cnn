# Parking Occupancy Classification Using CNNs and TFRecord Pipelines

This project explores image-level parking occupancy classification using Convolutional Neural Networks (CNNs) and TensorFlow, based on the PKLot dataset.

The project pipeline includes annotation-based label generation, image preprocessing, TFRecord serialization, CNN training, model evaluation, and model optimization for potential edge deployment.

The implementation and results are being documented incrementally to ensure reproducibility and technical accuracy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset and Label Generation](#dataset-and-label-generation)
- [Dataset Statistics](#dataset-statistics)
- [Preprocessing and TFRecords](#preprocessing-and-tfrecords)
- [Repository Structure](#repository-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Optimization](#model-optimization)
- [Requirements](#requirements)
- [Reproducibility](#reproducibility)

---

## Project Overview

The goal is to classify parking-lot images into two categories based on the proportion of annotated parking spaces that are occupied.

Unlike object detection, this project performs **image-level binary classification**. It predicts the occupancy category of an entire image rather than detecting or classifying each individual parking space.

The main stages are:

1. Generate binary labels from COCO-format annotations.
2. Resize and normalize images.
3. Serialize images and labels into TensorFlow TFRecord files.
4. Train and evaluate CNN models.
5. Investigate class-imbalance strategies.
6. Explore model pruning and quantization.

---

## Dataset and Label Generation

The project uses the PKLot parking-lot dataset, which provides images and parking-space annotations in COCO format.

An image-level occupancy label is calculated using the proportion of annotated parking spaces marked as occupied.

Let:

\[
r = \frac{N_{\text{occupied}}}{N_{\text{annotated spaces}}}
\]

The binary labeling rule is:

- **Class 1 — High occupancy:** \(r \geq 0.80\)
- **Class 0 — Lower occupancy:** \(r < 0.80\)

Class 0 does not necessarily mean that the parking lot is empty. It means that fewer than 80% of its annotated parking spaces are occupied.

Images without annotations are skipped because their occupancy proportion cannot be calculated from the available annotations.

---

## Dataset Statistics

The label-generation pipeline produced the following verified counts:

| Split | Total images | Class 0 | Class 1 |
|---|---:|---:|---:|
| Training | 8,502 | 5,416 | 3,086 |
| Validation | 2,424 | 1,500 | 924 |
| Test | 1,216 | 748 | 468 |
| **Total** | **12,142** | **7,664** | **4,478** |

A further 274 images were skipped because they had no annotations: 189 from training, 59 from validation, and 26 from testing.

These counts describe the generated labels, not the classification performance of a trained model.

---

## Preprocessing and TFRecords

Images are preprocessed and serialized into TensorFlow TFRecord files for use during model training.

The preprocessing pipeline includes:

- Loading RGB images from the dataset.
- Resizing images to **128 × 128 pixels**.
- Converting image values to `float32`.
- Normalizing pixel values to the range **[0, 1]**.
- Serializing image tensors and integer labels into `tf.train.Example` records.
- Parsing the generated TFRecords and checking example counts, class distributions, image shapes, and pixel-value ranges.

The expected output files are:

- `train.tfrecord`
- `valid.tfrecord`
- `test.tfrecord`

These generated files are stored in the Kaggle working directory and are not included in the GitHub repository because of their size.

The cleaned preprocessing script must pass its verification checks before its revised implementation is considered fully validated.

---

## Repository Structure

The current core pipeline is organized as follows:

| File | Description |
|---|---|
| `01_generate_labels.py` | Generates image-level binary labels from COCO annotations using the occupancy threshold. |
| `02_prepare_tfrecords.py` | Preprocesses images, writes TFRecord files, and verifies the serialized datasets. |

Additional scripts for CNN training, evaluation, class balancing, pruning, and quantization will be documented here as their implementations are finalized and verified.

---

## Model Training and Evaluation

The next stage is to establish a baseline CNN for binary classification.

The model architecture, training configuration, and evaluation procedure will be documented after the corresponding implementation has been reviewed and executed successfully.

Planned evaluation metrics include:

- Accuracy
- Precision
- Recall
- Area Under the ROC Curve (AUC-ROC)
- Area Under the Precision-Recall Curve (AUC-PR)
- Confusion matrix

Training and validation results will be reported separately from final test-set results.

---

## Model Optimization

After establishing the baseline, the project will investigate methods for addressing class imbalance and reducing model size.

Planned experiments include:

- **Class weighting:** assigning different loss weights to classes without changing the number of training samples.
- **Data augmentation:** augmenting the minority class to create a more balanced training dataset.
- **Model pruning:** introducing sparsity into model weights to investigate model-size and performance trade-offs.
- **Post-training quantization:** converting the model to TensorFlow Lite and evaluating the effects of reduced numerical precision.

The final documentation will include measured comparisons of predictive performance and model size where the experiments have been successfully reproduced.

---

## Requirements

The Python dependencies will be documented once the final training and optimization scripts have been reviewed and the required package versions confirmed.

The current pipeline uses TensorFlow, NumPy, and image-processing utilities from TensorFlow/Keras. Additional dependencies will be listed when required by the finalized implementation.

---

## Reproducibility

Data preparation is designed to run in a Kaggle Notebook with access to the PKLot dataset.

To reproduce the workflow:

1. Make the PKLot dataset available in the Kaggle environment.
2. Run `01_generate_labels.py` to generate the labels.
3. Run `02_prepare_tfrecords.py` to preprocess the images and create the TFRecords.
4. Verify the generated dataset counts and label distributions.
5. Run the finalized model-training and evaluation scripts once they are available.

Dataset paths and output directories may need to be adjusted for a different environment.

