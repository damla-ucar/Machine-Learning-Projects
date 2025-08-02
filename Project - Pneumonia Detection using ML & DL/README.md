# 🫁📊 Pneumonia Classifier

A comparative study of machine learning and deep learning models to detect pneumonia from chest X-ray images.

---
Authors: Damla Uçar, Ahmet Sayan, Arda Erkan, Bahar Özkırlı, Eren Keskin

## 📂 Overview
This project implements and evaluates four different models for the binary classification of pneumonia:
- Convolutional Neural Network (CNN)
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

All models were tested on the same dataset, with experiments performed both on original data and data augmented through image transformations.

---

## ⚙️ Models & Methods

### 📌 Convolutional Neural Network (CNN)
- **Architecture:**
  - Multiple convolutional layers with Leaky ReLU activation
  - Dense layers with dropout to reduce overfitting
  - Sigmoid output for binary classification
- **Optimization:**
  - Adam optimizer (`learning_rate=0.001`)
  - Binary cross-entropy loss
  - Callbacks:
    - Early stopping
    - ReduceLROnPlateau
    - Model checkpointing
- **Hyperparameter tuning:**
  - Batch size
  - Number of filters, kernel sizes
  - Dropout rates

### 📌 Random Forest
- **Hyperparameters tuned:**
  - Number of estimators
  - Max depth
  - Min samples split and leaf
  - Max features
- Best results achieved by balancing model complexity and overfitting.

### 📌 Support Vector Machine (SVM)
- **Hyperparameters tuned:**
  - Kernel type (RBF, linear, polynomial, sigmoid)
  - Regularization parameter `C`
  - Gamma for kernel coefficient

### 📌 K-Nearest Neighbors (KNN)
- **Hyperparameters tuned:**
  - Number of neighbors (`k`)

Grid search and cross-validation were used for tuning to find the best performing parameters.

---

## 📊 Results

| Model        | Dataset       | Precision | Recall | Accuracy | F1 Score |
|-------------|--------------|-----------|--------|----------|---------|
| **CNN** | Augmented    | 0.838     | 0.996  | 0.866    | **0.910** |
| Random Forest | Augmented    | 0.830     | 0.983  | 0.851    | 0.900   |
| SVM         | Augmented    | 0.800     | 0.979  | 0.818    | 0.881   |
| KNN         | Augmented    | 0.787     | 0.997  | 0.814    | 0.879   |

- CNN achieved the highest F1-Score (≈91%) but suffered from higher overfitting.
- Random Forest achieved competitive performance (≈90% F1) and was more stable.
- Augmentation generally improved F1-scores, especially for more complex models.

---

## 🧪 Dataset
- **Source:** [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- Images labeled as “Pneumonia” or “Normal”.

---

## 🧰 Requirements
- Python 3.7+
- scikit-learn
- keras / tensorflow
- numpy
- matplotlib
- pandas

```bash
pip install -r requirements.txt
