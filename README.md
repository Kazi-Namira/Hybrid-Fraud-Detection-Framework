# Hybrid Fraud Detection Framework

## Overview
This repository contains an **academic implementation** of a hybrid machine-learning framework for financial fraud detection. The project addresses the real-world challenge of **imbalanced datasets** using **SMOTE (Synthetic Minority Over-sampling Technique)** and leverages **XGBoost** for robust classification. 

The framework evaluates how imbalance handling and ensemble methods improve predictive performance in detecting fraudulent transactions.

---

## Dataset
> **Note:** The dataset is **not included** in this repository due to GitHub's 100 MB file size limit.

You can download the publicly available dataset from its original source:

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Place the downloaded CSV file in the project folder as `creditcard.csv`.

---

## Project Structure

Hybrid-Fraud-Detection-Framework/
│
├─ main.py # Main script: preprocessing, training, evaluation
├─ test.py # Testing script for model predictions
├─ requirements.txt # Python dependencies
├─ results_final.csv # Model predictions and metrics
├─ figure1_comprehensive_comparison.png
├─ figure2_confusion_matrix.png
├─ figure3_roc_curve.png (Extra)
├─ figure3_roc_curves.png
└─ README.md


- **main.py**: Preprocessing, SMOTE oversampling, training XGBoost, evaluation  
- **test.py**: Testing the trained model on sample data  
- **results_final.csv**: Predictions and performance metrics  
- **Figures**: Visualizations for model comparison, ROC curves, and confusion matrices  

---

## Methodology

1. **Data Preprocessing**
   - Handling missing values (if any)
   - Feature scaling and normalization
   - Train-test split

2. **Handling Imbalanced Data**
   - Applied **SMOTE** to oversample the minority class
   - Ensures balanced representation for model training

3. **Model Training**
   - **XGBoost Classifier** used for high accuracy on tabular data
   - Hyperparameter tuning for optimal performance

4. **Evaluation Metrics**
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC score

5. **Visualization**
   - Comparative analysis of models
   - ROC curves
   - Confusion matrices

---

## Requirements

Use Python 3.8+ and install dependencies:
pip install -r requirements.txt

---

## Key Libraries

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`, `seaborn`
- `imblearn` (for SMOTE)

---

## How to Run

1. Download the dataset (`creditcard.csv`) and place it in the root folder.  
2. Run the main training script:
- python main.py
  
3. Run the testing script:
- python test.py
  
Generated results and figures will be saved in the repository.

---

## Notes

The dataset is excluded due to GitHub size limits. Please download from the official source.

.gitignore is included to exclude CSVs, virtual environment files, and Python cache files.

---

## References

Liu, H., et al., “SMOTE: Synthetic Minority Over-sampling Technique”, Journal of Artificial Intelligence Research, 2002.

Chen, T., & Guestrin, C., “XGBoost: A Scalable Tree Boosting System”, KDD 2016.

Kaggle Credit Card Fraud Detection Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
