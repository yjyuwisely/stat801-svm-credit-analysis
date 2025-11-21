# STAT801 – Creditworthiness Classification  
Course: STAT801 – Statistical Methods for Data Science (Enhanced)
Author: Yeongjin Yu  
Date: 2025

This project applies multiple classification algorithms to predict creditworthiness using the `creditworthiness.csv` dataset.  
Methods include LDA, QDA (with PCA preprocessing), Naive Bayes, KNN, and Support Vector Machines (linear, RBF, and tuned).  
All analysis is implemented in R as part of STAT801 coursework.

---

## 🔍 Objectives

- Build and compare statistical learning models for credit classification  
- Evaluate model performance using confusion matrices and accuracy  
- Apply PCA for dimensionality reduction (QDA)  
- Tune SVM hyperparameters to identify the best-performing model  
- Predict the credit rating of 538 unclassified cases

---

## 📂 Files Included

- `credit_classification.R` – Full R script containing all model implementations  
- `creditworthiness.csv` – Dataset provided for the assignment  
- `Predicted_CreditRatings_538.csv` – Output containing predictions for unclassified cases  
- `README.md` – This summary file

---

## Methods & Models

### 1. **Linear Discriminant Analysis (LDA)**  
- Baseline model for comparison  
- Uses full dataset (labelled cases only)

### 2. **QDA with PCA**  
- PCA reduces dimensionality to top 10 components  
- Recommended approach from ISLR for high-dimensional QDA

### 3. **Naive Bayes (e1071)**  
- Assumes conditional independence  
- Fast and effective baseline classifier

### 4. **K-Nearest Neighbours (KNN)**  
- Standardised numerical predictors  
- k = 5 chosen for demonstration

### 5. **Support Vector Machines (SVM)**  
- Linear kernel  
- RBF kernel with different cost & gamma  
- Full hyperparameter tuning with `tune()`  
- Best-performing model used for final predictions

---

## Model Performance Summary

Each model was trained on the labelled portion of the dataset (n = 1,962) and evaluated on a 30% test split.  
Test accuracies were:

- **SVM (RBF, tuned): 0.584**
- **LDA: 0.577**
- QDA (with PCA): 0.501  
- KNN (k = 5): 0.477  
- Naive Bayes: 0.409  

The figure below shows SVM cross-validated accuracy across combinations of cost (C) and gamma (γ):

![SVM Tuning Heatmap](<https://github.com/user-attachments/assets/21043b83-3a68-4fe1-913a-8a46c3d7f853>)

A 2D demonstration of the SVM decision boundary is also included for intuition (subsetting to the first two predictors):

![SVM Decision Boundary](<https://github.com/user-attachments/assets/e686525b-4943-4293-b371-d00eac74d522>)

---

## Prediction of 538 Unclassified Cases

After model selection, the best SVM model (RBF kernel, **C = 1**, **gamma = 0.01**) was used to predict the credit ratings of **538 previously unlabelled cases**.

The predictions resulted in:

- **28** cases → Credit Rating **1 (A)**
- **89** cases → Credit Rating **2 (B)**
- **421** cases → Credit Rating **3 (C)**

The output file is included as:

- `Predicted_CreditRatings_538.csv`

---

