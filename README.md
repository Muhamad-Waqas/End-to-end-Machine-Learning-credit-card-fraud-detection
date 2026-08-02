# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. The objective is to build a classification model capable of identifying fraudulent transactions while minimizing false alarms.

The project follows a complete Machine Learning workflow, including Exploratory Data Analysis (EDA), data preprocessing, feature engineering, handling class imbalance, model training, evaluation, and comparison.

---

## Dataset

This project uses the Credit Card Fraud Detection Dataset available on Kaggle.

### Dataset Source

https://www.kaggle.com/code/abdelazizelserty/credit-card-fraud-detection-dataset/input

### Dataset Characteristics

| Metric | Value |
|----------|----------|
| Total Transactions | 10,000 |
| Legitimate Transactions | 9,849 |
| Fraudulent Transactions | 151 |
| Fraud Ratio | ~1.5% |

### Challenge

The dataset is highly imbalanced, with fraudulent transactions representing only a small fraction of the total data. This makes fraud detection challenging because a model can achieve high accuracy while still failing to detect fraud effectively.

---

# Business Problem

Financial institutions process thousands of transactions daily, making it difficult to manually identify fraudulent activity.

The goal of this project is to build a machine learning model that can accurately identify fraudulent transactions while balancing:

- Fraud Detection Rate (Recall)
- False Fraud Alerts (Precision)

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Logistic Regression

---

# Project Workflow

```text
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Data Preprocessing
        ↓   
Model Training
        ↓
Class Imbalance Handling
        ↓
Model Evaluation
        ↓
Insights & Conclusion
```

---

# Stage 1: Exploratory Data Analysis & Preprocessing

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Missing Value Analysis
- Duplicate Record Analysis
- Target Class Distribution
- Numerical Feature Distribution
- Outlier Detection
- Correlation Analysis

### Key Finding

The target variable was highly imbalanced:

- Legitimate Transactions: 98.49%
- Fraudulent Transactions: 1.51%

This indicated that accuracy alone would not be a reliable evaluation metric.

---

## Feature Engineering & Preprocessing

### Log Transformation

Log1p transformation was applied to highly right-skewed numerical features:

- Amount
- Velocity Last 24 Hours

This helped reduce skewness and compress extreme values.

### One-Hot Encoding

Categorical variables were converted into numerical representations using One-Hot Encoding.

### Feature Scaling

StandardScaler was applied to numerical features:

- Amount
- Cardholder Age
- Transaction Hour
- Device Trust Score
- Velocity Last 24 Hours

Binary features were not scaled.

### Data Leakage Prevention

To prevent data leakage:

- Train-Test Split was performed before preprocessing.
- Encoders and Scalers were fitted only on training data.
- Test data was transformed using the fitted training parameters.
- SMOTE was applied only on the training set.

---

# Stage 2: Machine Learning Modeling

## Baseline Model

### Logistic Regression

A baseline Logistic Regression model was trained on the preprocessed dataset.

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 99.2% |
| Precision | 94% |
| Recall | 50% |
| F1-Score | 65% |

### Confusion Matrix

```text
[[1969    1]
 [  15   15]]
```

### Observation

The model achieved excellent accuracy and precision but failed to detect half of the fraudulent transactions.

---

# Handling Class Imbalance

## Approach 1: Class Weight

Logistic Regression was retrained using:

```python
class_weight='balanced'
```

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 92.95% |
| Precision | 18% |
| Recall | 100% |
| F1-Score | 30% |

### Confusion Matrix

```text
[[1829 141]
 [   0  30]]
```

### Observation

The model successfully detected all fraudulent transactions but generated a large number of false positives.

---

## Approach 2: SMOTE

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the training dataset.

### Results

| Metric | Value |
|----------|----------|
| Accuracy | 93.95% |
| Precision | 20% |
| Recall | 100% |
| F1-Score | 33% |

### Confusion Matrix

```text
[[1849 121]
 [   0  30]]
```

### Observation

SMOTE improved fraud detection performance compared to the baseline model but still produced a high number of false positive predictions.

---

# Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|----------|----------|----------|----------|----------|
| Logistic Regression | 99.2% | 94% | 50% | 65% |
| Logistic Regression + Class Weight | 92.95% | 18% | 100% | 30% |
| Logistic Regression + SMOTE | 93.95% | 20% | 100% | 33% |

---


## Model Comparison visually

![Model Comparison](./Images/Compare_result.jpg)

---

# Key Findings

- Accuracy alone can be misleading when working with imbalanced datasets.
- Precision and Recall provide a better understanding of fraud detection performance.
- The baseline model achieved the highest Precision and F1-Score.
- Class Weight and SMOTE improved Recall from 50% to 100%.
- Improving Recall introduced a significant increase in False Positives.
- Fraud detection requires balancing business objectives rather than optimizing a single metric.

---

# What I Learned

Through this project, I gained practical experience in:

- End-to-end Machine Learning workflows
- Exploratory Data Analysis
- Feature Engineering
- Handling Class Imbalance
- Preventing Data Leakage
- Model Evaluation using Precision, Recall, F1-Score, and Confusion Matrix
- Understanding Precision-Recall Tradeoffs
- Comparing multiple approaches for solving imbalanced classification problems

One of the most valuable lessons was learning that:

> A model is not better simply because it has higher accuracy. The best model is the one that aligns with the business objective and balances the right performance metrics.

---

# Conclusion

Logistic Regression provided a strong baseline model with high accuracy and precision. However, due to the severe class imbalance, it detected only 50% of fraudulent transactions.

Applying Class Weight and SMOTE successfully increased Recall to 100%, ensuring all fraud cases were detected. However, this improvement came at the cost of a significant increase in false positives and reduced Precision.

This project highlights the importance of evaluating machine learning models beyond accuracy and understanding the tradeoffs involved in fraud detection systems.


## Author

**Muhammad Waqas**

Machine Learning & Data Science Enthusiast

Building end-to-end machine learning projects and sharing the learning journey publicly.
