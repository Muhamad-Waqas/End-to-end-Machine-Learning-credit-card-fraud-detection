Credit Card Fraud Detection
Project Overview
This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. The objective is to build a classification model capable of identifying fraudulent transactions while minimizing false alarms.
The dataset contains transaction-level information such as transaction amount, merchant category, transaction velocity, device trust score, and other fraud-related indicators.

Business Problem
Fraudulent transactions represent a small percentage of total transactions, making fraud detection a highly imbalanced classification problem.
A model that predicts every transaction as legitimate may achieve high accuracy while completely failing to detect fraud.
Therefore, the primary goal is not only achieving high accuracy but also maximizing fraud detection performance using metrics such as Recall, Precision, and F1-Score.



Dataset Summary
Metric	Value
Total Transactions	10,000
Legitimate Transactions	9,849
Fraudulent Transactions	151
Fraud Ratio	~1.5%
The dataset exhibits significant class imbalance, requiring specialized evaluation and balancing techniques.

Exploratory Data Analysis & Preprocessing

Exploratory Data Analysis
The following analyses were performed:
•	Missing value analysis
•	Duplicate detection
•	Target class distribution analysis
•	Feature distribution analysis
•	Outlier identification
•	Correlation analysis

Train Test Split 
To prevent data leakage, the dataset was split into training and testing sets before preprocessing. Scaling, encoding, and SMOTE were fitted using only the training data and then applied to the test set.

Data Preprocessing
The preprocessing pipeline included:
Log Transformation
Log1p transformation was applied to:
•	Amount
•	Velocity Last 24 Hours
This reduced right-skewness and compressed extreme values.

One-Hot Encoding
Categorical features were converted into numerical representations using One-Hot Encoding.

Feature Scaling
StandardScaler was applied to numerical features:
•	Amount
•	Velocity Last 24 Hours
•	Cardholder Age
•	Transaction Hour
•	Device Trust Score
Binary features were left unchanged.


 Machine Learning Modeling And Improving Model
Baseline Model
Logistic Regression
The first model trained was Logistic Regression without any imbalance handling techniques.

Results
Accuracy: 0.9395


Confusion Matrix :
 [[1849  121]
 [   0   30]]


Classification Report :
               precision    recall  f1-score   support

           0       1.00      0.94      0.97      1970
           1       0.20      1.00      0.33        30

    accuracy                           0.94      2000
   macro avg       0.60      0.97      0.65      2000
weighted avg       0.99      0.94      0.96      2000	


Observations
Although the model achieved excellent accuracy and precision, it detected only 50% of fraudulent transactions.
This highlights the limitations of relying solely on accuracy when evaluating imbalanced datasets.

Handling Class Imbalance
Approach 1: Class Weight
Logistic Regression was retrained using:
class_weight='balanced'

Results
Accuracy: 0.9295

Confusion Matrix: 
[[1829  141]
 [   0   30]]
Classfication Report:
              precision    recall  f1-score   support
          0       1.00      0.93      0.96      1970
           1       0.18      1.00      0.30        30
 accuracy                           0.93      2000
   macro avg       0.59      0.96      0.63      2000
weighted avg       0.99      0.93      0.95      2000

Observations
The model successfully detected all fraud cases but produced a large number of false positives, significantly reducing precision.

Approach 2: SMOTE
SMOTE was applied to the training data to generate synthetic fraud samples and balance class distribution.

Results
Accuracy: 0.9395

Confusion Matrix :
 [[1849  121]
 [   0   30]]

Classification Report :
               precision    recall  f1-score   support
           0       1.00      0.94      0.97      1970
           1       0.20      1.00      0.33        30
    accuracy                           0.94      2000
   macro avg       0.60      0.97      0.65      2000
weighted avg       0.99      0.94      0.96      2000

Observations
SMOTE improved fraud detection performance compared to the baseline model but introduced many false positive predictions.


Model Comparison
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	99.2%	94%	50%	65%
Logistic Regression + Class Weight	92.95%	18%	100%	30%
Logistic Regression + SMOTE	93.95%	20%	100%	33%


Key Learnings
This project provided several practical machine learning insights:
•	Accuracy is not sufficient for evaluating imbalanced classification problems.
•	Precision and Recall must be analyzed together.
•	Improving Recall often comes at the cost of Precision.
•	Class imbalance techniques should be validated through experimentation rather than assumed to improve model performance.
•	Confusion Matrix analysis provides deeper insight into model behavior than accuracy alone.

Conclusion
The baseline Logistic Regression model achieved the strongest balance between Precision and F1-Score but failed to detect half of the fraudulent transactions.
Class Weight and SMOTE successfully improved Recall to 100%, ensuring all fraud cases were detected. However, this improvement came with a substantial increase in false positives.
These experiments demonstrate the tradeoff between fraud detection performance and false alarm rates in highly imbalanced datasets.



