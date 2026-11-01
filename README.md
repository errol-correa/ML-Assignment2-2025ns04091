# Machine Learning Assignment - 2: Customer Churn Classification & Deployment

## a. Problem Statement
The objective of this project is to develop, evaluate, and deploy an end-to-end machine learning classification system to predict customer churn in the telecommunications industry. Using historical subscriber demographics, account information, and service usage patterns, the model identifies customers at high risk of leaving, enabling proactive customer retention strategies.

## b. Dataset Description
- **Dataset Source:** Telco Customer Churn (Kaggle public repository)
- **Total Instances:** 7,043 customer records
- **Total Features:** 20 predictor attributes and 1 binary target variable (`Churn`)
- **Target Variable:** Binary classification indicating whether the customer churned (`Yes` = 1, `No` = 0)
- **Key Features:**
  - *Demographics:* `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - *Account / Services:* `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - *Contract & Billing:* `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Preprocessing Pipeline:**
  - Removed arbitrary identifier (`customerID`).
  - Converted blank spaces in `TotalCharges` to numeric and imputed missing values with the median.
  - One-hot encoded all categorical features using `pd.get_dummies(..., drop_first=True)`.
  - Scaled numerical and encoded features using `StandardScaler`.
  - Preserved an 80/20 stratified train/test split.

## c. Repository and Live Application Links
- **GitHub Repository Link:** [Add your GitHub Repo URL here]
- **Live Streamlit App Link:** [Add your Streamlit Community Cloud URL here]

## d. Models Used & Evaluation Metrics Comparison

All 5 classification models were trained and evaluated on the exact same dataset split (80% train, 20% test).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8070** | **0.8416** | **0.6584** | 0.5668 | **0.6092** | **0.4843** |
| **Decision Tree** | 0.7942 | 0.8284 | 0.6296 | 0.5455 | 0.5845 | 0.4507 |
| **kNN** | 0.7601 | 0.7839 | 0.5511 | 0.5187 | 0.5344 | 0.3734 |
| **Naive Bayes** | 0.6558 | 0.8093 | 0.4269 | **0.8663** | 0.5719 | 0.3951 |
| **Support Vector Machine** | 0.7928 | 0.7961 | 0.6444 | 0.4893 | 0.5562 | 0.4312 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the strongest overall balanced performance with the highest Accuracy (0.8070), AUC (0.8416), Precision (0.6584), F1 Score (0.6092), and MCC (0.4843). The standardized continuous and one-hot encoded features mapped effectively to its linear decision boundary without severe overfitting. |
| **Decision Tree** | Yielded solid predictive power with an Accuracy of 0.7942 and AUC of 0.8284. Constraining the maximum depth prevented excessive variance and kept false positives controlled, achieving 0.6296 precision. |
| **kNN** | Delivered moderate results (Accuracy: 0.7601, AUC: 0.7839, MCC: 0.3734). Performance is somewhat dampened by the high dimensionality of one-hot encoded columns, which dilutes Euclidean distance metrics. |
| **Naive Bayes** | Captured the highest Recall (0.8663) across all models, effectively identifying churn cases. However, due to the conditional independence assumption among correlated billing and service features, it generated a large volume of false positives, dropping Precision to 0.4269 and Accuracy to 0.6558. |
| **Support Vector Machine** | Produced competitive Accuracy (0.7928) and Precision (0.6444) with an RBF kernel, but had lower Recall (0.4893) compared to Logistic Regression, resulting in an F1 score of 0.5562. |

### Overall Winner for Your Dataset
- **Winner:** **Logistic Regression**
- **Justification:** Logistic Regression outperformed all other models across 5 of the 6 evaluation metrics (Accuracy, AUC, Precision, F1 Score, and MCC). It provides the most balanced precision-recall trade-off and calibrated probability estimates for subscriber churn risk.