import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# 1. Directory and Data Path
os.makedirs("model", exist_ok=True)
data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not os.path.exists(data_path):
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(data_path)

# 2. Data Preprocessing
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# TotalCharges contains blank spaces for new users; convert to numeric and impute median
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Target encoding: Yes -> 1, No -> 0
target_col = "Churn"
y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
X_raw = df.drop(columns=[target_col])

# Categorical vs Numeric columns
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()

# One-hot encode categorical features
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
feature_names = X_encoded.columns.tolist()

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save artifacts for Streamlit app
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")
joblib.dump(categorical_cols, "model/categorical_cols.pkl")
joblib.dump(numeric_cols, "model/numeric_cols.pkl")

# Save test split for Streamlit uploader (contains raw original columns + target)
test_data_export = df.iloc[X_test.index].copy()
test_data_export.to_csv("test_data.csv", index=False)
print("Saved clean test_data.csv successfully.")

# 5. Model Definitions
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=42),
}

# 6. Train, Evaluate and Save Models
results = []
print("\nTraining models and calculating evaluation metrics...")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics computation
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    # Save model binary
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join("model", fname))
    print(f"Saved: model/{fname}")

# 7. Print Output Table for Assignment README
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("EVALUATION METRICS TABLE (For README.md and PDF submission):")
print("=" * 70)
print(results_df.to_markdown(index=False))
print("=" * 70)