import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction & Model Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Classification & Evaluation Dashboard")
st.markdown(
    "Interactive evaluation dashboard for 5 pre-trained machine learning classifiers. "
    "You can switch between models, or upload a custom test dataset (`test_data.csv`) using the sidebar."
)

# Sidebar: Model and File Controls
st.sidebar.header("Navigation & Configuration")

model_choice = st.sidebar.selectbox(
    "Choose ML Model to Evaluate:",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Support Vector Machine",
    ),
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"],
    help="Upload test_data.csv containing test instances (1 mark feature requirement).",
)

# Cache preprocessing artifacts and models
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join("model", "scaler.pkl"))
    feature_names = joblib.load(os.path.join("model", "feature_names.pkl"))
    categorical_cols = joblib.load(os.path.join("model", "categorical_cols.pkl"))
    numeric_cols = joblib.load(os.path.join("model", "numeric_cols.pkl"))
    return scaler, feature_names, categorical_cols, numeric_cols

@st.cache_resource
def load_model(name):
    filename = name.lower().replace(" ", "_") + ".pkl"
    return joblib.load(os.path.join("model", filename))

# Determine dataset source: uploaded file takes priority, otherwise use default test_data.csv
df_test = None
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded CSV file.")
elif os.path.exists("test_data.csv"):
    df_test = pd.read_csv("test_data.csv")
    st.sidebar.info("Using default repo `test_data.csv`.")
else:
    st.error("No test data found. Please upload `test_data.csv`.")

# Main Evaluation Logic
if df_test is not None:
    try:
        scaler, feature_names, categorical_cols, numeric_cols = load_artifacts()
        model = load_model(model_choice)

        st.subheader("1. Test Data Preview")
        st.write(f"Loaded test dataset: **{df_test.shape[0]} rows**, **{df_test.shape[1]} columns**")
        st.dataframe(df_test.head(5), use_container_width=True)

        target_col = "Churn"
        if target_col not in df_test.columns:
            st.error(f"Error: Target column '{target_col}' not found in the dataset.")
            st.stop()

        # Clean customerID if present
        if "customerID" in df_test.columns:
            df_test_features = df_test.drop(columns=["customerID"])
        else:
            df_test_features = df_test.copy()

        # Handle TotalCharges numeric conversion & missing values
        if "TotalCharges" in df_test_features.columns:
            df_test_features["TotalCharges"] = pd.to_numeric(df_test_features["TotalCharges"], errors="coerce")
            df_test_features["TotalCharges"] = df_test_features["TotalCharges"].fillna(df_test_features["TotalCharges"].median())

        # Ground truth
        y_true = df_test_features[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
        X_raw = df_test_features.drop(columns=[target_col])

        # One-hot encode categorical features and align columns to training set
        X_encoded = pd.get_dummies(X_raw, columns=[col for col in categorical_cols if col in X_raw.columns], drop_first=True)
        for col in feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[feature_names]

        # Scale features
        X_scaled = scaler.transform(X_encoded)

        # Generate Predictions
        y_pred = model.predict(X_scaled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_scaled)[:, 1]
            auc_val = roc_auc_score(y_true, y_proba)
        else:
            auc_val = 0.0

        # Calculate Metrics
        acc_val = accuracy_score(y_true, y_pred)
        prec_val = precision_score(y_true, y_pred, zero_division=0)
        rec_val = recall_score(y_true, y_pred, zero_division=0)
        f1_val = f1_score(y_true, y_pred, zero_division=0)
        mcc_val = matthews_corrcoef(y_true, y_pred)

        st.subheader(f"2. Evaluation Metrics: {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc_val:.4f}")
        col2.metric("AUC Score", f"{auc_val:.4f}")
        col3.metric("Precision", f"{prec_val:.4f}")
        col4.metric("Recall", f"{rec_val:.4f}")
        col5.metric("F1 Score", f"{f1_val:.4f}")
        col6.metric("MCC Score", f"{mcc_val:.4f}")

        st.markdown("---")

        # Visualizations: Confusion Matrix & Classification Report
        st.subheader("3. Model Diagnostics & Visualizations")
        viz_col1, viz_col2 = st.columns([1, 1])

        with viz_col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Churn (0)", "Churn (1)"],
                yticklabels=["No Churn (0)", "Churn (1)"],
                cbar=False,
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with viz_col2:
            st.markdown("#### Classification Report")
            report_dict = classification_report(
                y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error during model evaluation: {e}")