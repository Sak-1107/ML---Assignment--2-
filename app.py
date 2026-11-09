
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
    """
    This application evaluates five trained Machine Learning
    classification models for predicting customer churn.

    Upload customer data, select a model, and view predictions
    and performance metrics.
    """
)

st.divider()


# =====================================================
# 2. MODEL CONFIGURATION
# =====================================================

model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# =====================================================
# 3. MODEL LOADING FUNCTION
# =====================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# =====================================================
# 4. DATA UPLOAD
# =====================================================

st.header("1. Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Overview")

        col1, col2 = st.columns(2)

        col1.metric("Total Records", data.shape[0])
        col2.metric("Total Columns", data.shape[1])

        st.dataframe(data.head(), use_container_width=True)

        # =====================================================
        # 5. IDENTIFY TARGET COLUMN
        # =====================================================

        target_column = "churn"

        if target_column in data.columns:

            actual_target = data[target_column].copy()

            X = data.drop(columns=[target_column])

            has_target = True

        else:

            X = data.copy()

            actual_target = None

            has_target = False

        # =====================================================
        # 6. MODEL SELECTION
        # =====================================================

        st.divider()

        st.header("2. Select Classification Model")

        selected_model = st.selectbox(
            "Choose a trained model",
            list(model_paths.keys())
        )

        # =====================================================
        # 7. LOAD SELECTED MODEL
        # =====================================================

        model_path = model_paths[selected_model]

        if not os.path.exists(model_path):

            st.error(
                f"Model file not found: {model_path}"
            )

            st.stop()

        model = load_model(model_path)

        # =====================================================
        # 8. PREDICTION
        # =====================================================

        st.divider()

        st.header("3. Customer Churn Predictions")

        # Validate feature columns
        expected_features = list(
            model.named_steps["preprocessing"].feature_names_in_
        )

        missing_columns = [
            col for col in expected_features
            if col not in X.columns
        ]

        extra_columns = [
            col for col in X.columns
            if col not in expected_features
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

            st.stop()

        if extra_columns:

            st.warning(
                f"Extra columns will be ignored: {extra_columns}"
            )

        # Keep only training features in original order
        X = X[expected_features]

        # Generate predictions
        y_pred = model.predict(X)

        # Generate churn probabilities
        y_prob = model.predict_proba(X)[:, 1]

        # Convert predictions to readable labels
        prediction_labels = pd.Series(y_pred).map(
            {0: "no", 1: "yes"}
        )

        # Create output dataframe
        prediction_results = X.copy()

        prediction_results["Predicted_Churn"] = (
            prediction_labels.values
        )

        prediction_results["Churn_Probability"] = y_prob

        if has_target:

            prediction_results["Actual_Churn"] = (
                actual_target.values
            )

        st.success("Predictions generated successfully!")

        st.dataframe(
            prediction_results.head(20),
            use_container_width=True
        )

        # Download predictions
        csv_data = prediction_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv"
        )

        # =====================================================
        # 9. PREDICTION SUMMARY
        # =====================================================

        st.subheader("Prediction Summary")

        churn_count = int(
            np.sum(y_pred == 1)
        )

        non_churn_count = int(
            np.sum(y_pred == 0)
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Customers",
            len(y_pred)
        )

        col2.metric(
            "Predicted Churn",
            churn_count
        )

        col3.metric(
            "Predicted Non-Churn",
            non_churn_count
        )

        st.bar_chart(
            pd.Series(
                {
                    "Churn": churn_count,
                    "Non-Churn": non_churn_count
                }
            )
        )

        # =====================================================
        # 10. MODEL EVALUATION
        # =====================================================

        if has_target:

            st.divider()

            st.header("4. Model Performance Evaluation")

            # Convert actual labels to numeric
            y_true = actual_target.astype(str).str.lower().map(
                {"no": 0, "yes": 1}
            )

            if y_true.isna().any():

                st.error(
                    "Invalid target values. The churn column "
                    "must contain only 'yes' or 'no'."
                )

                st.stop()

            y_true = y_true.astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)

            auc = roc_auc_score(y_true, y_prob)

            precision = precision_score(
                y_true, y_pred, zero_division=0
            )

            recall = recall_score(
                y_true, y_pred, zero_division=0
            )

            f1 = f1_score(
                y_true, y_pred, zero_division=0
            )

            mcc = matthews_corrcoef(y_true, y_pred)

            # Display metrics
            st.subheader(
                f"Evaluation Metrics – {selected_model}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            # =====================================================
            # 11. CONFUSION MATRIX
            # =====================================================

            st.divider()

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )

            st.bar_chart(cm_df)

            # =====================================================
            # 12. CLASSIFICATION REPORT
            # =====================================================

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                target_names=["No Churn", "Churn"],
                zero_division=0,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

            # =====================================================
            # 13. COMPARE ALL FIVE MODELS
            # =====================================================

            st.divider()

            st.header("5. Compare All Classification Models")

            if st.button("Evaluate All Five Models"):

                comparison_results = []

                for name, path in model_paths.items():

                    if not os.path.exists(path):
                        st.warning(
                            f"{name} model file is missing."
                        )
                        continue

                    classifier = load_model(path)

                    predictions = classifier.predict(X)

                    probabilities = (
                        classifier.predict_proba(X)[:, 1]
                    )

                    comparison_results.append({

                        "ML Model Name": name,

                        "Accuracy": accuracy_score(
                            y_true, predictions
                        ),

                        "AUC": roc_auc_score(
                            y_true, probabilities
                        ),

                        "Precision": precision_score(
                            y_true, predictions,
                            zero_division=0
                        ),

                        "Recall": recall_score(
                            y_true, predictions,
                            zero_division=0
                        ),

                        "F1": f1_score(
                            y_true, predictions,
                            zero_division=0
                        ),

                        "MCC": matthews_corrcoef(
                            y_true, predictions
                        )
                    })

                comparison_df = pd.DataFrame(
                    comparison_results
                )

                st.dataframe(
                    comparison_df.round(4),
                    use_container_width=True
                )

                st.subheader("Model Performance Comparison")

                st.bar_chart(
                    comparison_df.set_index(
                        "ML Model Name"
                    )[["Accuracy", "AUC", "Precision",
                       "Recall", "F1"]]
                )

        else:

            st.info(
                "No actual churn column was found. "
                "Predictions are displayed, but evaluation "
                "metrics cannot be calculated without actual labels."
            )

    except Exception as e:

        st.error(
            f"An error occurred while processing the dataset: {e}"
        )

else:

    st.info(
        "Please upload a customer dataset in CSV format to begin."
    )


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "M.Tech (AIML) | BITS Pilani WILP | Machine Learning Assignment 2"
)
