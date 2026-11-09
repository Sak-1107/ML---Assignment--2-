import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.10f}'.format)
pd.set_option('display.max_colwidth', None)
from datetime import datetime
import gc
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

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
    matthews_corrcoef
)

data = pd.read_csv(r"C:\Users\Sakshi\Downloads\ML Assigmnet 2 AIML\ML Assigmnet 2 AIML\train.csv",
                sep=',', low_memory = False)

data.shape

print("Dataset Shape:", data.shape)
print("\nTarget Distribution:")
print(data["churn"].value_counts())

# Remove duplicate records if any
data = data.drop_duplicates()

# Separate features and target
X = data.drop(columns=["churn"])
y = data["churn"].map({"no": 0, "yes": 1})

# --------------------------------------------------
# 2. IDENTIFY FEATURE TYPES
# --------------------------------------------------

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:", num_cols)
print("\nCategorical Features:", cat_cols)

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Save test data with original target labels
test_data = X_test.copy()
test_data["churn"] = y_test.map({0: "no", 1: "yes"})

test_data.to_csv("test_data.csv", index=False)

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical",
            StandardScaler(),
            num_cols
        ),
        ( "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_cols )],
    sparse_threshold=0.0
)

# --------------------------------------------------
# 5. DEFINE CLASSIFICATION MODELS
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# --------------------------------------------------
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []

for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability predictions for AUC
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {
        "ML Model Name": name,

        "Accuracy": accuracy_score(
            y_test, y_pred
        ),

        "AUC": roc_auc_score(
            y_test, y_prob
        ),

        "Precision": precision_score(
            y_test, y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test, y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test, y_pred
        )
    }

    results.append(metrics)

    # Save complete pipeline
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        pipeline,
        f"model/{filename}.joblib"
    )

    print("Metrics:", metrics)
    print("Model saved successfully!")

# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.round(4)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nAll five models trained successfully!")
print("Test data and model artifacts saved.")

