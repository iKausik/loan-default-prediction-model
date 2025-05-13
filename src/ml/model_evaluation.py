import os
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from feature_engineering import get_processed_data

# Load preprocessed data
X_train, X_test, y_train, y_test = get_processed_data()

# Paths to saved model files
model_dir = "../../models"
report_dir = "../../reports"
model_files = {
    "Logistic Regression": f"{model_dir}/logistic_regression.pkl",
    "Random Forest": f"{model_dir}/random_forest.pkl", 
    "XGBoost": f"{model_dir}/xgboost.pkl"
}

# Ensure reports directory exists
os.makedirs(report_dir, exist_ok=True)

def get_evaluation_metrics():
    metrics = {}
    all_metrics = []

    for model_name, model_path in model_files.items():
        if not os.path.exists(model_path):
            print(f"Model file for {model_name} not found at {model_path}. Skipping.")
            continue

        # Load the model
        model = joblib.load(model_path)

        # Convert X_test to numpy
        X_test_array = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test.values

        # Make predictions
        y_pred = model.predict(X_test_array)

        try:
            y_proba = model.predict_proba(X_test_array)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except AttributeError:
            y_proba = None
            roc_auc = None

        # Calculate evaluation metrics
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")

        # Store metrics
        metrics[model_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        # Print metrics
        print(f"\n{model_name} Evaluation Metrics")
        print("-" * 40)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Add to DataFrame list
        all_metrics.append({
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        })

    # Save to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(report_dir, "model_metrics.csv"), index=False)

    return metrics

get_evaluation_metrics()
