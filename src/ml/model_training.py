import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from feature_engineering import get_processed_data
from time import time
import psutil
from imblearn.over_sampling import SMOTE

# Ensure models directory exists
os.makedirs("../../models", exist_ok=True)

# Get preprocessed data
X_train, X_test, y_train, y_test = get_processed_data()

# Apply SMOTE directly on Polars data by converting to numpy arrays
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train.to_numpy(), y_train.to_numpy())

# Compute scale_pos_weight for XGBoost
num_negatives = (y_train_balanced == 0).sum()
num_positives = (y_train_balanced == 1).sum()
scale_pos_weight = num_negatives / num_positives if num_positives > 0 else 1

# Initialize models with tuned parameters
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500,
        solver="liblinear",
        class_weight="balanced",
        penalty="l2",
        C=0.3,
        n_jobs=-1,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=8,
        max_features="sqrt",
        class_weight="balanced_subsample",  # Use balanced subsample for better handling of class imbalance
        bootstrap=True,   
        oob_score=True,    
        n_jobs=-1,        # Use all CPU cores
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="aucpr",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
}

# Function to monitor resource usage
def get_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory().percent
    return cpu_percent, memory_info

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"\n{name} training...")
    cpu_before, mem_before = get_resource_usage()
    start_time = time()

    # Train models on balanced numpy arrays
    model.fit(X_train_balanced, y_train_balanced)
    
    train_time = time() - start_time
    cpu_after, mem_after = get_resource_usage()

    # Predict and evaluate on test set
    y_pred = model.predict(X_test.to_numpy())
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"CPU Usage Before: {cpu_before:.2f}% | After: {cpu_after:.2f}%")
    print(f"Memory Usage Before: {mem_before:.2f}% | After: {mem_after:.2f}%")

    # Save the trained model
    model_filename = f"../../models/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved model to {model_filename}")
