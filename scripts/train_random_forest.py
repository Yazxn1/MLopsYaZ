import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add project root to path

from src.data_utils import load_and_clean_data
from src.preprocessing import create_preprocessing_pipeline

data_path = os.path.join('data', 'diabetes.csv')
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "Pima Diabetes Classification - Baseline" # Keep same experiment for comparison
MLFLOW_RUN_NAME = "RandomForest_Default" # Changed run name

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- Load and Prepare Data ---
print("Loading and cleaning data...")
df_processed = load_and_clean_data(data_path)

print("Separating features and target...")
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Define Model and Pipeline ---
print("Creating preprocessing pipeline...")
preprocessor = create_preprocessing_pipeline()

print("Defining Random Forest model...")
rf_params = {
    'n_estimators': 200, 
    'class_weight': 'balanced', 
    'random_state': RANDOM_STATE,
    'n_jobs': -1 
}
rf_model = RandomForestClassifier(**rf_params)

print("Creating final ML pipeline...")
final_pipeline = Pipeline([('preprocessing', preprocessor), 
                         ('classifier', rf_model)])

# --- MLflow Tracking ---
print(f"Starting MLflow run: {MLFLOW_RUN_NAME}")
with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
    # Log parameters
    print("Logging parameters...")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_params(rf_params)
    mlflow.log_param("preprocessing_steps", [step[0] for step in preprocessor.steps])
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    # Fit the pipeline
    print("Fitting the final pipeline...")
    final_pipeline.fit(X_train, y_train)
    print("Pipeline fitting complete.")

    # Make predictions
    print("Making predictions on the test set...")
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    print("Logging metrics...")
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    mlflow.log_metrics(metrics)
    print(f"Test Metrics: {metrics}")

    # Log the entire pipeline (preprocessor + model)
    print("Logging the scikit-learn pipeline...")
    mlflow.sklearn.log_model(
        sk_model=final_pipeline, 
        artifact_path="sklearn-model", 
        input_example=X_train.iloc[:5], 
        registered_model_name="pima-random-forest-default"
    )
    print("Model logging complete.")

    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"MLflow Run completed for {MLFLOW_RUN_NAME}.")

print("\nScript finished. To view results, run 'mlflow ui' in your terminal in the project root directory.") 