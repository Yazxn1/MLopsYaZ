import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier # Changed model
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score # Add missing metrics

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from src.data_utils import load_and_clean_data
from src.preprocessing import create_preprocessing_pipeline

# --- Configuration ---
DATA_PATH = os.path.join('data', 'diabetes.csv')
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TRIALS = 50 # Number of Optuna trials to run
CV_FOLDS = 5 # Number of cross-validation folds
OPTIMIZE_METRIC = 'f1_weighted' # Metric to maximize during tuning

MLFLOW_EXPERIMENT_NAME = "Pima Diabetes Classification - Tuning" # Keep same experiment
PARENT_RUN_NAME = "Tune_RandomForest_Optuna" # Changed parent run name
CHILD_RUN_PREFIX = "RF_Optuna_Trial" 
REGISTERED_MODEL_NAME = "pima-rf-tuned" 

# --- MLflow Setup ---
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- Load and Prepare Data (only need Train set for tuning) ---
print("Loading and cleaning data...")
df_processed = load_and_clean_data(DATA_PATH)

print("Separating features and target...")
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

print("Splitting data into train and test sets...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train/Validation set size: {X_train_val.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Define Optuna Objective Function ---
def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    with mlflow.start_run(run_name=f"{CHILD_RUN_PREFIX}_{trial.number}", nested=True):
        
        # 1. Suggest hyperparameters for RandomForest
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        # max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) # Or suggest_float(0.1, 1.0)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            # 'max_features': max_features, 
            'criterion': criterion,
            'class_weight': 'balanced', 
            'random_state': RANDOM_STATE,
            'n_jobs': -1 
        }
                
        mlflow.log_params(trial.params)
        
        # 2. Create pipeline
        preprocessor = create_preprocessing_pipeline()
        model = RandomForestClassifier(**rf_params) # Use RF
        pipeline = Pipeline([('preprocessing', preprocessor), 
                             ('classifier', model)])
        
        # 3. Evaluate using cross-validation
        try:
            scores = cross_val_score(pipeline, X_train_val, y_train_val, 
                                     cv=CV_FOLDS, scoring=OPTIMIZE_METRIC, n_jobs=-1)
            metric_value = np.mean(scores)
            metric_std = np.std(scores)
            mlflow.log_metric(f"cv_{OPTIMIZE_METRIC}_mean", metric_value)
            mlflow.log_metric(f"cv_{OPTIMIZE_METRIC}_std", metric_std)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            print(f"Params: {trial.params}")
            mlflow.log_metric(f"cv_{OPTIMIZE_METRIC}_mean", -1.0)
            metric_value = -1.0

        return metric_value

# --- Run Optuna Study ---
mlflow_callback = optuna.integration.MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name=f"cv_{OPTIMIZE_METRIC}_mean",
    create_experiment=False
)

print(f"Starting Optuna study: {PARENT_RUN_NAME}")
study = optuna.create_study(direction='maximize', study_name=PARENT_RUN_NAME)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[mlflow_callback], n_jobs=1)

print("Optuna study finished.")
print(f"Best trial ({OPTIMIZE_METRIC}): {study.best_value}")
print("Best hyperparameters found: ", study.best_params)

# --- Train final model with best params ---
print("Training final model with best parameters...")
best_params_optuna = study.best_params
final_rf_params = {
    'n_estimators': best_params_optuna['n_estimators'],
    'max_depth': best_params_optuna['max_depth'],
    'min_samples_split': best_params_optuna['min_samples_split'],
    'min_samples_leaf': best_params_optuna['min_samples_leaf'],
    # 'max_features': best_params_optuna.get('max_features'), # Uncomment if used
    'criterion': best_params_optuna['criterion'],
    'class_weight': 'balanced', 
    'random_state': RANDOM_STATE,
    'n_jobs': -1 
}

print("Logging best parameters and final model to a new parent run...")
with mlflow.start_run(run_name=f"{PARENT_RUN_NAME}_Final") as parent_run:
    mlflow.log_params(study.best_params)
    mlflow.log_metric(f"best_cv_{OPTIMIZE_METRIC}_mean", study.best_value)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_param("cv_folds", CV_FOLDS)
    
    final_preprocessor = create_preprocessing_pipeline()
    final_model = RandomForestClassifier(**final_rf_params)
    final_pipeline = Pipeline([('preprocessing', final_preprocessor), 
                             ('classifier', final_model)])
    
    final_pipeline.fit(X_train_val, y_train_val)
    print("Final model training complete.")
    
    # Evaluate on test set
    y_test_pred = final_pipeline.predict(X_test)
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba)
    }
    mlflow.log_metrics(test_metrics)
    print(f"Final Model Test Metrics: {test_metrics}")

    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path="sklearn-model",
        input_example=X_train_val.iloc[:5],
        registered_model_name=REGISTERED_MODEL_NAME 
    )
    print(f"Registered final tuned model as: {REGISTERED_MODEL_NAME}")

print("\nScript finished.") 