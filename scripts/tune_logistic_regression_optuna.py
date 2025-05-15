import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

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

MLFLOW_EXPERIMENT_NAME = "Pima Diabetes Classification - Tuning" # New experiment for tuning runs
PARENT_RUN_NAME = "Tune_LogisticRegression_Optuna"
CHILD_RUN_PREFIX = "LR_Optuna_Trial"
REGISTERED_MODEL_NAME = "pima-logreg-tuned"

# --- MLflow Setup ---
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- Load and Prepare Data (only need Train set for tuning) ---
print("Loading and cleaning data...")
df_processed = load_and_clean_data(DATA_PATH)

print("Separating features and target...")
X = df_processed.drop('Outcome', axis=1)
y = df_processed['Outcome']

print("Splitting data into train and test sets...")
# We only use X_train_val, y_train_val for tuning
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train/Validation set size: {X_train_val.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Define Optuna Objective Function ---
def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    # Start a nested MLflow run for this trial
    with mlflow.start_run(run_name=f"{CHILD_RUN_PREFIX}_{trial.number}", nested=True) as child_run:
        
        # 1. Suggest hyperparameters
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        
        # Suggest penalty from the full potential list first
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        
        # Prune invalid combinations
        if solver == 'liblinear' and penalty == 'elasticnet':
            raise optuna.TrialPruned(f"Solver '{solver}' does not support penalty '{penalty}'.")
            
        # Suggest l1_ratio only if penalty is 'elasticnet' (which implies solver is 'saga')
        l1_ratio = None
        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
                
        C = trial.suggest_float('C', 1e-4, 1e2, log=True) # Log scale for regularization
        
        lr_params = {
            'solver': solver,
            'penalty': penalty,
            'C': C,
            'l1_ratio': l1_ratio, # Keep it, will be None if not elasticnet
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'max_iter': 1000 # Increase max_iter for saga solver
        }
        # Remove None params for the LogisticRegression call
        lr_params_cleaned = {k: v for k, v in lr_params.items() if v is not None}
        
        # Log suggested parameters to MLflow (log the potentially None l1_ratio for clarity)
        mlflow.log_params(trial.params)
        if l1_ratio is not None:
            mlflow.log_param('l1_ratio', l1_ratio) # Explicitly log l1_ratio if suggested
        
        # 2. Create pipeline with suggested params
        preprocessor = create_preprocessing_pipeline()
        model = LogisticRegression(**lr_params_cleaned)
        pipeline = Pipeline([('preprocessing', preprocessor), 
                             ('classifier', model)])
        
        # 3. Evaluate using cross-validation on the training/validation set
        # Important: Use the *training* data (X_train_val) for CV during tuning
        try:
            scores = cross_val_score(pipeline, X_train_val, y_train_val, 
                                     cv=CV_FOLDS, scoring=OPTIMIZE_METRIC, n_jobs=-1)
            metric_value = np.mean(scores)
            metric_std = np.std(scores)
            
            # Log the result metric
            mlflow.log_metric(f"cv_{OPTIMIZE_METRIC}_mean", metric_value)
            mlflow.log_metric(f"cv_{OPTIMIZE_METRIC}_std", metric_std)
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            print(f"Params: {trial.params}")
            # Log failure and return a bad score (or handle as Optuna prefers, e.g., raise optuna.TrialPruned)
            mlflow.log_metric(f"cv_{OPTIMIZE_METRIC}_mean", -1.0) # Indicate failure
            metric_value = -1.0 # Return a value indicating failure

        return metric_value # Optuna maximizes this value

# --- Run Optuna Study with MLflow Callback ---
# Callback to automatically log study results and best trial to a parent run
# Note: Optuna's MLflowCallback creates the parent run automatically
mlflow_callback = optuna.integration.MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name=f"cv_{OPTIMIZE_METRIC}_mean", # Metric logged inside objective
    create_experiment=False # Experiment already set
    # Use run_name=PARENT_RUN_NAME ? Let's test if callback names it or if we need parent context
)

print(f"Starting Optuna study: {PARENT_RUN_NAME}")
study = optuna.create_study(direction='maximize', 
                            study_name=PARENT_RUN_NAME) # Give study a name
study.optimize(objective, n_trials=N_TRIALS, callbacks=[mlflow_callback], n_jobs=1) # n_jobs=1 for sequential trials to avoid MLflow logging issues if not thread-safe

print("Optuna study finished.")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial ({OPTIMIZE_METRIC}): {study.best_value}")
print("Best hyperparameters found: ", study.best_params)

# --- Optional: Train final model with best params and log/register ---
print("Training final model with best parameters found by Optuna...")
best_params_optuna = study.best_params
# Map optuna params to model params (handle l1_ratio etc.)
final_lr_params = {
    'solver': best_params_optuna['solver'],
    'penalty': best_params_optuna.get('penalty'), # Use .get() for safety
    'C': best_params_optuna['C'],
    'l1_ratio': best_params_optuna.get('l1_ratio'), # Will be None if not elasticnet
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'max_iter': 1000
}
final_lr_params = {k: v for k, v in final_lr_params.items() if v is not None}


# Need to get the parent run ID created by MLflowCallback to log the final model
# This is a bit tricky, check Optuna docs/examples for best practice
# Alternative: Create the parent run explicitly *before* the study
print("Logging best parameters and final model to a new parent run (manual approach)...")
with mlflow.start_run(run_name=f"{PARENT_RUN_NAME}_Final") as parent_run:
    mlflow.log_params(study.best_params)
    mlflow.log_metric(f"best_cv_{OPTIMIZE_METRIC}_mean", study.best_value)
    mlflow.log_param("n_trials", N_TRIALS)
    mlflow.log_param("cv_folds", CV_FOLDS)
    
    # Train final model on the full training set (X_train_val, y_train_val)
    final_preprocessor = create_preprocessing_pipeline()
    final_model = LogisticRegression(**final_lr_params)
    final_pipeline = Pipeline([('preprocessing', final_preprocessor), 
                             ('classifier', final_model)])
    
    final_pipeline.fit(X_train_val, y_train_val)
    print("Final model training complete.")
    
    # Evaluate on the *test* set (optional, but good to log final performance)
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

    # Log and register the final tuned pipeline
    mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path="sklearn-model",
        input_example=X_train_val.iloc[:5],
        registered_model_name=REGISTERED_MODEL_NAME 
    )
    print(f"Registered final tuned model as: {REGISTERED_MODEL_NAME}")

print("Script finished.") 