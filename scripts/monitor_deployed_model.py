import requests
import json
import time
import random
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os

# --- Configuration ---
FASTAPI_ENDPOINT_URL = "http://127.0.0.1:8000/predict"  # Your FastAPI prediction endpoint
MLFLOW_MONITORING_EXPERIMENT_NAME = "Pima_Model_Monitoring"
N_BATCHES = 10  # Number of monitoring batches to simulate
BATCH_SIZE = 20  # Number of samples per batch
DELAY_BETWEEN_BATCHES_SECONDS = 5  # Simulate time passing

# Path to your original dataset to sample "new" data from
# In a real scenario, this would be a stream of new, unseen data
SAMPLE_DATA_PATH = os.path.join('data', 'diabetes.csv') 

# --- Helper Functions ---
def get_sample_data_with_ground_truth(n_samples):
    """
    Simulates fetching new data and its ground truth.
    Loads from the original CSV, samples, and returns features + true labels.
    """
    try:
        df = pd.read_csv(SAMPLE_DATA_PATH)
        if len(df) < n_samples:
            # If dataset is smaller than requested samples, use the whole dataset
            sample_df = df.sample(n=len(df), replace=False, random_state=random.randint(1, 1000))
        else:
            sample_df = df.sample(n=n_samples, replace=False, random_state=random.randint(1, 1000))
        
        X_new = sample_df.drop('Outcome', axis=1)
        y_true_new = sample_df['Outcome']
        return X_new, y_true_new
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {SAMPLE_DATA_PATH}")
        return None, None
    except Exception as e:
        print(f"Error loading or sampling data: {e}")
        return None, None

def make_prediction_request(input_data_dict):
    """Sends a single prediction request to the FastAPI endpoint."""
    try:
        response = requests.post(FASTAPI_ENDPOINT_URL, json=input_data_dict)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()['prediction'] # Assumes 'prediction' key from PredictionOutput
    except requests.exceptions.RequestException as e:
        print(f"Error during prediction request for {input_data_dict}: {e}")
        return None # Or handle as appropriate, e.g., return a default non-prediction value

# --- Main Monitoring Loop ---
if __name__ == "__main__":
    print(f"Starting model performance monitoring for {N_BATCHES} batches.")
    print(f"FastAPI Endpoint: {FASTAPI_ENDPOINT_URL}")
    print(f"MLflow Experiment for Monitoring: {MLFLOW_MONITORING_EXPERIMENT_NAME}")

    # Set MLflow experiment for monitoring
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_MONITORING_EXPERIMENT_NAME)
        if experiment is None:
            print(f"Experiment '{MLFLOW_MONITORING_EXPERIMENT_NAME}' not found. Creating it.")
            experiment_id = mlflow.create_experiment(MLFLOW_MONITORING_EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)
        print(f"Using MLflow experiment ID: {experiment_id}")
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        exit(1)

    for batch_num in range(1, N_BATCHES + 1):
        print(f"\n--- Monitoring Batch {batch_num}/{N_BATCHES} ---")

        # 1. Simulate fetching new data and ground truth for the batch
        X_batch, y_true_batch = get_sample_data_with_ground_truth(BATCH_SIZE)

        if X_batch is None or y_true_batch is None:
            print(f"Failed to get data for batch {batch_num}. Skipping.")
            time.sleep(DELAY_BETWEEN_BATCHES_SECONDS)
            continue
        
        print(f"Fetched {len(X_batch)} new data samples for batch {batch_num}.")

        # 2. Get predictions from the deployed model (via FastAPI)
        predictions_batch = []
        for i in range(len(X_batch)):
            # Convert row to dict for FastAPI Pydantic model
            # FastAPI expects a flat JSON object, not a list of objects for single predictions
            input_data_row_dict = X_batch.iloc[i].to_dict()
            
            # Simulate some processing time or network latency for individual requests
            # time.sleep(0.05) 
            
            prediction = make_prediction_request(input_data_row_dict)
            if prediction is not None:
                predictions_batch.append(prediction)
            else:
                # Handle failed prediction if necessary, e.g., skip this sample
                # For simplicity, we might just end up with fewer predictions than y_true
                print(f"Skipping a sample in batch {batch_num} due to prediction error.")
        
        if not predictions_batch or len(predictions_batch) != len(y_true_batch):
             print(f"Not enough successful predictions for batch {batch_num} to calculate metrics. Expected {len(y_true_batch)}, got {len(predictions_batch)}")
             time.sleep(DELAY_BETWEEN_BATCHES_SECONDS)
             continue


        # 3. Calculate performance metrics
        # Ensure y_true_batch and predictions_batch are aligned and of same length
        # If make_prediction_request can fail and skip, you need to align y_true_batch
        # For now, assuming all predictions were successful and in order.
        
        # Ensure y_true_batch corresponds to the samples for which predictions were made
        # This simple example assumes direct correspondence and all predictions succeed.
        # A more robust version would track which true labels correspond to successful predictions.

        accuracy = accuracy_score(y_true_batch, predictions_batch)
        f1 = f1_score(y_true_batch, predictions_batch, zero_division=0)
        precision = precision_score(y_true_batch, predictions_batch, zero_division=0)
        recall = recall_score(y_true_batch, predictions_batch, zero_division=0)
        
        # ROC AUC needs probabilities. If your FastAPI endpoint doesn't provide them,
        # you can't calculate it here directly unless your model object in FastAPI can also give probas.
        # For simplicity, we'll skip roc_auc if only class predictions are available.
        # y_proba_batch = ... # if you get probabilities
        # roc_auc = roc_auc_score(y_true_batch, y_proba_batch)

        print(f"Batch {batch_num} Metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

        # 4. Log metrics to MLflow
        with mlflow.start_run(run_name=f"Monitoring_Batch_{batch_num}") as run:
            mlflow.log_param("batch_number", batch_num)
            mlflow.log_param("batch_size", len(X_batch)) # Actual size used
            mlflow.log_param("successful_predictions", len(predictions_batch))

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            # mlflow.log_metric("roc_auc", roc_auc) # if calculated

            # You could also log simple data drift metrics here, e.g., mean of a few key features
            # for i, col in enumerate(X_batch.columns):
            #    if i < 3: # Log mean of first 3 features as an example
            #        mlflow.log_metric(f"mean_{col}", X_batch[col].mean())
            
            print(f"Logged metrics for batch {batch_num} to MLflow run ID: {run.info.run_id}")

        # 5. Wait before processing the next batch
        if batch_num < N_BATCHES:
            print(f"Waiting for {DELAY_BETWEEN_BATCHES_SECONDS} seconds before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES_SECONDS)

    print("\n--- Model performance monitoring finished ---")
    print(f"Check the '{MLFLOW_MONITORING_EXPERIMENT_NAME}' experiment in the MLflow UI.") 