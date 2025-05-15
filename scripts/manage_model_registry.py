import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MODEL_NAME = "pima-rf-tuned"  # The model we want to manage
TARGET_STAGE = "Staging"
STAGING_DESCRIPTION = f"Transitioning latest version of {MODEL_NAME} to Staging for validation."

# --- Main Logic ---
def transition_latest_model_to_staging(client, model_name):
    print(f"Attempting to transition latest version of model '{model_name}' to '{TARGET_STAGE}'...")

    try:
        # Get the latest version of the model
        latest_versions = client.get_latest_versions(name=model_name)
        if not latest_versions:
            print(f"No versions found for model '{model_name}'. Please register a version first.")
            return

        # Assuming we want the absolute latest, typically the first in the list if sorted by version number descending
        # Or, if you know the specific version string, you can use that.
        # For simplicity, let's take the one with the highest version number.
        latest_version_obj = max(latest_versions, key=lambda mv: int(mv.version))
        version_to_stage = latest_version_obj.version

        print(f"Found latest version: {version_to_stage} for model '{model_name}'.")

        print(f"Transitioning Version {version_to_stage} to '{TARGET_STAGE}'...")
        client.transition_model_version_stage(
            name=model_name,
            version=version_to_stage,
            stage=TARGET_STAGE,
            archive_existing_versions=True  # Archive other versions that might be in Staging for this model
        )
        print(f"Successfully transitioned Version {version_to_stage} of '{model_name}' to '{TARGET_STAGE}'.")
        
        client.update_model_version(
            name=model_name,
            version=version_to_stage,
            description=STAGING_DESCRIPTION
        )
        print(f"Added/Updated description for Version {version_to_stage} in '{TARGET_STAGE}'.")

    except Exception as e:
        print(f"Error during model transition or update for '{model_name}': {e}")

if __name__ == "__main__":
    # Initialize the MLflow Client
    # It will automatically connect to your local tracking server (mlruns folder by default)
    # or use MLFLOW_TRACKING_URI if set.
    try:
        client = MlflowClient()
        # Ensure the experiment and runs that register "pima-rf-tuned" have been executed
        # e.g., scripts/tune_random_forest_optuna.py
        transition_latest_model_to_staging(client, MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize MLflow client or run transition: {e}")
    
    print("\nScript finished. Check the MLflow UI 'Models' tab to see the updated stages.") 