# MLOps Project: Pima Diabetes Prediction with MLflow and FastAPI

This project demonstrates a comprehensive machine learning (ML) system to manage the entire lifecycle of an ML model for the Pima Diabetes dataset. It includes experiment tracking, model training, hyperparameter tuning, model deployment as a REST API using FastAPI, and a basic framework for performance monitoring.

## Project Structure

```
MLopsYaZ/
├── api/
│   └── main.py               # FastAPI application for model serving
├── data/
│   └── diabetes.csv          # Dataset (ensure this is present)
├── mlruns/                   # MLflow tracking data (generated automatically)
├── notebooks/
│   └── 01_data_exploration.ipynb # Jupyter notebook for EDA
├── scripts/
│   ├── data_utils.py         # Utility functions for data loading and cleaning
│   ├── manage_model_registry.py # Script to manage model stages in MLflow Registry
│   ├── monitor_deployed_model.py # Script to simulate monitoring of the deployed model
│   ├── preprocessing.py      # Preprocessing pipeline definition
│   ├── query_model.py        # Script to query the deployed FastAPI model
│   ├── train_baseline.py     # Training script for a baseline Logistic Regression model
│   ├── train_random_forest.py # Training script for a Random Forest model
│   ├── train_xgboost.py      # Training script for an XGBoost model
│   ├── tune_logistic_regression_optuna.py # Optuna tuning for Logistic Regression
│   ├── tune_random_forest_optuna.py    # Optuna tuning for Random Forest
│   └── tune_xgboost_optuna.py          # Optuna tuning for XGBoost
├── .gitignore
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Setup and Installation

1.  **Clone the Repository (if applicable) or Unpack the Project Files**
    If you received this project as a ZIP file or similar, unpack it to your desired location.
    ```bash
    # Example: git clone <your-repo-url>
    # cd MLopsYaZ
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Dependencies**
    Ensure you have the `data/diabetes.csv` file in place.
    ```bash
    pip install -r requirements.txt
    ```

**Important Note on `mlruns` Directory:**
This project uses MLflow, which stores experiment tracking data and model artifacts in a local directory named `mlruns/`.

*   **If this project package includes an `mlruns/` directory:**
    *   This directory contains a snapshot of the experiment history from the original development environment. You can explore this history by running `mlflow ui` in the project root and navigating to `http://127.0.0.1:5000`.
    *   **Crucially, for the model serving application (FastAPI) and other scripts that load models from the registry to work correctly on your machine, you MUST delete this provided `mlruns/` directory (you may want to back it up first if you wish to preserve the original history snapshot separately).**
    *   After deleting it, you must then run the training, tuning, and model management scripts as described in "Step 2" and "Step 3" below. This will generate a new `mlruns/` directory with the correct model artifact paths for *your* local system.
*   **If no `mlruns/` directory is included, or after you've deleted a provided one:**
    *   The necessary `mlruns/` directory and its contents (with correct local file paths for your system) will be **automatically generated** when you execute the training and tuning scripts in the steps below.

## Running the Project: Step-by-Step

1.  **Run MLflow UI (Optional but Recommended)**
    Open a terminal in the project root directory and run:
    ```bash
    mlflow ui
    ```
    This will start the MLflow UI, typically accessible at `http://127.0.0.1:5000`. Keep this running in a separate terminal to observe experiments and model registry changes as you run the scripts below.

2.  **Run Training and Hyperparameter Tuning Scripts**
    **This is a crucial step.** Executing these scripts will train the models, log experiments to MLflow, and register models in the MLflow Model Registry. This process also ensures that MLflow records the correct file paths for the model artifacts on *your current system*.

    Open a terminal in the project root directory and run the following commands sequentially:
    ```bash
    python scripts/train_baseline.py
    python scripts/train_random_forest.py
    python scripts/train_xgboost.py
    
    python scripts/tune_logistic_regression_optuna.py
    python scripts/tune_random_forest_optuna.py 
    python scripts/tune_xgboost_optuna.py
    ```
    After running `tune_random_forest_optuna.py`, a model named `pima-rf-tuned` will be registered in MLflow.

3.  **Transition `pima-rf-tuned` Model to Staging**
    This script will find the latest version of the `pima-rf-tuned` model (which you just trained and registered in the previous step) and move it to the "Staging" stage in the MLflow Model Registry.
    ```bash
    python scripts/manage_model_registry.py
    ```
    You can verify this change in the MLflow UI under the "Models" tab.

4.  **Run the FastAPI Model Serving Application**
    This will start the API server, which loads the `pima-rf-tuned` model from the "Staging" stage.
    ```bash
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   The server will be accessible on your local machine.
    *   To view the API documentation (Swagger UI), open your browser and go to: `http://127.0.0.1:8000/docs` or `http://localhost:8000/docs`.
    *   The main endpoint showing model status is `http://127.0.0.1:8000/` or `http://localhost:8000/`.

5.  **Query the Deployed Model**
    While the FastAPI server is running, open a new terminal and run:
    ```bash
    python scripts/query_model.py
    ```
    This script will send sample data to the `/predict` endpoint of your FastAPI application and print the predictions.

6.  **Run Model Performance Monitoring Simulation**
    While the FastAPI server is running, open another new terminal and run:
    ```bash
    python scripts/monitor_deployed_model.py
    ```
    This script simulates new data arriving in batches, queries the FastAPI endpoint for predictions, calculates performance metrics against (simulated) ground truth, and logs these metrics to a new MLflow experiment named "Pima_Model_Monitoring". You can observe these monitoring runs in the MLflow UI.

## Key Technologies Used

*   **MLflow**: For end-to-end ML lifecycle management (tracking, registry, packaging).
*   **Scikit-learn**: For ML model development and preprocessing pipelines.
*   **XGBoost**: For an alternative gradient boosting model.
*   **Optuna**: For hyperparameter optimization.
*   **FastAPI**: For serving the ML model as a REST API.
*   **Uvicorn**: As an ASGI server for FastAPI.
*   **Pandas & NumPy**: For data manipulation.

## Notes

*   The `MLFLOW_TRACKING_URI` environment variable is not explicitly set in these instructions, so MLflow will use a local `mlruns` directory (created in your project root) for storage. This is suitable for local development and demonstration. For a more robust setup, you might configure a remote tracking server, backend database, and artifact store.
*   The Pima Diabetes dataset (`data/diabetes.csv`) is assumed to be present in the `data` directory.
*   The numpy version warning seen during FastAPI startup is due to a slight mismatch between the environment where the model was logged and the current serving environment. For production, ensuring exact dependency matches is crucial.
*   **Troubleshooting Model Loading Issues:** If the FastAPI application reports an error like "No such file or directory" when trying to load a model, it almost always means that MLflow is trying to use incorrect paths for the model artifacts.
    *   If you started with an `mlruns/` directory provided with the project, ensure you have **deleted it and regenerated it** by running all training, tuning (Step 2), and model management scripts (Step 3) in your current project environment *before* starting the FastAPI server.
    *   If you did not start with a provided `mlruns/` directory, simply ensure you have run all the scripts mentioned above (Step 2 and Step 3) to populate it correctly.