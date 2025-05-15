from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os


MODEL_NAME = "pima-rf-tuned"  
MODEL_STAGE = "Staging"    

MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pima Diabetes Prediction API", 
    version="1.0",
    description=f"API for Pima Diabetes prediction using MLflow model: {MODEL_NAME} ({MODEL_STAGE})"
)

# --- Load MLflow Model --- 

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Attempting to load model from MLflow registry: {MLFLOW_MODEL_URI}")
        # Ensure MLFLOW_TRACKING_URI is set if you are using a remote tracking server
        
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        print(f"Model '{MLFLOW_MODEL_URI}' loaded successfully.")
    except Exception as e:
        print(f"Critical Error: Model could not be loaded at startup: {e}")
       
        model = None 

# --- Pydantic Models for Input and Output ---
class PredictionInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

    class Config:
        schema_extra = {
            "example": {
                "Pregnancies": 6,
                "Glucose": 148.0,
                "BloodPressure": 72.0,
                "SkinThickness": 35.0,
                "Insulin": 0.0,
                "BMI": 33.6,
                "DiabetesPedigreeFunction": 0.627,
                "Age": 50
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    
# --- API Endpoints ---
@app.get("/", summary="Root endpoint with API and model information")
async def read_root():
    model_status = "Loaded" if model is not None else "Not loaded (Error during startup)"
    return {
        "message": "Welcome to the Pima Diabetes Prediction API",
        "model_uri_configured": MLFLOW_MODEL_URI,
        "model_status": model_status
    }

@app.post("/predict", response_model=PredictionOutput, summary="Make a single prediction")
async def predict(data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model ({MLFLOW_MODEL_URI}) is not loaded or failed to load. Cannot make predictions."
        )
    try:
        # The MLflow scikit-learn pipeline expects a DataFrame
        # Define the expected order of columns, matching the training data
        expected_column_order = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        input_df = pd.DataFrame([data.dict()], columns=expected_column_order)
        
       
       
        prediction_result = model.predict(input_df)
        
        predicted_class = int(prediction_result[0])
        
        return {"prediction": predicted_class}
    
    except mlflow.exceptions.MlflowException as mle:
        print(f"MLflow Exception during prediction: {mle}")
        raise HTTPException(status_code=500, detail=f"MLflow related error during prediction: {str(mle)}")
    except Exception as e:
        print(f"General Exception during prediction: {e}")
        # Log the full error for debugging if needed: print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

