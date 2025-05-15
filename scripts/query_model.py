import requests
import json
import pandas as pd
import os

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:8000/predict" # Pointing to FastAPI endpoint
SAMPLE_DATA_PATH = os.path.join('data', 'diabetes.csv')
N_SAMPLES = 3 # Number of samples to send for prediction

# --- Load Sample Data ---
print(f"Loading sample data from {SAMPLE_DATA_PATH}...")
try:
    df = pd.read_csv(SAMPLE_DATA_PATH)
    # Important: Use the *original* features BEFORE cleaning/preprocessing
    # The deployed pipeline expects the raw input format
    sample_df = df.drop('Outcome', axis=1).head(N_SAMPLES) 
except Exception as e:
    print(f"Error loading or sampling data: {e}")
    exit(1)

print(f"\nSample input data ({N_SAMPLES} rows):")
print(sample_df)

# FastAPI expects a flat JSON object that matches the PredictionInput Pydantic model.
# We'll send one sample at a time for simplicity with the current FastAPI setup.

# --- Send Requests (One by One) ---
print(f"\nSending POST requests to {SERVER_URL}...")
headers = {'Content-Type': 'application/json'}
all_predictions = []

for i in range(len(sample_df)):
    # Convert the row to a dictionary
    # This dictionary structure should match the Pydantic model in api/main.py
    data_dict_row = sample_df.iloc[i].to_dict()
    json_data_row = json.dumps(data_dict_row) # Send the flat dictionary as JSON

    print(f"\nSending data for sample {i+1}:")
    print(json_data_row)

    try:
        response = requests.post(SERVER_URL, headers=headers, data=json_data_row)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print("Response received:")
        print(f"Status Code: {response.status_code}")
        try:
            prediction_response = response.json()
            print("Prediction Response:")
            print(json.dumps(prediction_response, indent=4))
            if 'prediction' in prediction_response: # Assuming your FastAPI returns {'prediction': X}
                all_predictions.append(prediction_response['prediction'])
            else:
                print("Warning: 'prediction' key not found in response.")
                all_predictions.append(None) # Or handle error appropriately

        except json.JSONDecodeError:
            print("Response content is not valid JSON:")
            print(response.text)
            all_predictions.append(None)

    except requests.exceptions.RequestException as e:
        print(f"Error sending request for sample {i+1}: {e}")
        all_predictions.append(None)

print("\n--- All Predictions --- (from query_model.py)")
# Compare these predictions to the first N_SAMPLES of the 'Outcome' column in your data if desired.
print(all_predictions)

print("\nScript finished.") 