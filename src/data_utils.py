import pandas as pd
import numpy as np
import os

def load_and_clean_data(data_path):
    """
    Loads the Pima diabetes dataset from the specified path,
    replaces 0s with NaNs in specified columns, and returns
    the processed DataFrame.

    Args:
        data_path (str): The relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame with NaNs instead of 0s 
                      in relevant columns.
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully from {data_path}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {data_path}")
        # Consider raising an error or returning None based on desired handling
        raise 

    # Columns where 0 is biologically impossible and might mean missing data
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace 0 with NaN in these columns using .loc to avoid warnings
    df.loc[:, cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    
    print("Replaced 0s with NaNs in Glucose, BloodPressure, SkinThickness, Insulin, BMI.")
    
    return df 