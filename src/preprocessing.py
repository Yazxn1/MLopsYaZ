from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

def create_preprocessing_pipeline():
    """
    Creates and returns the scikit-learn preprocessing pipeline
    combining KNN and Median imputation with power transformation and scaling.
    """
    
    # Define features for each imputation strategy
    knn_impute_features = ['SkinThickness', 'Insulin']
    median_impute_features = ['Glucose', 'BloodPressure', 'BMI']
    
    # Create the ColumnTransformer for combined imputation
    imputer = ColumnTransformer(
        transformers=[
            ('knn_imp', KNNImputer(n_neighbors=5), knn_impute_features),
            ('median_imp', SimpleImputer(strategy="median", add_indicator=True), median_impute_features)
        ],
        remainder='passthrough' # Pass through other columns (Age, Pregnancies, DPF etc.)
    )

    # Define the subsequent numeric transformations 
    numeric_transformer = Pipeline(steps=[
        ('power_transform', PowerTransformer(method='yeo-johnson')), # Handles skewness
        ('scaler', StandardScaler()) # Scales data
    ])

    # Create the full preprocessing pipeline
    full_preprocessor = Pipeline(steps=[
        ('imputer', imputer),             # Apply KNN and Median imputation
        ('transformer', numeric_transformer) # Apply PowerTransform and Scaling
    ])
    
    return full_preprocessor 