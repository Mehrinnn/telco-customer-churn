import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw DataFrame:
    1. Coerces TotalCharges to float64 (blank strings ' ' become NaN).
    2. Maps Churn binary target: 'Yes' -> 1, 'No' -> 0.
    """
    df_clean = df.copy()
    
    # Coerce TotalCharges to float
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Encode target if present
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
    return df_clean

def get_feature_lists():
    """
    Returns column groupings: numerical features, categorical features, identifier column, target column.
    """
    id_col = 'customerID'
    target_col = 'Churn'
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    return num_cols, cat_cols, id_col, target_col

def create_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer pipeline for numerical and categorical features.
    
    Numerical Pipeline:
      - SimpleImputer(strategy='median')
      - StandardScaler()
      
    Categorical Pipeline:
      - SimpleImputer(strategy='most_frequent')
      - OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    """
    num_cols, cat_cols, _, _ = get_feature_lists()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='drop'  # Automatically drops customerID and target column
    )
    
    return preprocessor

if __name__ == '__main__':
    # Sanity test execution
    print("Testing Preprocessing Module...")
    raw_df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cleaned_df = clean_raw_data(raw_df)
    
    num_cols, cat_cols, id_col, target_col = get_feature_lists()
    X = cleaned_df[num_cols + cat_cols]
    y = cleaned_df[target_col]
    
    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    
    # Get feature names out of encoder
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_cols)
    
    print(f"\nRaw Features Shape : {X.shape}")
    print(f"Transformed X Shape: {X_trans.shape}")
    print(f"Total Features Generated after One-Hot Encoding: {len(all_feature_names)}")
    print("\nSample Encoded Feature Names:")
    print(all_feature_names[:10])
