import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc

from preprocess import create_preprocessor, get_feature_lists

def train_logistic_regression():
    """
    Trains a Logistic Regression model within a scikit-learn Pipeline on X_train,
    evaluates initial validation performance, and saves the trained model artifact.
    """
    processed_dir = os.path.join("data", "processed")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    
    num_cols, cat_cols, _, target_col = get_feature_lists()
    feature_cols = num_cols + cat_cols
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    
    # 1. Create Preprocessing + Classifier Pipeline
    preprocessor = create_preprocessor()
    classifier = LogisticRegression(
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # 2. Fit ONLY on Training Data
    print("Fitting Logistic Regression Pipeline on Training Data...")
    model_pipeline.fit(X_train, y_train)
    
    # 3. Validation Evaluation
    y_val_pred = model_pipeline.predict(X_val)
    y_val_proba = model_pipeline.predict_proba(X_val)[:, 1]
    
    # Metrics calculation
    val_f1 = f1_score(y_val, y_val_pred)
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_val_proba)
    val_pr_auc = auc(recall_vals, precision_vals)
    
    print("\n" + "="*60)
    print("PHASE 6: MODEL TRAINING & INITIAL VALIDATION RESULTS")
    print("="*60)
    print(f"Validation F1-Score  : {val_f1:.4f}")
    print(f"Validation PR-AUC    : {val_pr_auc:.4f}")
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred, target_names=['Retained (0)', 'Churned (1)']))
    
    # 4. Extract Top Positive and Negative Coefficients
    cat_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_cols
    
    coefs = model_pipeline.named_steps['classifier'].coef_[0]
    coef_df = pd.DataFrame({'Feature': all_feature_names, 'Coefficient': coefs})
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    
    print("\n--- TOP 5 POSITIVE COEFFICIENTS (Increases Churn Risk) ---")
    print(coef_df.head(5)[['Feature', 'Coefficient']].to_string(index=False))
    
    print("\n--- TOP 5 NEGATIVE COEFFICIENTS (Decreases Churn Risk) ---")
    print(coef_df.tail(5)[['Feature', 'Coefficient']].to_string(index=False))
    
    # 5. Save Model Artifact
    models_dir = os.path.join("src", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "logistic_regression_pipeline.joblib")
    joblib.dump(model_pipeline, model_path)
    print(f"\nSaved trained model pipeline to {model_path}")
    
    return model_pipeline

if __name__ == '__main__':
    train_logistic_regression()
