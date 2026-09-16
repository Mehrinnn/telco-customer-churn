"""
===============================================================================
EXPERIMENTAL COMPARISON SCRIPT (PHASE 9)
===============================================================================
NOTE: THIS IS AN OPTIONAL INTERNAL BENCHMARK SCRIPT.
OFFICIAL SUBMISSION MODEL: Logistic Regression (trained in src/train.py)

This script compares the official Logistic Regression model against a Random
Forest classifier to evaluate whether non-linear decision trees resolve subgroup
error blind spots (such as long-tenure churners).
===============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc, confusion_matrix

from preprocess import create_preprocessor, get_feature_lists

def compare_model_improvements():
    """
    Compares baseline Logistic Regression against Random Forest Classifier.
    Evaluates metrics on Validation (X_val) and Test (X_test).
    """
    processed_dir = os.path.join("data", "processed")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    num_cols, cat_cols, _, target_col = get_feature_lists()
    feature_cols = num_cols + cat_cols
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    preprocessor = create_preprocessor()
    
    # -------------------------------------------------------------
    # MODEL 1: Baseline Logistic Regression (OFFICIAL SUBMISSION MODEL)
    # -------------------------------------------------------------
    model_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42))
    ])
    model_lr.fit(X_train, y_train)
    val_probas_lr = model_lr.predict_proba(X_val)[:, 1]
    test_probas_lr = model_lr.predict_proba(X_test)[:, 1]
    
    # -------------------------------------------------------------
    # MODEL 2: Random Forest Classifier (EXPERIMENTAL BENCHMARK ONLY)
    # -------------------------------------------------------------
    model_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42))
    ])
    model_rf.fit(X_train, y_train)
    val_probas_rf = model_rf.predict_proba(X_val)[:, 1]
    test_probas_rf = model_rf.predict_proba(X_test)[:, 1]
    
    print("="*75)
    print("PHASE 9: EXPERIMENTAL MODEL COMPARISON (BENCHMARK ONLY)")
    print("="*75)
    
    def evaluate_model_pipeline(name, val_probas, test_probas, val_y, test_y):
        best_t, best_val_f1 = 0.50, 0.0
        for t in np.linspace(0.10, 0.90, 81):
            f1 = f1_score(val_y, (val_probas >= t).astype(int))
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_t = t
                
        val_prec, val_rec, _ = precision_recall_curve(val_y, val_probas)
        val_pr_auc = auc(val_rec, val_prec)
        
        test_preds = (test_probas >= best_t).astype(int)
        test_f1 = f1_score(test_y, test_preds)
        test_prec_arr, test_rec_arr, _ = precision_recall_curve(test_y, test_probas)
        test_pr_auc = auc(test_rec_arr, test_prec_arr)
        
        tn, fp, fn, tp = confusion_matrix(test_y, test_preds).ravel()
        
        print(f"\n--- {name} ---")
        print(f"  Tuned Val Threshold  : {best_t:.2f}")
        print(f"  Validation F1-Score  : {best_val_f1:.4f}")
        print(f"  Validation PR-AUC    : {val_pr_auc:.4f}")
        print(f"  Test F1-Score        : {test_f1:.4f}")
        print(f"  Test PR-AUC          : {test_pr_auc:.4f}")
        print(f"  Test Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        return {'model': name, 'val_f1': best_val_f1, 'test_f1': test_f1, 'test_pr_auc': test_pr_auc, 'fn': fn, 'fp': fp}

    res_lr = evaluate_model_pipeline("Logistic Regression (OFFICIAL SUBMISSION MODEL)", val_probas_lr, test_probas_lr, y_val, y_test)
    res_rf = evaluate_model_pipeline("Random Forest (Experimental Benchmark)", val_probas_rf, test_probas_rf, y_val, y_test)

    # Inspect Long-Tenure / Multi-Year Contract Misses for Random Forest
    rf_test_preds = (test_probas_rf >= 0.52).astype(int)
    rf_fn_long_tenure = ((test_df['tenure'] > 48) & (y_test == 1) & (rf_test_preds == 0)).sum()
    lr_test_preds = (test_probas_lr >= 0.57).astype(int)
    lr_fn_long_tenure = ((test_df['tenure'] > 48) & (y_test == 1) & (lr_test_preds == 0)).sum()
    
    print("\n--- SUBGROUP COMPARISON (Long-Tenure Customer Misses > 48 Mos) ---")
    print(f"  Logistic Regression Missed Churners (FN): {lr_fn_long_tenure} out of 26")
    print(f"  Random Forest Missed Churners (FN)      : {rf_fn_long_tenure} out of 26")

if __name__ == '__main__':
    compare_model_improvements()
