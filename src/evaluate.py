import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, precision_recall_curve, auc
)
from preprocess import get_feature_lists

def run_evaluation():
    """
    Evaluates trained model pipeline:
    1. Performs threshold tuning on Validation set (X_val) to maximize F1-score.
    2. Evaluates selected threshold ONCE on untouched Test set (X_test).
    3. Saves confusion matrix and PR curve plots into results/figures/.
    """
    figures_dir = os.path.join("results", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    processed_dir = os.path.join("data", "processed")
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    num_cols, cat_cols, _, target_col = get_feature_lists()
    feature_cols = num_cols + cat_cols
    
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    model_path = os.path.join("src", "models", "logistic_regression_pipeline.joblib")
    model_pipeline = joblib.load(model_path)
    
    # -------------------------------------------------------------
    # STEP 1: Threshold Tuning on Validation Set (X_val ONLY)
    # -------------------------------------------------------------
    val_probas = model_pipeline.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.50
    best_val_f1 = 0.0
    
    thresholds = np.linspace(0.10, 0.90, 81)
    threshold_results = []
    
    for t in thresholds:
        preds = (val_probas >= t).astype(int)
        f1 = f1_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        threshold_results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = t
            
    thresh_df = pd.DataFrame(threshold_results)
    
    # Compute PR-AUC on Validation
    val_prec_arr, val_rec_arr, _ = precision_recall_curve(y_val, val_probas)
    val_pr_auc = auc(val_rec_arr, val_prec_arr)
    
    print("="*60)
    print("PHASE 7: VALIDATION THRESHOLD TUNING")
    print("="*60)
    print(f"Default (0.50) Val F1-Score : {f1_score(y_val, (val_probas >= 0.50).astype(int)):.4f}")
    print(f"Optimal Threshold Selected  : {best_threshold:.2f}")
    print(f"Optimal Val F1-Score        : {best_val_f1:.4f}")
    print(f"Validation PR-AUC           : {val_pr_auc:.4f}")
    
    # -------------------------------------------------------------
    # STEP 2: Final Unbiased Evaluation on Test Set (X_test)
    # -------------------------------------------------------------
    test_probas = model_pipeline.predict_proba(X_test)[:, 1]
    test_preds_default = (test_probas >= 0.50).astype(int)
    test_preds_tuned = (test_probas >= best_threshold).astype(int)
    
    test_prec_arr, test_rec_arr, _ = precision_recall_curve(y_test, test_probas)
    test_pr_auc = auc(test_rec_arr, test_prec_arr)
    
    test_f1_default = f1_score(y_test, test_preds_default)
    test_f1_tuned = f1_score(y_test, test_preds_tuned)
    test_prec_tuned = precision_score(y_test, test_preds_tuned)
    test_rec_tuned = recall_score(y_test, test_preds_tuned)
    test_acc_tuned = accuracy_score(y_test, test_preds_tuned)
    
    cm = confusion_matrix(y_test, test_preds_tuned)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*60)
    print("FINAL UNBIASED TEST SET EVALUATION (X_test)")
    print("="*60)
    print(f"Test PR-AUC                 : {test_pr_auc:.4f}")
    print(f"Test F1-Score (Default 0.50): {test_f1_default:.4f}")
    print(f"Test F1-Score (Tuned {best_threshold:.2f}): {test_f1_tuned:.4f}")
    print(f"Test Precision              : {test_prec_tuned:.4f}")
    print(f"Test Recall                 : {test_rec_tuned:.4f}")
    print(f"Test Accuracy (Supplementary): {test_acc_tuned:.4f}")
    
    print("\nConfusion Matrix (Test Set at tuned threshold):")
    print(f"  True Negatives  (TN) = {tn} (Correct Retained)")
    print(f"  False Positives (FP) = {fp} (Incorrectly Flagged Churned)")
    print(f"  False Negatives (FN) = {fn} (Missed Churners!)")
    print(f"  True Positives  (TP) = {tp} (Correctly Caught Churned)")
    
    print("\nFull Classification Report (Test Set):")
    print(classification_report(y_test, test_preds_tuned, target_names=['Retained (0)', 'Churned (1)']))
    
    # -------------------------------------------------------------
    # STEP 3: Save Plots
    # -------------------------------------------------------------
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (0)', 'Churned (1)'],
                yticklabels=['Retained (0)', 'Churned (1)'])
    plt.title(f'Confusion Matrix (Test Set, Threshold = {best_threshold:.2f})', fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrix_test.png"), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curve Plot
    plt.figure(figsize=(7, 5))
    plt.plot(test_rec_arr, test_prec_arr, color='#2980b9', lw=2, label=f'Logistic Regression (PR-AUC = {test_pr_auc:.3f})')
    plt.axhline(y=y_test.mean(), color='red', linestyle='--', label=f'Baseline (No Skill = {y_test.mean():.3f})')
    plt.scatter(test_rec_tuned, test_prec_tuned, color='darkorange', s=100, zorder=5, label=f'Tuned Threshold ({best_threshold:.2f})')
    plt.title('Precision-Recall Curve (Test Set)', fontweight='bold')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pr_curve_test.png"), dpi=300)
    plt.close()
    
    print(f"Saved test evaluation figures to {figures_dir} successfully.")

if __name__ == '__main__':
    run_evaluation()
