import os
import joblib
import pandas as pd
import numpy as np

from preprocess import get_feature_lists

def run_error_analysis():
    """
    Performs rigorous subgroup error analysis on the Test set (X_test).
    Calculates False Positive Rates (FPR), False Negative Rates (FNR), Precision, 
    and Recall within customer subgroups.
    """
    processed_dir = os.path.join("data", "processed")
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    num_cols, cat_cols, _, target_col = get_feature_lists()
    feature_cols = num_cols + cat_cols
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    model_path = os.path.join("src", "models", "logistic_regression_pipeline.joblib")
    model_pipeline = joblib.load(model_path)
    
    # Predict probabilities and apply tuned threshold (0.57)
    test_probas = model_pipeline.predict_proba(X_test)[:, 1]
    best_threshold = 0.57
    test_preds = (test_probas >= best_threshold).astype(int)
    
    # Attach predictions and error categorization to test dataframe
    analysis_df = test_df.copy()
    analysis_df['Predicted_Churn'] = test_preds
    analysis_df['Churn_Probability'] = test_probas.round(4)
    
    # Categorize error types
    # TN: y=0, pred=0 | FP: y=0, pred=1 | FN: y=1, pred=0 | TP: y=1, pred=1
    conditions = [
        (analysis_df['Churn'] == 0) & (analysis_df['Predicted_Churn'] == 0),
        (analysis_df['Churn'] == 0) & (analysis_df['Predicted_Churn'] == 1),
        (analysis_df['Churn'] == 1) & (analysis_df['Predicted_Churn'] == 0),
        (analysis_df['Churn'] == 1) & (analysis_df['Predicted_Churn'] == 1)
    ]
    choices = ['TN', 'FP', 'FN', 'TP']
    analysis_df['Error_Type'] = np.select(conditions, choices, default='Unknown')
    
    # Create Tenure Bins for granular analysis
    analysis_df['Tenure_Group'] = pd.cut(
        analysis_df['tenure'], 
        bins=[-1, 12, 24, 48, 72], 
        labels=['0-12 Mos', '13-24 Mos', '25-48 Mos', '49-72 Mos']
    )
    
    print("="*70)
    print("PHASE 8: SUBGROUP ERROR ANALYSIS (TEST SET)")
    print("="*70)
    print(f"Total Test Samples: {len(analysis_df)}")
    print(f"Overall Error Breakdown: {analysis_df['Error_Type'].value_counts().to_dict()}\n")
    
    def analyze_subgroup(feature_name):
        print(f"--- ERROR RATES BY SUBGROUP: {feature_name} ---")
        groups = analysis_df.groupby(feature_name)
        results = []
        
        for name, group in groups:
            total = len(group)
            actual_retained = (group['Churn'] == 0).sum()
            actual_churned = (group['Churn'] == 1).sum()
            
            fp = (group['Error_Type'] == 'FP').sum()
            fn = (group['Error_Type'] == 'FN').sum()
            tp = (group['Error_Type'] == 'TP').sum()
            tn = (group['Error_Type'] == 'TN').sum()
            
            # False Positive Rate (FPR) = FP / Actual Retained
            fpr = (fp / actual_retained * 100) if actual_retained > 0 else 0.0
            # False Negative Rate (FNR) = FN / Actual Churned
            fnr = (fn / actual_churned * 100) if actual_churned > 0 else 0.0
            # Group Recall = TP / Actual Churned
            group_recall = (tp / actual_churned * 100) if actual_churned > 0 else 0.0
            # Group Precision = TP / (TP + FP)
            group_prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            
            results.append({
                'Group': name,
                'Total': total,
                'Actual Churned': actual_churned,
                'Actual Retained': actual_retained,
                'FP (False Alarms)': fp,
                'FN (Missed Churn)': fn,
                'FPR (%)': round(fpr, 2),
                'FNR (%)': round(fnr, 2),
                'Recall (%)': round(group_recall, 2),
                'Precision (%)': round(group_prec, 2)
            })
            
        res_df = pd.DataFrame(results).sort_values(by='Total', ascending=False)
        print(res_df.to_string(index=False))
        print("\n")
        return res_df

    # Analyze key subgroups
    analyze_subgroup('Contract')
    analyze_subgroup('InternetService')
    analyze_subgroup('PaymentMethod')
    analyze_subgroup('Tenure_Group')

if __name__ == '__main__':
    run_error_analysis()
