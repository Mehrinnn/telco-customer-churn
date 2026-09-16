import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import clean_raw_data, get_feature_lists

def create_stratified_splits(random_state=42):
    """
    Loads dataset, cleans raw data, drops customerID, and performs a 70/15/15 stratified split.
    Saves train, val, and test dataframes into data/processed/.
    """
    raw_csv = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(raw_csv)
    df_clean = clean_raw_data(df)
    
    num_cols, cat_cols, id_col, target_col = get_feature_lists()
    
    # Feature matrix X (excluding customerID and target) and target y
    X = df_clean[num_cols + cat_cols]
    y = df_clean[target_col]
    
    # Step 1: Split off 15% test set (85% train_val, 15% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    
    # Step 2: Split 85% train_val into 70% train and 15% val
    # Relative fraction: 0.15 / 0.85 ≈ 0.17647
    val_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_fraction, stratify=y_train_val, random_state=random_state
    )
    
    # Save processed splits to disk
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Re-attach target y to X for saving clean dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print("="*60)
    print("PHASE 5: DATA SPLITTING SUMMARY (70% Train / 15% Val / 15% Test)")
    print("="*60)
    
    total_len = len(df_clean)
    for name, split_y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        count = len(split_y)
        pct_of_total = (count / total_len) * 100
        churn_count = split_y.sum()
        churn_pct = split_y.mean() * 100
        print(f"\n{name} Split:")
        print(f"  - Rows      : {count} ({pct_of_total:.2f}% of full dataset)")
        print(f"  - Retained  : {(split_y == 0).sum()} ({(100 - churn_pct):.2f}%)")
        print(f"  - Churned   : {churn_count} ({churn_pct:.2f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    create_stratified_splits()
