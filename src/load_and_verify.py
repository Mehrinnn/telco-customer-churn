import os
import shutil
import pandas as pd
import numpy as np

# Step 1: Create project directory structure
dirs = ['data', 'notebooks', 'src', 'reports', 'results/figures']
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Directory ready: {d}")

# Step 2: Copy dataset into data/
src_csv = r"C:\Users\Mehrin\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
dst_csv = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
if os.path.exists(src_csv) and not os.path.exists(dst_csv):
    shutil.copy(src_csv, dst_csv)
    print(f"Copied dataset to {dst_csv}")
elif os.path.exists(dst_csv):
    print(f"Dataset already exists at {dst_csv}")

# Step 3: Load dataset
df = pd.read_csv(dst_csv)

# Step 4: Verification
print("\n" + "="*50)
print("DATASET VERIFICATION SUMMARY")
print("="*50)
print(f"1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n2. Column Names & Data Types:")
print(df.dtypes)

print("\n3. Standard Missing Values (NaN/null):")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() > 0 else "No standard NaN/null values found.")

print("\n4. Checking for hidden whitespace/blank strings:")
blank_found = False
for col in df.select_dtypes(include='object').columns:
    blank_count = (df[col].astype(str).str.strip() == '').sum()
    if blank_count > 0:
        print(f"   -> Column '{col}': {blank_count} blank/whitespace entries found!")
        blank_found = True
if not blank_found:
    print("   No blank strings found in object columns.")

print("\n5. Target Column ('Churn') Distribution:")
val_counts = df['Churn'].value_counts()
val_pcts = df['Churn'].value_counts(normalize=True) * 100
target_df = pd.DataFrame({'Count': val_counts, 'Percentage (%)': val_pcts.round(2)})
print(target_df)

print("\n6. First 3 Rows Preview:")
print(df.head(3))
