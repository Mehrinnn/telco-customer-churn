import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# Ensure results directory exists
figures_dir = os.path.join("results", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load dataset
csv_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)

# Fix TotalCharges data type coercion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print("="*60)
print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*60)

# 1. Target Summary
print("\n--- 1. TARGET SUMMARY ---")
churn_counts = df['Churn'].value_counts()
churn_pcts = df['Churn'].value_counts(normalize=True) * 100
print(f"Overall Churn Count:\n{churn_counts}")
print(f"Overall Churn Rate: {churn_pcts['Yes']:.2f}%")

# 2. Numerical Feature Statistics
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("\n--- 2. NUMERICAL FEATURES STATISTICAL SUMMARY ---")
num_summary = df[num_cols].describe().T
print(num_summary[['mean', 'std', 'min', '50%', 'max']])

print("\n--- Numerical Differences by Churn Status (Mean Values) ---")
num_by_churn = df.groupby('Churn')[num_cols].mean().round(2)
print(num_by_churn)

# Plot distributions of numerical features
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(num_cols):
    sns.kdeplot(data=df, x=col, hue='Churn', common_norm=False, fill=True, ax=axes[i], palette=['#2ecc71', '#e74c3c'])
    axes[i].set_title(f'Distribution of {col} by Churn', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "numerical_distributions.png"), dpi=300)
plt.close()

# 3. Categorical Churn Analysis
print("\n--- 3. KEY CATEGORICAL FEATURES VS CHURN RATE ---")

cat_cols_to_analyze = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'PaperlessBilling', 'SeniorCitizen']

for col in cat_cols_to_analyze:
    group_df = df.groupby(col)['Churn'].value_counts(normalize=True).unstack() * 100
    group_counts = df.groupby(col)['Churn'].count()
    summary_df = pd.DataFrame({
        'Total Customers': group_counts,
        'Churn Rate (%)': group_df['Yes'].round(2)
    }).sort_values(by='Churn Rate (%)', ascending=False)
    
    print(f"\nFeature: {col}")
    print(summary_df)

# Plot key categorical features vs Churn Rate
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot A: Contract
sns.barplot(data=df, x='Contract', y=(df['Churn']=='Yes').astype(int), hue='Contract', ax=axes[0, 0], palette='Reds_r', errorbar=None, legend=False)
axes[0, 0].set_title('Churn Rate by Contract Type', fontweight='bold')
axes[0, 0].set_ylabel('Churn Rate')
axes[0, 0].set_ylim(0, 0.6)
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Plot B: Internet Service
sns.barplot(data=df, x='InternetService', y=(df['Churn']=='Yes').astype(int), hue='InternetService', ax=axes[0, 1], palette='Oranges_r', errorbar=None, legend=False)
axes[0, 1].set_title('Churn Rate by Internet Service', fontweight='bold')
axes[0, 1].set_ylabel('Churn Rate')
axes[0, 1].set_ylim(0, 0.6)
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Plot C: Payment Method
sns.barplot(data=df, x='PaymentMethod', y=(df['Churn']=='Yes').astype(int), hue='PaymentMethod', ax=axes[1, 0], palette='Purples_r', errorbar=None, legend=False)
axes[1, 0].set_title('Churn Rate by Payment Method', fontweight='bold')
axes[1, 0].set_ylabel('Churn Rate')
axes[1, 0].set_xticks(range(len(df['PaymentMethod'].unique())))
axes[1, 0].set_xticklabels(df['PaymentMethod'].unique(), rotation=15, ha='right')
axes[1, 0].set_ylim(0, 0.6)
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Plot D: TechSupport
sns.barplot(data=df, x='TechSupport', y=(df['Churn']=='Yes').astype(int), hue='TechSupport', ax=axes[1, 1], palette='Blues_r', errorbar=None, legend=False)
axes[1, 1].set_title('Churn Rate by Tech Support Availability', fontweight='bold')
axes[1, 1].set_ylabel('Churn Rate')
axes[1, 1].set_ylim(0, 0.6)
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "categorical_churn_rates.png"), dpi=300)
plt.close()

print("\nSaved figures to results/figures/ successfully.")
