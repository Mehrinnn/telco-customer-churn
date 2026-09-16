# Telco Customer Churn Prediction

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Project-Complete-brightgreen.svg)]()

Industrial-grade end-to-end machine learning solution for predicting customer churn using the **IBM Telco Customer Churn** dataset, built for the **ACM SIG AI Recruitment Task**.

---

## 📌 Executive Summary

Customer acquisition costs in telecommunications ($500–$700 per user) significantly outweigh customer retention costs. Predicting high-risk customers allows proactive intervention via targeted retention offers. 

This repository implements a modular, reproducible **Logistic Regression** pipeline featuring:
- **Strict Data Leakage Prevention**: Encapsulated `ColumnTransformer` & `Pipeline` fitted exclusively on training data.
- **Stratified 70 / 15 / 15 Splitting**: Preserves class proportions (~26.5% positive churn) across Train, Validation, and Test sets.
- **Validation Threshold Tuning**: Decision threshold optimized on validation data ($\tau = 0.57$) to maximize F1-score.
- **Subgroup Error Analysis**: Quantitative evaluation of group-level False Positive and False Negative rates across contract types, internet services, and tenure tiers.

---

## 📁 Repository Structure

```text
telco-customer-churn/
│
├── README.md                    # Project overview, installation, and reproducibility guide
├── requirements.txt             # Python dependencies
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Raw dataset
│   └── processed/               # Stratified train, val, and test split CSVs
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   └── telco_customer_churn.ipynb # Interactive end-to-end Jupyter Notebook
├── src/
│   ├── load_and_verify.py       # Phase 2: Directory setup & raw data verification
│   ├── eda.py                   # Phase 3: Exploratory Data Analysis & figure generation
│   ├── preprocess.py            # Phase 4: Scikit-learn ColumnTransformer & Pipeline
│   ├── data_split.py            # Phase 5: Stratified 70/15/15 train/val/test splitter
│   ├── train.py                 # Phase 6: Model training & pipeline artifact persistence
│   ├── evaluate.py              # Phase 7: Validation threshold tuning & test set evaluation
│   ├── error_analysis.py        # Phase 8: Subgroup error rate analysis
│   ├── model_improvements.py    # Phase 9: Random Forest comparative benchmark
│   └── models/
│       └── logistic_regression_pipeline.joblib # Saved trained model pipeline
├── reports/
│   └── one_page_writeup.md      # Official 1-page executive technical write-up
└── results/
    └── figures/                 # Evaluation plots (Confusion Matrix, PR Curve, EDA figures)
        ├── numerical_distributions.png
        ├── categorical_churn_rates.png
        ├── confusion_matrix_test.png
        └── pr_curve_test.png
```

---

## 🚀 Installation & Environment Setup

### 1. Prerequisites
- Python `3.10+` (Tested on Python `3.13.5`)

### 2. Install Dependencies
```bash
git clone https://github.com/your-username/telco-customer-churn.git
cd telco-customer-churn
pip install -r requirements.txt
```

---

## 🛠️ How to Run the Code

You can run the entire pipeline end-to-end via Python scripts or interactively in Jupyter:

### Option A: Execute Scripts Sequentially
```bash
# 1. Dataset verification and setup
python src/load_and_verify.py

# 2. Run Exploratory Data Analysis (Generates EDA plots in results/figures/)
python src/eda.py

# 3. Create Stratified 70/15/15 Splits (Saves to data/processed/)
python src/data_split.py

# 4. Train Logistic Regression Pipeline (Saves artifact to src/models/)
python src/train.py

# 5. Perform Validation Threshold Tuning & Test Evaluation (Generates Test plots)
python src/evaluate.py

# 6. Run Subgroup Error Analysis
python src/error_analysis.py

# 7. Compare against Non-Linear Baseline (Random Forest)
python src/model_improvements.py
```

### Option B: Interactive Jupyter Notebook
Launch the notebook to inspect interactive visualizations and step-by-step commentary:
```bash
jupyter notebook notebooks/telco_customer_churn.ipynb
```

---

## 📊 Key Results & Evaluation (Test Set `X_test`)

Evaluated at the validation-tuned threshold ($\tau = 0.57$):

| Metric | Score | Note / Benchmark |
| :--- | :---: | :--- |
| **PR-AUC** | **0.6394** | Outperforms random baseline ($0.2650$) |
| **F1-Score** | **0.6287** | Harmonic mean of Precision & Recall |
| **Recall (Sensitivity)** | **75.00%** | Catches **210 out of 280** actual churners |
| **Precision** | **54.12%** | Positive predictive value |
| **Accuracy (Supplementary)** | **76.54%** | Total overall correct predictions |

### Test Confusion Matrix ($\tau = 0.57$)
```text
                  Predicted Retained (0)    Predicted Churned (1)
Actual Retained (0)       TN = 599                 FP = 178  (False Alarms)
Actual Churned  (1)       FN = 70  (Missed)        TP = 210  (Caught)
```

---

## 🔍 Subgroup Error Analysis Summary

Group-level error analysis revealed two operational performance profiles:
1. **Month-to-Month & Fiber Optic Users**: High Sensitivity (**83.8% Recall**), but high False Positive Rate (**52.1% FPR**). Strong positive feature weights flag many loyal customers unnecessarily.
2. **Two-Year Contract & Long-Tenure Users (>48 Mos)**: High False Negative Rate (**100% FNR** on 2-Year contracts; **69.2% FNR** on long-tenure). Heavy negative linear weights overpower subtle churn signals in low-churn sub-populations.

---

## 🔮 Future Roadmap (1-Week Extension Plan)
1. **Behavioral Feature Engineering**: Create ratio attributes like `Charges_per_Tenure` ($\frac{\text{TotalCharges}}{\text{tenure}+1}$) and explicit interaction terms (`tenure` $\times$ `MonthlyCharges`).
2. **Non-Linear Tree Models**: Integrate **LightGBM** / **XGBoost** with `Optuna` hyperparameter tuning.
3. **Financial Utility Tuning**: Optimize threshold against an asymmetric financial loss function ($C_{\text{FP}} = \$50$ retention offer vs $C_{\text{FN}} = \$600$ lost customer LTV).

---

## 📄 License & Attribution
- **Dataset**: IBM Telco Customer Churn (Available on Kaggle / IBM Community).
- **Project**: ACM SIG AI Recruitment Task submission.
