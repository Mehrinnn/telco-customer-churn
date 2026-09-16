# Executive Technical Report: Telco Customer Churn Prediction

**Project**: ACM SIG AI Recruitment Task — Telco Customer Churn Prediction  
**Dataset**: IBM Telco Customer Churn (7,043 rows, 21 columns)  
**Primary Model**: Logistic Regression with Balanced Class Weighting (`scikit-learn`)  
**Repository Structure**: Industrial-grade `src/`, `data/`, `notebooks/`, `reports/`, `results/` structure  

---

## 1. Problem Statement & Business Context
Customer churn represents the rate at which existing subscribers discontinue their services. In the telecommunications industry, customer acquisition costs ($500–$700 per user) far exceed retention costs. Predicting high-risk customers allows proactive intervention via targeted retention offers. We formulate this task as a **binary classification problem** to predict the target variable `Churn` (`Yes` / `No`).

---

## 2. Dataset Profile & Target Imbalance
- **Dataset Size**: 7,043 records, 20 feature columns, 1 binary target (`Churn`).
- **Class Distribution**: 5,174 Retained (`No` = 73.46%) vs. 1,869 Churned (`Yes` = 26.54%).
- **Class Imbalance Impact**: Because ~73.5% of customers do not churn, naive accuracy is a misleading metric (a dummy model predicting `No` achieves 73.5% accuracy but catches 0% of churners). Evaluation prioritizes **F1-Score**, **PR-AUC**, **Precision**, and **Recall**.

---

## 3. Exploratory Data Analysis (EDA) Highlights
1. **Contract Type**: Month-to-month contract holders churn at **42.71%**, compared to 11.27% for 1-year and **2.83%** for 2-year contract holders.
2. **Internet Service & Charges**: Fiber optic subscribers churn at **41.89%** (mean monthly charge of \$74.44 vs \$61.27 for retained customers).
3. **Tenure Dynamics**: Mean tenure for retained customers is **37.57 months** vs **17.98 months** for churned customers; risk is highest in months 0–12.
4. **Payment Method**: Electronic check users exhibit a **45.29% churn rate**, compared to ~16% for automated payments.

---

## 4. Preprocessing Decisions & Anti-Leakage Justification
- **Identifier Exclusion**: Dropped `customerID` to prevent high-cardinality memorization.
- **Data Type Coercion**: Converted `TotalCharges` from `object` to `float64`, exposing 11 missing values (`" "`) belonging to new customers (`tenure == 0`).
- **Pipeline Architecture**: Wrapped all transformations in a `ColumnTransformer`:
  - *Numerical Features* (`tenure`, `MonthlyCharges`, `TotalCharges`): Median Imputation + `StandardScaler` (standardizes features to $\mu=0, \sigma=1$ for gradient stability and regularized linear modeling).
  - *Categorical Features* (16 features): Mode Imputation + `OneHotEncoder(handle_unknown='ignore')` (expands categories into 43 binary indicator variables).
- **Anti-Leakage Guarantee**: Transformers are fitted **strictly on `X_train`**; parameter states ($\mu, \sigma$, categories) are applied onto validation and test splits without leaking out-of-sample data.

---

## 5. Model Selection & Class Weighting Rationale
We selected **Logistic Regression** as our primary model:
- **Interpretability**: Linear coefficients ($w_i$) map directly to log-odds of churn, enabling transparent business decision-making.
- **Calibrated Probabilities**: Produces smooth probability predictions $P(\text{Churn}=1 \mid X) \in [0, 1]$.
- **Class Weight Balancing (`class_weight='balanced'`)**: Re-weights loss inversely proportional to class frequencies ($w_j = \frac{N}{2 N_j}$). This penalizes missed churners $\approx 1.88\times$ more than false alarms, shifting the decision threshold to boost **Recall** (catching 75% of churners).

---

## 6. Train / Validation / Test Splitting Methodology
- **Split Ratio**: 70% Training (4,929 rows) / 15% Validation (1,057 rows) / 15% Test (1,057 rows).
- **Stratified Sampling (`stratify=y`)**: Guaranteed exact target proportions across all splits (~26.5% Churned, ~73.5% Retained).
- **Vault Discipline**: Validation data (`X_val`) was used exclusively for threshold tuning; the Test set (`X_test`) remained completely untouched until final evaluation.

---

## 7. Final Model Evaluation Metrics (Test Set: `X_test`)

Threshold tuning on validation data selected an optimal decision threshold of **$\tau = 0.57$**. The final unbiased test set results are:

| Metric | Score | Benchmark / Interpretation |
| :--- | :---: | :--- |
| **PR-AUC** | **0.6394** | Significantly outperforms no-skill baseline (0.2650) |
| **F1-Score** | **0.6287** | Tuned threshold improves default 0.50 F1-score (0.6245) |
| **Recall (Sensitivity)** | **75.00%** | Correctly catches **210 out of 280** actual churners |
| **Precision** | **54.12%** | 54.1% of flagged customers actually churned |
| **Accuracy (Supplementary)** | **76.54%** | Total correct predictions over full test set |

---

## 8. Confusion Matrix Breakdown (Test Set, $\tau = 0.57$)

```text
                  Predicted Retained (0)    Predicted Churned (1)
Actual Retained (0)       TN = 599                 FP = 178  (False Alarms)
Actual Churned  (1)       FN = 70  (Missed)        TP = 210  (Caught)
```

- **True Negatives (TN = 599)**: Retained customers correctly classified.
- **True Positives (TP = 210)**: Churned customers successfully detected.
- **False Positives (FP = 178)**: Retained customers falsely flagged. (Low cost: receiving a proactive retention promotion does not harm customer relationship).
- **False Negatives (FN = 70)**: High-risk churners missed. (High cost: customer leaves the company permanently).

---

## 9. Subgroup Error Analysis: "Where Does the Model Break?"
Group-level error analysis revealed two major operational failure modes:
1. **High False Alarm Rate on Short-Tenure & Month-to-Month Profiles (FPR ~47–52%)**:
   - For Month-to-month contract holders, False Positive Rate is **52.08%** (175 loyal customers falsely flagged). Strong positive feature weights ($w = +0.72$) push almost all short-term users above the threshold.
2. **Severe Blind Spot on Long-Tenure & Multi-Year Contracts (FNR ~70–100%)**:
   - For 2-Year contract holders, False Negative Rate is **100.0%** (missed 5 out of 5 churners).
   - For Long-Tenure users (>48 months), False Negative Rate is **69.23%** (missed 18 out of 26 churners).
   - *Technical Cause*: Large negative feature weights (`tenure` $w = -1.16$, `Contract_Two year` $w = -0.79$) pull linear log-odds scores down so heavily that even when a long-term customer churns, the score remains below 0.57.

---

## 10. Technical & Methodological Limitations
1. **Linear Boundary Constraints**: Logistic regression cannot capture complex non-linear feature interactions without manual interaction engineering.
2. **Static Snapshot Data**: The dataset lacks temporal longitudinal tracking (e.g., trend in monthly billing changes over time).
3. **Missing Critical Behavioral Features**: Lacks domain signals such as customer service call logs, network downtime records, or competitor pricing shifts.

---

## 11. 1-Week Future Improvement Roadmap
If granted one additional week, we would implement the following high-priority enhancements:
1. **Behavioral Feature Engineering**: Construct domain features such as `Charges_per_Tenure` ($\frac{\text{TotalCharges}}{\text{tenure} + 1}$), `Addon_Service_Count` (sum of security/backup/support services), and explicit non-linear interaction terms (`tenure` $\times$ `MonthlyCharges`).
2. **Tree-Based Ensembles**: Upgrade pipeline classifier to **LightGBM** / **XGBoost** with Bayesian Hyperparameter Optimization (`Optuna`) to handle non-linear decision boundaries cleanly.
3. **Cost-Sensitive Threshold Optimization**: Replace generic F1-score threshold tuning with a custom business utility function based on financial metrics ($C_{\text{FP}} = \$50$ retention offer vs. $C_{\text{FN}} = \$600$ lost customer lifetime value).
