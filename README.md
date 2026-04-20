# Churn-Prediction
A classification based model on a telecommunication system that predicts customers that are more likely to leave.

# Telco Customer Churn Prediction

## Overview
This project builds a machine learning classification model to predict 
whether a telecom customer will churn (leave the service) based on their 
demographic information, account details, and subscribed services. 
The goal is to help the business identify high-risk customers early 
so retention campaigns can be targeted effectively.

## Dataset
- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers, 21 features
- **Target variable:** Churn (0 = Stays, 1 = Leaves)
- **Class distribution:** 73% stay, 27% churn — imbalanced dataset

## Project Structure
├── telco_churn_prediction.ipynb    # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── model_comparison.png            # Model comparison chart
├── feature_importance.png          # Feature importance chart
└── README.md                       # Project documentation

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Methodology

### 1. Exploratory Data Analysis
- Visualised churn distribution — confirmed 27% churn rate
- Analysed churn rates across all categorical features using countplots
- Computed correlation heatmap on numerical features
- Key EDA findings:
  - Month-to-month contract customers churn at ~43% vs only 3% for two-year contracts
  - Fiber optic internet customers churn more than DSL customers despite higher service cost
  - Electronic check payment customers churn significantly more than automatic payment customers
  - Customers without dependents or partners churn more than those with family commitments
  - Gender showed almost no difference in churn rate — not a useful predictor

### 2. Data Preprocessing
- Converted TotalCharges from string to numeric — found 11 blank values filled with median
- Dropped customerID — identifier column with no predictive value
- Encoded internet service columns explicitly mapping Yes=1, No=0, 
  No internet service=0
- Encoded MultipleLines mapping Yes=1, No=0, No phone service=0
- Encoded remaining binary columns using label encoding
- Applied one-hot encoding to Contract, InternetService and PaymentMethod

### 3. Feature Engineering Experiment
Created 7 new features — ChargesPerTenure, ChargesDifference, 
TotalStreamingServices, TotalValueServices, TotalServices, 
IsNewCustomer, IsLoyalCustomer — and tested whether they improved 
model performance.

Result: The engineered features produced marginally worse test ROC-AUC 
(0.8443 vs 0.8659) and increased variance across CV folds. The original 
features already captured the predictive information effectively. 
The baseline tuned model was retained as the final model.

This experiment demonstrates the importance of measuring feature 
engineering impact rather than assuming more features equals better performance.

### 4. Models Built
Three models were trained and evaluated in progression:
- Decision Tree (baseline — to demonstrate overfitting)
- Random Forest (default parameters — to demonstrate ensemble improvement)
- Random Forest with GridSearchCV tuning (final model)

### 5. Handling Class Imbalance
GridSearchCV identified class_weight='balanced' as an optimal parameter,
meaning the model automatically adjusts to give more weight to the 
minority class (churners) during training.

### 6. Hyperparameter Tuning
GridSearchCV with 5-fold cross validation tested 216 parameter 
combinations across:
- n_estimators: [100, 200, 300]
- max_depth: [None, 5, 10, 15]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 5]
- class_weight: ['balanced', None]

Best parameters found:
- n_estimators: 300
- max_depth: 10
- min_samples_leaf: 5
- min_samples_split: 2
- class_weight: balanced

## Results

### Model Progression
| Model | CV Accuracy | CV ROC-AUC | CV F1 | Train/Test Gap |
|---|---|---|---|---|
| Decision Tree | 0.7231 | 0.6545 | 0.4918 | 0.2725 (severe overfit) |
| Random Forest (default) | 0.7659 | 0.8423 | 0.6281 | Low |
| Random Forest (tuned) | — | 0.8424 | — | Low |

### Final Test Set Performance
| Metric | Score |
|---|---|
| Accuracy | 0.7842 |
| ROC-AUC | 0.8659 |
| Precision (churners) | 0.57 |
| Recall (churners) | 0.79 |
| F1 Score (churners) | 0.66 |

### Confusion Matrix
| | Predicted Stay | Predicted Churn |
|---|---|---|
| Actual Stay | 809 (TN) | 227 (FP) |
| Actual Churn | 77 (FN) | 296 (TP) |

The model correctly identifies 296 out of 373 actual churners (79% recall) 
— meaning the business can intervene on nearly 4 out of every 5 customers 
about to leave.

## Key Findings
- Tuned Random Forest achieved ROC-AUC of 0.8659 — strong performance 
  for a real world imbalanced classification problem
- Decision Tree severely overfitted (train accuracy 0.9986 vs test 0.7260 
  — gap of 0.2725) demonstrating why ensemble methods are needed
- Random Forest reduced the overfitting gap to near zero while improving 
  all metrics significantly
- Tenure is the strongest predictor of churn — longer tenure = much 
  lower churn risk
- TotalCharges and MonthlyCharges are the second and third most 
  important features
- Contract type is the most important categorical feature — two-year 
  contracts are the strongest indicator of customer retention
- Gender, PhoneService and some payment methods contributed near-zero 
  importance — confirming EDA observations

## Business Recommendations
Based on the model's findings the telecom company should:
- **Target new customers (tenure < 6 months)** with onboarding support 
  and loyalty incentives — highest churn risk period
- **Offer contract upgrades** to month-to-month customers — moving them 
  to annual contracts dramatically reduces churn probability
- **Investigate fiber optic pricing** — fiber customers churn more despite 
  paying more, suggesting a value perception problem
- **Promote automatic payment methods** — customers on electronic check 
  churn significantly more than those on automatic payments

## What I Would Do Next
- Apply SMOTE oversampling and compare against class_weight='balanced'
- Try XGBoost and LightGBM — gradient boosting models often outperform 
  Random Forest on tabular data
- Tune the decision threshold below 0.5 to further improve recall 
  on churners at the cost of some precision
- Deploy the model as a Streamlit web app where customer details can 
  be entered and churn probability returned in real time
- Explore SHAP values for individual prediction explanations — 
  useful for explaining to business stakeholders why a specific 
  customer is flagged as high risk

## Author
Daniel Abifarin
Electrical Engineering Student | University of Lagos
Aspiring MLOps Engineer
GitHub: github.com/Daniel-Abifarin
