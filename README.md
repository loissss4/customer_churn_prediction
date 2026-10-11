# Customer Churn Prediction

## Project Overview
An end-to-end machine learning project that predicts whether a telecom customer 
is likely to churn, based on their account, service, and billing information. 
Completed as a 5-week internship project covering the full ML workflow from 
business understanding through model optimization.

## Business Problem
Customer churn — when a customer stops using a company's service — is costly, 
since acquiring a new customer is far more expensive than retaining an existing 
one. This project builds a classification model to flag at-risk customers 
early, so a business can intervene with retention offers before they leave.

## Dataset
Telco Customer Churn dataset (Kaggle) — 7,043 customers, 21 original features, 
including demographics, account details, subscribed services, contract type, 
and billing information. Target variable: Churn (Yes/No).

## Tools and Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab

## Project Workflow
Business Understanding → Data Cleaning → Exploratory Data Analysis → Feature 
Engineering → Model Training → Evaluation → Imbalance Handling → Optimization 
→ Final Model → Business Interpretation

## Exploratory Data Analysis
Key findings: churn is concentrated in month-to-month contracts (55% of 
customers); tenure is the strongest predictor of retention (correlation 
−0.35); fiber optic internet subscribers churn more than DSL or no-internet 
customers; churned customers pay higher average monthly charges.

## Machine Learning Models
Classification: Logistic Regression, Decision Tree, Random Forest (model 
selected via cross-validation on the training set only).
Regression (supplementary): Linear Regression, Random Forest Regressor, 
predicting MonthlyCharges.

## Evaluation Metrics
Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
Regression: MAE, MSE, RMSE, R².

## Key Findings
Contract length and tenure are the strongest factors in customer retention; 
fiber optic internet is the strongest churn risk factor. The final model 
correctly identifies churned customers meaningfully better than a naive 
baseline, though recall remains a limitation (see below).

## Business Recommendations
1. Incentivize month-to-month customers to switch to annual contracts.
2. Prioritize retention outreach for customers in their first 12 months.
3. Investigate why fiber optic customers churn at a higher rate.

## Model Limitations
Recall of ~60% means the model misses a meaningful share of actual churners. 
The dataset is a single time snapshot and excludes external factors like 
competitor pricing or service complaints.

## How to Run the Project
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Open notebooks/customer_churn_analysis.ipynb in Jupyter or Google Colab
4. Run cells in order from top to bottom

## Project Structure
