# Customer Churn Prediction

An end-to-end machine learning project predicting customer churn for a telecom company, from raw data through a deployed, usable application.

## Overview

This project predicts which customers are likely to churn (cancel their service), using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data, ~7,000 customers). The goal isn't just a trained model — it's a complete, working system: an API that serves predictions and a dashboard that business users could actually use.

## Results

The final model (tuned XGBoost) achieves:
- **ROC-AUC: 0.844**
- **Recall: 79.1%** (catches ~4 in 5 actual churners)
- **F1: 0.628**

Chosen over a marginally-higher-F1 Random Forest because recall — catching actual churners — matters most for this business problem. A business-cost analysis (assigning dollar values to missed churners vs. false alarms) showed roughly an **$80,000 lower net cost** on the test set versus the baseline model, since a missed churner is far more expensive than a false retention outreach.

## Key Findings

- **Contract type** is the single strongest churn driver: month-to-month customers churn ~4x more than two-year contract customers.
- **Tenure** matters heavily: churn is concentrated in a customer's first few months.
- **Fiber optic internet** and **electronic check payment** are both strong risk factors.
- These findings were consistent across exploratory data analysis, Logistic Regression coefficients, and SHAP values on the final model.

## Project Structure
├── notebooks/ # EDA, preprocessing, feature engineering, modeling, interpretation, business impact
├── api/ # FastAPI service that serves predictions
├── dashboard/ # Streamlit app (form, batch scoring, insights)
├── data/ # Raw + processed data (not tracked in git)
├── models/ # Trained model + preprocessing artifacts (not tracked in git)
└── requirements.txt


## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · Streamlit

## Running Locally

1. Clone the repo and install dependencies: pip install -r requirements.txt
2. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into `data/`.
3. Run the notebooks in order (`01` through `08`) to reproduce preprocessing, training, and saved artifacts.
4. Start the API: uvicorn api.main:app --reload
5. In a separate terminal, start the dashboard: streamlit run dashboard/app.py


## Known Limitations

- The test set was reused across model comparison and final evaluation, which likely makes the reported metrics slightly optimistic versus a fully held-out test set.
- Business-cost figures in the impact analysis are reasonable estimates, not measured company data.