from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List

app = FastAPI(title="Customer Churn Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

model = joblib.load(os.path.join(MODELS_DIR, "final_model_xgboost.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])
    input_encoded = pd.get_dummies(input_df, columns=[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ])

    input_encoded['AvgMonthlySpend'] = input_encoded['TotalCharges'] / (input_encoded['tenure'] + 1)

    tenure_val = input_encoded['tenure'].iloc[0]
    if tenure_val <= 12:
        tenure_group = '0-12mo'
    elif tenure_val <= 24:
        tenure_group = '13-24mo'
    elif tenure_val <= 48:
        tenure_group = '25-48mo'
    else:
        tenure_group = '49mo+'

    for col in ['TenureGroup_13-24mo', 'TenureGroup_25-48mo', 'TenureGroup_49mo+']:
        input_encoded[col] = 1 if col == f'TenureGroup_{tenure_group}' else 0

    service_cols = ['OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
                     'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
    input_encoded['NumServices'] = sum(
        input_encoded[c].iloc[0] if c in input_encoded.columns else 0 for c in service_cols
    )

    input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    input_final[numeric_cols] = scaler.transform(input_final[numeric_cols])

    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]


@app.post("/predict/batch")
def predict_batch(batch: BatchCustomerData):
    results = []
    for customer in batch.customers:
        result = predict(customer)
        results.append(result)
    return {"predictions": results}