import streamlit as st
import requests
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

API_URL = "http://127.0.0.1:8000"


@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("Customer Churn Prediction Dashboard")

st.header("Predict Churn for a Single Customer")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col3:
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)

total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly_charges * tenure))

if st.button("Predict Churn", type="primary"):
    payload = {
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        prob = result["churn_probability"]
        prediction = result["churn_prediction"]

        if prediction:
            st.error(f"⚠️ High Churn Risk — Probability: {prob:.1%}")
        else:
            st.success(f"✅ Low Churn Risk — Probability: {prob:.1%}")

        st.progress(prob)
    else:
        st.error(f"API Error: {response.text}")

st.header("Batch Prediction from CSV")
st.write("Upload a CSV with the same columns as the training data (no customerID or Churn column needed).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(batch_df)} customers")
    st.dataframe(batch_df.head())

    if st.button("Run Batch Prediction"):
        customers_list = batch_df.to_dict(orient="records")
        batch_payload = {"customers": customers_list}

        with st.spinner("Scoring customers..."):
            response = requests.post(f"{API_URL}/predict/batch", json=batch_payload)

        if response.status_code == 200:
            predictions = response.json()["predictions"]
            results_df = batch_df.copy()
            results_df["churn_probability"] = [p["churn_probability"] for p in predictions]
            results_df["churn_prediction"] = [p["churn_prediction"] for p in predictions]
            results_df = results_df.sort_values("churn_probability", ascending=False)

            st.subheader("Results (sorted by risk)")
            st.dataframe(results_df)

            high_risk_count = results_df["churn_prediction"].sum()
            st.metric("High-Risk Customers", f"{high_risk_count} / {len(results_df)}")

            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv_output, "churn_predictions.csv", "text/csv")
        else:
            st.error(f"API Error: {response.text}")

st.header("Churn Insights (Historical Data)")

raw_df = load_raw_data()

tab1, tab2, tab3 = st.tabs(["Churn by Contract", "Churn by Internet Service", "Overall Churn Rate"])

with tab1:
    contract_churn = pd.crosstab(raw_df['Contract'], raw_df['Churn'], normalize='index')
    st.bar_chart(contract_churn)

with tab2:
    internet_churn = pd.crosstab(raw_df['InternetService'], raw_df['Churn'], normalize='index')
    st.bar_chart(internet_churn)

with tab3:
    churn_rate = raw_df['Churn'].value_counts(normalize=True)
    st.bar_chart(churn_rate)
    st.metric("Overall Churn Rate", f"{churn_rate['Yes']:.1%}")