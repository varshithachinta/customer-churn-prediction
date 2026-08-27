from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Customer Churn Prediction API is running"}


def test_predict_high_risk_customer():
    payload = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1


def test_predict_missing_field_returns_422():
    incomplete_payload = {"gender": "Female"}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422

def test_predict_batch():
    payload = {
        "customers": [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        ]
    }

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 1

    prediction = data["predictions"][0]

    assert isinstance(prediction["churn_prediction"], bool)
    assert isinstance(prediction["churn_probability"], float)
    assert 0 <= prediction["churn_probability"] <= 1