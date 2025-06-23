from fastapi.testclient import TestClient
from main import app  # Adjust the import path to your main app

client = TestClient(app)

# Test 1: GET request to root
def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Income Prediction API!"}

# Test 2: POST request that predicts <=50K
def test_post_inference_low_income():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]

# Test 3: POST request that predicts >50K (use a high-income example)
def test_post_inference_high_income():
    payload = {
        "age": 52,
        "workclass": "Private",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 99999,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    response = client.post("/inference", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
