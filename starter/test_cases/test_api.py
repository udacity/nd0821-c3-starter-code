from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_say_hello():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"greeting": "Hello World!"}


def test_predict_below():
    data = {
            "age": 25,
            "workclass": "Self-emp-not-inc",
            "fnlwgt": 176756,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Farming-fishing",
            "relationship": "Own-child",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 35,
            "native_country": "United-States"
        }
    request = client.post('/predict', json=data)
    assert request.status_code == 200
    assert request.json() == {"Prediction": "<=50K"}


def test_predict_over():
    data = {
            "age": 29,
            "workclass": "Self-emp-not-inc",
            "fnlwgt": 162298,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Sales",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 70,
            "native_country": "United-States"
        }
    request = client.post('/predict', json=data)
    assert request.status_code == 200
    assert request.json() == {"Prediction": ">50K"}


def test_bad_request():
    data = {
            "age": 29,
            "workclass": "",
            "fnlwgt": 162298,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Sales",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 70,
            "native_country": "United-States"
        }
    request = client.post('/predict', json=data)
    assert request.status_code == 422


def test_get_malformed():
    resp = client.get("/predict")
    assert resp.status_code != 200
