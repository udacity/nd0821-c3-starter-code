from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_welcome():
    request = client.get('/')
    assert request.status_code == 200
    assert request.content == b'"Welcome to the small vector machines for Adult income prediction endpoint"'


def test_prediction_1():
    response = client.post(
        "/infer",
        json={
            "age": 31,
            "workclass": "Local-gov",
            "fnlgt": 189265,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        },
    )
    assert response.status_code == 200
    assert response.content == b'"Predicted income: <=50K"'


def test_prediction_2():
    response = client.post(
        "/infer",
        json={
            "age": 31,
            "workclass": "Local-gov",
            "fnlgt": 189265,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 20000,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States",
        },
    )
    assert response.status_code == 200
    assert response.content == b'"Predicted income: >50K"'
