from fastapi.testclient import TestClient
import os
import json
from starter.api import app


client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data


def make_payload(age):
    return {
        "age": age,
        "workclass": "Private",
        "fnlgt": 12345,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


def test_post_predict_lower():
    # Use deterministic fallback rule during tests
    os.environ["DETERMINISTIC"] = "1"
    payload = make_payload(age=30)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_post_predict_higher():
    os.environ["DETERMINISTIC"] = "1"
    payload = make_payload(age=70)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"
