from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_root() -> None:
    """Test the GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}


# Example that should predict "<=50K" (or class 0)
low_income_payload = {
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
    "native-country": "United-States",
}


# Example that should predict ">50K" (or class 1)
high_income_payload = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def test_predict_low_income() -> None:
    """POST should predict class for a low-income example."""
    response = client.post("/predict", json=low_income_payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction == "<=50K"


def test_predict_high_income() -> None:
    """POST should predict class for a high-income example."""
    response = client.post("/predict", json=high_income_payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction == ">50K"
