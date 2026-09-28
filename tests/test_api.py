from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_root_returns_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Income Prediction API!"}


def test_post_predict_below_50k():
    person = {
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
    r = client.post("/predict", json=person)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predict_above_50k():
    person = {
        "age": 46,
        "workclass": "Self-emp-inc",
        "fnlgt": 192779,
        "education": "Prof-school",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=person)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_predict_rejects_missing_fields():
    r = client.post("/predict", json={"age": 39})
    assert r.status_code == 422
